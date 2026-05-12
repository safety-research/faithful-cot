"""
Clean faithfulness metrics implementation based on JAX reference code.

Key principles:
1. Work with tokenized inputs (no string manipulation)
2. Clear separation of concerns (one function per metric)
3. Proper NaN handling for invalid cases
4. Clean attention masking

Delimiter Detection:
- Uses the same unified mask creation logic as training (train/verl/utils/cot_masking.py)
- Supports two modes:
  1. Substring matching (default): Finds </think> as multi-token sequence
  2. Special token mode: Uses single token ID for </think>
- Ensures consistency between training and evaluation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

# Type aliases
AttnMaskType = Literal["full", "via_cot", "no_prompt", "no_cot"]


@dataclass
class TokenizedInputs:
    """Tokenized inputs with masks identifying different regions."""
    input_ids: torch.Tensor  # [batch, seq_len]
    prompt_mask: torch.Tensor  # [batch, seq_len] - True for prompt tokens
    cot_mask: torch.Tensor  # [batch, seq_len] - True for CoT tokens (between <think> and </think>)
    answer_mask: torch.Tensor  # [batch, seq_len] - True for answer tokens (after </think>)
    attention_mask: torch.Tensor | None  # [batch, seq_len, seq_len] - Custom attention mask (optional)


@dataclass
class FaithfulnessMetrics:
    """All faithfulness metrics."""
    # KL-based metrics (attention masking comparisons within same model)
    kl_direct_effect: torch.Tensor  # [batch] - DKL(full || via_cot)
    kl_cot_necessity: torch.Tensor  # [batch] - DKL(full || no_cot)
    kl_leakage: torch.Tensor  # [batch] - DKL(no_prompt || via_cot)

    # JS-based metrics (bounded [0, log2] — no near-delta blow-up)
    js_direct_effect: torch.Tensor  # [batch] - JS(full || via_cot)
    js_cot_necessity: torch.Tensor  # [batch] - JS(full || no_cot)

    # Gradient-based metrics (L1 and L2 norms)
    grad_de_l1: torch.Tensor  # [batch] - Prompt contribution (L1)
    grad_de_l2: torch.Tensor  # [batch] - Prompt contribution (L2)
    grad_cot_necessity_l1: torch.Tensor  # [batch] - CoT contribution (L1)
    grad_cot_necessity_l2: torch.Tensor  # [batch] - CoT contribution (L2)
    grad_leakage_l1: torch.Tensor  # [batch] - Leakage (L1)
    grad_leakage_l2: torch.Tensor  # [batch] - Leakage (L2)
    grad_pathway_change_l1: torch.Tensor  # [batch] - (via_cot - full) / full (L1)
    grad_pathway_change_l2: torch.Tensor  # [batch] - (via_cot - full) / full (L2)

    # Entropy/NLL metrics
    entropy_full: torch.Tensor  # [batch] - H(A|P,C)
    entropy_via_cot: torch.Tensor  # [batch] - H(A|C)
    entropy_no_prompt: torch.Tensor  # [batch] - H(A)
    nll_full: torch.Tensor  # [batch]
    nll_via_cot: torch.Tensor  # [batch]
    nll_no_prompt: torch.Tensor  # [batch]


# ============================================================================
# Attention Mask Creation
# ============================================================================

def create_via_cot_mask(
    prompt_mask: torch.Tensor,
    cot_mask: torch.Tensor,
    answer_mask: torch.Tensor
) -> torch.Tensor:
    """
    Create via_cot attention mask: blocks answer→prompt, allows CoT→prompt.
    Model applies causal masking internally.

    Fallback when no </think> delimiter found: treat entire response (cot_mask)
    as the answer region, so response→prompt is still blocked.

    Args:
        prompt_mask: [batch, seq_len]
        cot_mask: [batch, seq_len]
        answer_mask: [batch, seq_len]

    Returns:
        attention_mask: [batch, seq_len, seq_len] - False = blocked
    """
    batch_size, seq_len = prompt_mask.shape
    device = prompt_mask.device

    # Base: causal lower-triangular mask
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

    # Fallback: if no delimiter, treat entire response as answer
    has_answer = answer_mask.any(dim=1, keepdim=True)  # [batch, 1]
    effective_answer_mask = torch.where(has_answer, answer_mask, cot_mask)  # [batch, seq_len]

    # Additional block: answer→prompt
    block = effective_answer_mask.unsqueeze(2) & prompt_mask.unsqueeze(1)  # [batch, seq_len, seq_len]

    # Combine: causal AND NOT blocked
    attention_mask = causal.unsqueeze(0) & ~block  # [batch, seq_len, seq_len]

    return attention_mask


def create_no_prompt_mask(
    prompt_mask: torch.Tensor,
    seq_len: int
) -> torch.Tensor:
    """
    Create no_prompt attention mask: blocks all attention to prompt.
    Model applies causal masking internally.

    Args:
        prompt_mask: [batch, seq_len]
        seq_len: Sequence length

    Returns:
        attention_mask: [batch, seq_len, seq_len] - False = blocked
    """
    device = prompt_mask.device

    # Base: causal lower-triangular mask
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

    # Combine: causal AND NOT prompt-key positions
    # prompt_mask.unsqueeze(1): [batch, 1, seq_len] broadcasts over all query positions
    attention_mask = causal.unsqueeze(0) & ~prompt_mask.unsqueeze(1)  # [batch, seq_len, seq_len]

    return attention_mask


def create_no_cot_mask(
    cot_mask: torch.Tensor,
    seq_len: int
) -> torch.Tensor:
    """
    Create no_cot attention mask: blocks all attention to CoT.
    Model applies causal masking internally.

    Args:
        cot_mask: [batch, seq_len]
        seq_len: Sequence length

    Returns:
        attention_mask: [batch, seq_len, seq_len] - False = blocked
    """
    device = cot_mask.device

    # Base: causal lower-triangular mask
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

    # Combine: causal AND NOT cot-key positions
    # cot_mask.unsqueeze(1): [batch, 1, seq_len] broadcasts over all query positions
    attention_mask = causal.unsqueeze(0) & ~cot_mask.unsqueeze(1)  # [batch, seq_len, seq_len]

    return attention_mask


def create_full_mask(batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create full causal attention mask (standard autoregressive: each token attends to itself and all prior tokens).

    Returns:
        attention_mask: [batch, seq_len, seq_len] - lower-triangular bool
    """
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    return causal.unsqueeze(0).expand(batch_size, -1, -1)


def resolve_attention_mask(
    inputs: TokenizedInputs,
    attn_type: AttnMaskType
) -> torch.Tensor:
    """
    Resolve attention mask based on type.

    Args:
        inputs: TokenizedInputs
        attn_type: Type of attention mask

    Returns:
        attention_mask: [batch, seq_len, seq_len]
    """
    batch_size, seq_len = inputs.input_ids.shape

    if attn_type == "full":
        return create_full_mask(batch_size, seq_len, inputs.input_ids.device)
    elif attn_type == "via_cot":
        return create_via_cot_mask(inputs.prompt_mask, inputs.cot_mask, inputs.answer_mask)
    elif attn_type == "no_prompt":
        return create_no_prompt_mask(inputs.prompt_mask, seq_len)
    elif attn_type == "no_cot":
        return create_no_cot_mask(inputs.cot_mask, seq_len)
    else:
        raise ValueError(f"Unknown attention mask type: {attn_type}")


# ============================================================================
# Helper Functions
# ============================================================================

def get_hidden_states(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Get hidden states from model.

    Args:
        model: Language model
        input_ids: [batch, seq_len]
        attention_mask: [batch, seq_len, seq_len] or None

    Returns:
        hidden: [batch, seq_len, hidden_dim]
    """
    if attention_mask is not None:
        # Infer dtype from model parameters (e.g., bfloat16, float16, float32)
        model_dtype = next(model.parameters()).dtype

        # Convert 3D boolean mask [batch, seq_len, seq_len] to 4D float mask
        # [batch, 1, seq_len, seq_len] for broadcasting across attention heads
        # True = can attend (0.0), False = cannot attend (-inf)
        attn_mask_4d = torch.where(
            attention_mask,
            torch.tensor(0.0, dtype=model_dtype, device=attention_mask.device),
            torch.tensor(float('-inf'), dtype=model_dtype, device=attention_mask.device)
        ).unsqueeze(1)  # Add head dimension: [batch, 1, seq_len, seq_len]

        outputs = model(
            input_ids=input_ids,
            attention_mask=attn_mask_4d,
            output_hidden_states=True
        )
    else:
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True
        )

    # Get last hidden state
    return outputs.hidden_states[-1]


def compute_kl_divergence_from_hidden(
    model: nn.Module,
    hidden_student: torch.Tensor,
    hidden_teacher: torch.Tensor,
    answer_mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute KL divergence between distributions from hidden states.

    Args:
        model: Language model (for unembed)
        hidden_student: [batch, seq_len, hidden_dim]
        hidden_teacher: [batch, seq_len, hidden_dim]
        answer_mask: [batch, seq_len] - True for answer token positions

    Returns:
        kl: [batch] - KL divergence averaged over answer positions
    """
    # Get logits
    logits_student = model.lm_head(hidden_student)  # [batch, seq_len, vocab]
    logits_teacher = model.lm_head(hidden_teacher)  # [batch, seq_len, vocab]

    # Compute log probabilities
    log_p_student = F.log_softmax(logits_student, dim=-1)
    log_p_teacher = F.log_softmax(logits_teacher, dim=-1)
    p_teacher = log_p_teacher.exp()

    # KL divergence: sum over vocab, then average over positions
    kl_per_position = (p_teacher * (log_p_teacher - log_p_student)).sum(dim=-1)  # [batch, seq_len]

    # Shift for next-token prediction (logits[t] predicts token[t+1])
    # To measure KL for predicting answer tokens, we need positions where next token is an answer
    kl_per_position_shift = kl_per_position[:, :-1]  # Trim last position
    shift_mask = answer_mask[:, 1:]  # Positions where next token is an answer

    # Average over shifted positions
    mask_sum = shift_mask.float().sum(dim=1).clamp(min=1e-9)  # Avoid division by zero
    kl = (kl_per_position_shift * shift_mask.float()).sum(dim=1) / mask_sum

    # Return NaN if no answer tokens
    has_answer = shift_mask.any(dim=1)
    kl = torch.where(has_answer, kl, torch.tensor(float('nan'), device=kl.device))

    return kl


def compute_js_divergence_from_hidden(
    model: nn.Module,
    hidden_p: torch.Tensor,
    hidden_q: torch.Tensor,
    answer_mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute Jensen-Shannon divergence between distributions from hidden states.

    JS(P || Q) = (1/2) KL(P || M) + (1/2) KL(Q || M), where M = (P + Q) / 2.
    Bounded in [0, log(2)] — avoids the near-delta blow-up of forward KL.

    Args:
        model: Language model (for unembed)
        hidden_p: [batch, seq_len, hidden_dim]
        hidden_q: [batch, seq_len, hidden_dim]
        answer_mask: [batch, seq_len]

    Returns:
        js: [batch] - JS divergence averaged over answer positions
    """
    logits_p = model.lm_head(hidden_p)  # [batch, seq_len, vocab]
    logits_q = model.lm_head(hidden_q)

    log_p = F.log_softmax(logits_p, dim=-1)
    log_q = F.log_softmax(logits_q, dim=-1)
    p = log_p.exp()
    q = log_q.exp()

    m = (p + q) / 2
    log_m = torch.log(m.clamp(min=1e-40))

    js_per_position = 0.5 * (
        (p * (log_p - log_m)).sum(dim=-1) +
        (q * (log_q - log_m)).sum(dim=-1)
    )  # [batch, seq_len]

    # Shift for next-token prediction (consistent with KL and entropy)
    js_per_position_shift = js_per_position[:, :-1]
    shift_mask = answer_mask[:, 1:]

    mask_sum = shift_mask.float().sum(dim=1).clamp(min=1e-9)
    js = (js_per_position_shift * shift_mask.float()).sum(dim=1) / mask_sum

    has_answer = shift_mask.any(dim=1)
    js = torch.where(has_answer, js, torch.tensor(float('nan'), device=js.device))

    return js


def compute_entropy_from_hidden(
    model: nn.Module,
    hidden: torch.Tensor,
    answer_mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute entropy from hidden states.

    Uses the same next-token prediction shift as compute_kl_divergence_from_hidden
    and compute_nll_from_hidden: logits[t] predicts token[t+1], so we measure
    entropy at positions t where token t+1 is an answer token (answer_mask[:, 1:]).

    Args:
        model: Language model
        hidden: [batch, seq_len, hidden_dim]
        answer_mask: [batch, seq_len]

    Returns:
        entropy: [batch] - averaged over answer prediction positions
    """
    logits = model.lm_head(hidden)  # [batch, seq_len, vocab]
    log_p = F.log_softmax(logits, dim=-1)
    p = log_p.exp()

    # Entropy: -sum(p * log(p))
    entropy_per_position = -(p * log_p).sum(dim=-1)  # [batch, seq_len]

    # Shift to align with next-token prediction (consistent with KL and NLL)
    entropy_per_position_shift = entropy_per_position[:, :-1]  # [batch, seq_len-1]
    shift_mask = answer_mask[:, 1:]                             # positions predicting an answer token

    # Average over shifted answer positions
    mask_sum = shift_mask.float().sum(dim=1).clamp(min=1e-9)
    entropy = (entropy_per_position_shift * shift_mask.float()).sum(dim=1) / mask_sum

    # Return NaN if no answer tokens
    has_answer = shift_mask.any(dim=1)
    entropy = torch.where(has_answer, entropy, torch.tensor(float('nan'), device=entropy.device))

    return entropy


def compute_nll_from_hidden(
    model: nn.Module,
    hidden: torch.Tensor,
    labels: torch.Tensor,
    answer_mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute negative log-likelihood from hidden states.

    Args:
        model: Language model
        hidden: [batch, seq_len, hidden_dim]
        labels: [batch, seq_len]
        answer_mask: [batch, seq_len]

    Returns:
        nll: [batch] - averaged over answer positions
    """
    logits = model.lm_head(hidden)  # [batch, seq_len, vocab]

    # Shift for next token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = answer_mask[:, 1:]

    # Compute cross entropy
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    nll_per_position = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    ).view(shift_labels.shape)

    # Average over answer positions
    mask_sum = shift_mask.float().sum(dim=1).clamp(min=1e-9)
    nll = (nll_per_position * shift_mask.float()).sum(dim=1) / mask_sum

    # Return NaN if no answer tokens
    has_answer = shift_mask.any(dim=1)
    nll = torch.where(has_answer, nll, torch.tensor(float('nan'), device=nll.device))

    return nll


# ============================================================================
# KL-based Metrics
# ============================================================================

@torch.no_grad()
def kl_direct_effect(
    model: nn.Module,
    inputs: TokenizedInputs
) -> torch.Tensor:
    """
    KL direct effect: DKL(full || via_cot).
    Measures how much prompt directly affects answer when bypassing CoT.

    Args:
        model: Language model
        inputs: TokenizedInputs

    Returns:
        kl: [batch]
    """
    # Full attention
    hidden_full = get_hidden_states(
        model,
        inputs.input_ids,
        resolve_attention_mask(inputs, "full")
    )

    # Via CoT attention (answer→prompt blocked)
    hidden_via_cot = get_hidden_states(
        model,
        inputs.input_ids,
        resolve_attention_mask(inputs, "via_cot")
    )

    return compute_kl_divergence_from_hidden(
        model,
        hidden_student=hidden_via_cot,
        hidden_teacher=hidden_full,
        answer_mask=inputs.answer_mask
    )


@torch.no_grad()
def kl_cot_necessity(
    model: nn.Module,
    inputs: TokenizedInputs
) -> torch.Tensor:
    """
    KL CoT necessity: DKL(full || no_cot).
    Measures how much CoT contributes to answer.

    Args:
        model: Language model
        inputs: TokenizedInputs

    Returns:
        kl: [batch]
    """
    # Full attention
    hidden_full = get_hidden_states(
        model,
        inputs.input_ids,
        resolve_attention_mask(inputs, "full")
    )

    # No CoT attention (CoT blocked)
    hidden_no_cot = get_hidden_states(
        model,
        inputs.input_ids,
        resolve_attention_mask(inputs, "no_cot")
    )

    return compute_kl_divergence_from_hidden(
        model,
        hidden_student=hidden_no_cot,
        hidden_teacher=hidden_full,
        answer_mask=inputs.answer_mask
    )


@torch.no_grad()
def kl_leakage(
    model: nn.Module,
    inputs: TokenizedInputs
) -> torch.Tensor:
    """
    KL leakage: DKL(no_prompt || via_cot).
    Measures information leakage through CoT activations.

    Args:
        model: Language model
        inputs: TokenizedInputs

    Returns:
        kl: [batch]
    """
    # Via CoT attention
    hidden_via_cot = get_hidden_states(
        model,
        inputs.input_ids,
        resolve_attention_mask(inputs, "via_cot")
    )

    # No prompt attention
    hidden_no_prompt = get_hidden_states(
        model,
        inputs.input_ids,
        resolve_attention_mask(inputs, "no_prompt")
    )

    return compute_kl_divergence_from_hidden(
        model,
        hidden_student=hidden_no_prompt,
        hidden_teacher=hidden_via_cot,
        answer_mask=inputs.answer_mask
    )


# ============================================================================
# Gradient-based Metrics
# ============================================================================

def compute_gradient_norms(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    answer_mask: torch.Tensor,
    attention_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Compute per-token gradient norms ||∂L/∂e_t|| w.r.t. answer loss.

    Args:
        model: Language model
        input_ids: [batch, seq_len]
        labels: [batch, seq_len]
        answer_mask: [batch, seq_len]
        attention_mask: [batch, seq_len, seq_len] or None

    Returns:
        grad_norms: [batch, seq_len]
    """
    batch_size, seq_len = input_ids.shape

    # Get embeddings and enable gradients
    embeddings = model.get_input_embeddings()(input_ids)
    embeddings.requires_grad_(True)
    embeddings.retain_grad()  # Needed for non-leaf tensors

    # Infer dtype from model parameters
    model_dtype = embeddings.dtype

    # Convert attention mask to 4D if provided
    if attention_mask is not None:
        attn_mask_4d = torch.where(
            attention_mask,
            torch.tensor(0.0, dtype=model_dtype, device=attention_mask.device),
            torch.tensor(float('-inf'), dtype=model_dtype, device=attention_mask.device)
        ).unsqueeze(1)  # [batch, 1, seq_len, seq_len]

        outputs = model(
            inputs_embeds=embeddings,
            attention_mask=attn_mask_4d
        )
    else:
        outputs = model(
            inputs_embeds=embeddings
        )

    # Compute loss on answer region
    shift_logits = outputs.logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_answer_mask = answer_mask[:, 1:]

    loss_fct = nn.CrossEntropyLoss(reduction='none')
    loss_per_token = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    ).view(batch_size, seq_len - 1)

    # Apply answer mask and compute mean
    masked_loss = (loss_per_token * shift_answer_mask.float()).sum() / shift_answer_mask.float().sum().clamp(min=1)

    # Backward
    masked_loss.backward()

    # Compute gradient norms
    assert embeddings.grad is not None
    grad_norms = torch.norm(embeddings.grad, dim=-1)  # [batch, seq_len]

    # Detach from computation graph and clean up gradients
    grad_norms = grad_norms.detach()

    # Clear gradients to free memory (keep del statements, but not empty_cache)
    embeddings.grad = None
    del embeddings
    del outputs

    return grad_norms


def grad_direct_effect(
    model: nn.Module,
    inputs: TokenizedInputs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gradient-based direct effect.
    Computes prompt contribution under full attention.

    Args:
        model: Language model
        inputs: TokenizedInputs

    Returns:
        (grad_l1, grad_l2): Both [batch]
    """
    grad_norms = compute_gradient_norms(
        model,
        inputs.input_ids,
        inputs.input_ids,  # labels = input_ids for next token prediction
        inputs.answer_mask,
        resolve_attention_mask(inputs, "full")
    )

    # L1: sum of norms
    prompt_sum = (grad_norms * inputs.prompt_mask.float()).sum(dim=1)
    total_sum = grad_norms.sum(dim=1).clamp(min=1e-9)
    grad_l1 = prompt_sum / total_sum

    # L2: sum of squared norms
    prompt_sq_sum = (grad_norms.pow(2) * inputs.prompt_mask.float()).sum(dim=1)
    total_sq_sum = grad_norms.pow(2).sum(dim=1).clamp(min=1e-9)
    grad_l2 = prompt_sq_sum / total_sq_sum

    # Return NaN if no answer tokens
    has_answer = inputs.answer_mask.any(dim=1)
    grad_l1 = torch.where(has_answer, grad_l1, torch.tensor(float('nan'), device=grad_l1.device))
    grad_l2 = torch.where(has_answer, grad_l2, torch.tensor(float('nan'), device=grad_l2.device))

    return grad_l1, grad_l2


def grad_cot_necessity(
    model: nn.Module,
    inputs: TokenizedInputs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gradient-based CoT necessity.
    Computes CoT contribution under full attention.

    Args:
        model: Language model
        inputs: TokenizedInputs

    Returns:
        (grad_l1, grad_l2): Both [batch]
    """
    grad_norms = compute_gradient_norms(
        model,
        inputs.input_ids,
        inputs.input_ids,
        inputs.answer_mask,
        resolve_attention_mask(inputs, "full")
    )

    # L1: sum of norms
    cot_sum = (grad_norms * inputs.cot_mask.float()).sum(dim=1)
    total_sum = grad_norms.sum(dim=1).clamp(min=1e-9)
    grad_l1 = cot_sum / total_sum

    # L2: sum of squared norms
    cot_sq_sum = (grad_norms.pow(2) * inputs.cot_mask.float()).sum(dim=1)
    total_sq_sum = grad_norms.pow(2).sum(dim=1).clamp(min=1e-9)
    grad_l2 = cot_sq_sum / total_sq_sum

    # Return NaN if no answer tokens
    has_answer = inputs.answer_mask.any(dim=1)
    grad_l1 = torch.where(has_answer, grad_l1, torch.tensor(float('nan'), device=grad_l1.device))
    grad_l2 = torch.where(has_answer, grad_l2, torch.tensor(float('nan'), device=grad_l2.device))

    return grad_l1, grad_l2


def grad_leakage(
    model: nn.Module,
    inputs: TokenizedInputs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gradient-based leakage.
    Computes prompt contribution under via_cot attention.

    Args:
        model: Language model
        inputs: TokenizedInputs

    Returns:
        (grad_l1, grad_l2): Both [batch]
    """
    grad_norms = compute_gradient_norms(
        model,
        inputs.input_ids,
        inputs.input_ids,
        inputs.answer_mask,
        resolve_attention_mask(inputs, "via_cot")
    )

    # L1: sum of norms
    prompt_sum = (grad_norms * inputs.prompt_mask.float()).sum(dim=1)
    total_sum = grad_norms.sum(dim=1).clamp(min=1e-9)
    grad_l1 = prompt_sum / total_sum

    # L2: sum of squared norms
    prompt_sq_sum = (grad_norms.pow(2) * inputs.prompt_mask.float()).sum(dim=1)
    total_sq_sum = grad_norms.pow(2).sum(dim=1).clamp(min=1e-9)
    grad_l2 = prompt_sq_sum / total_sq_sum

    # Return NaN if no answer tokens
    has_answer = inputs.answer_mask.any(dim=1)
    grad_l1 = torch.where(has_answer, grad_l1, torch.tensor(float('nan'), device=grad_l1.device))
    grad_l2 = torch.where(has_answer, grad_l2, torch.tensor(float('nan'), device=grad_l2.device))

    return grad_l1, grad_l2


def grad_pathway_change(
    model: nn.Module,
    inputs: TokenizedInputs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gradient pathway change from blocking direct prompt→answer attention.
    Measures: (||grad_prompt_via_cot|| - ||grad_prompt_full||) / ||grad_prompt_full||

    Positive = Blocking direct attention increases prompt gradient (more indirect pathway)
    Negative = Blocking direct attention decreases prompt gradient (relies on direct path)

    Args:
        model: Language model
        inputs: TokenizedInputs

    Returns:
        (change_l1, change_l2): Both [batch]
    """
    # Compute gradient norms under full attention
    grad_norms_full = compute_gradient_norms(
        model,
        inputs.input_ids,
        inputs.input_ids,
        inputs.answer_mask,
        resolve_attention_mask(inputs, "full")
    )

    # Compute gradient norms under via_cot attention (answer→prompt blocked)
    grad_norms_via_cot = compute_gradient_norms(
        model,
        inputs.input_ids,
        inputs.input_ids,
        inputs.answer_mask,
        resolve_attention_mask(inputs, "via_cot")
    )

    # L1: sum of norms on prompt region
    prompt_sum_full = (grad_norms_full * inputs.prompt_mask.float()).sum(dim=1).clamp(min=1e-9)
    prompt_sum_via_cot = (grad_norms_via_cot * inputs.prompt_mask.float()).sum(dim=1)
    change_l1 = (prompt_sum_via_cot - prompt_sum_full) / prompt_sum_full

    # L2: sum of squared norms on prompt region
    prompt_sq_sum_full = (grad_norms_full.pow(2) * inputs.prompt_mask.float()).sum(dim=1).clamp(min=1e-9)
    prompt_sq_sum_via_cot = (grad_norms_via_cot.pow(2) * inputs.prompt_mask.float()).sum(dim=1)
    change_l2 = (prompt_sq_sum_via_cot - prompt_sq_sum_full) / prompt_sq_sum_full

    # Return NaN if no answer tokens
    has_answer = inputs.answer_mask.any(dim=1)
    change_l1 = torch.where(has_answer, change_l1, torch.tensor(float('nan'), device=change_l1.device))
    change_l2 = torch.where(has_answer, change_l2, torch.tensor(float('nan'), device=change_l2.device))

    return change_l1, change_l2


# ============================================================================
# Entropy/NLL Metrics
# ============================================================================

@torch.no_grad()
def compute_entropy_nll(
    model: nn.Module,
    inputs: TokenizedInputs,
    attn_type: AttnMaskType
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute entropy and NLL under specified attention mask.

    Args:
        model: Language model
        inputs: TokenizedInputs
        attn_type: Type of attention mask

    Returns:
        (entropy, nll): Both [batch]
    """
    hidden = get_hidden_states(
        model,
        inputs.input_ids,
        resolve_attention_mask(inputs, attn_type)
    )

    entropy = compute_entropy_from_hidden(model, hidden, inputs.answer_mask)
    nll = compute_nll_from_hidden(model, hidden, inputs.input_ids, inputs.answer_mask)

    return entropy, nll


@torch.no_grad()
def compute_entropy_nll_from_hidden(
    model: nn.Module,
    inputs: TokenizedInputs,
    hidden: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute entropy and NLL from precomputed hidden states.

    Args:
        model: Language model
        inputs: TokenizedInputs
        hidden: [batch, seq_len, hidden_dim] - precomputed hidden states

    Returns:
        (entropy, nll): Both [batch]
    """
    entropy = compute_entropy_from_hidden(model, hidden, inputs.answer_mask)
    nll = compute_nll_from_hidden(model, hidden, inputs.input_ids, inputs.answer_mask)
    return entropy, nll


# ============================================================================
# Main Evaluation Function
# ============================================================================

def compute_all_metrics(
    model: nn.Module,
    inputs: TokenizedInputs,
    device: str = "cuda",
    compute_gradients: bool = True
) -> FaithfulnessMetrics:
    """
    Compute all faithfulness metrics on a single model.

    Args:
        model: Language model to evaluate
        inputs: TokenizedInputs with all masks defined
        device: Device to run on
        compute_gradients: Whether to compute gradient-based metrics (memory-intensive)
                          Set to False to save GPU memory (~2GB) if not needed

    Returns:
        FaithfulnessMetrics with all metrics

    Usage:
        # For generating model (all metrics)
        inputs_gen = create_tokenized_inputs(tokenizer, questions, outputs)
        metrics_gen = compute_all_metrics(model, inputs_gen)

        # For generating model (skip gradients to save memory)
        metrics_gen = compute_all_metrics(model, inputs_gen, compute_gradients=False)

        # For reference model (if using different tokenizer)
        inputs_ref = create_tokenized_inputs(ref_tokenizer, questions, outputs)
        metrics_ref = compute_all_metrics(ref_model, inputs_ref)
    """
    model = model.to(device)
    model.eval()

    # Move inputs to device
    inputs = TokenizedInputs(
        input_ids=inputs.input_ids.to(device),
        prompt_mask=inputs.prompt_mask.to(device),
        cot_mask=inputs.cot_mask.to(device),
        answer_mask=inputs.answer_mask.to(device),
        attention_mask=inputs.attention_mask.to(device) if inputs.attention_mask is not None else None
    )

    # ============== CACHE HIDDEN STATES (4 forward passes) ==============
    with torch.no_grad():
        hidden_cache = {
            "full": get_hidden_states(
                model, inputs.input_ids, resolve_attention_mask(inputs, "full")
            ),
            "via_cot": get_hidden_states(
                model, inputs.input_ids, resolve_attention_mask(inputs, "via_cot")
            ),
            "no_prompt": get_hidden_states(
                model, inputs.input_ids, resolve_attention_mask(inputs, "no_prompt")
            ),
            "no_cot": get_hidden_states(
                model, inputs.input_ids, resolve_attention_mask(inputs, "no_cot")
            ),
        }

    # ============== USE CACHED HIDDEN STATES ==============
    # All lm_head calls here must be inside no_grad — otherwise PyTorch builds a
    # computation graph for each lm_head(hidden) call (lm_head.weight.requires_grad=True),
    # retaining all [batch, seq_len, vocab] intermediates (~2GB each) until the
    # returned scalar tensor is GC'd. 8 calls × ~15GB of intermediates = ~120GB.
    with torch.no_grad():
        # KL-based metrics (use cache)
        kl_de = compute_kl_divergence_from_hidden(
            model, hidden_cache["via_cot"], hidden_cache["full"], inputs.answer_mask
        )
        kl_cot_nec = compute_kl_divergence_from_hidden(
            model, hidden_cache["no_cot"], hidden_cache["full"], inputs.answer_mask
        )
        kl_leak = compute_kl_divergence_from_hidden(
            model, hidden_cache["no_prompt"], hidden_cache["via_cot"], inputs.answer_mask
        )

        # Entropy/NLL metrics (use cache)
        entropy_full, nll_full = compute_entropy_nll_from_hidden(model, inputs, hidden_cache["full"])
        entropy_via_cot, nll_via_cot = compute_entropy_nll_from_hidden(model, inputs, hidden_cache["via_cot"])
        entropy_no_prompt, nll_no_prompt = compute_entropy_nll_from_hidden(model, inputs, hidden_cache["no_prompt"])

    # Clear cache BEFORE gradient computation to free memory
    del hidden_cache
    torch.cuda.empty_cache()

    # Gradient-based metrics (most memory-intensive - compute after clearing cache)
    if compute_gradients:
        # Gradient checkpointing requires model.training=True to activate.
        # Switch to train mode just for gradient computation (no dropout in 12B models).
        _ckpt_enabled = False
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.train()
            model.gradient_checkpointing_enable()
            _ckpt_enabled = True

        # grad_direct_effect and grad_cot_necessity both use the "full" causal attention mask,
        # which is identical to the model's default (attention_mask=None).
        # Passing None lets the model use FlashAttention instead of SDPA with a custom 4D mask.
        # SDPA with a custom mask stores full attention matrices for backward:
        # 48 layers × [batch, 32 heads, 2048, 2048] × 2 bytes ≈ 25GB.
        # FlashAttention avoids this, reducing backward activation memory by ~25GB.
        grad_norms_full = compute_gradient_norms(
            model,
            inputs.input_ids,
            inputs.input_ids,
            inputs.answer_mask,
            attention_mask=None  # default causal = "full" mask, but FlashAttention compatible
        )
        torch.cuda.empty_cache()

        # Direct effect: prompt token contribution
        prompt_sum = (grad_norms_full * inputs.prompt_mask.float()).sum(dim=1)
        total_sum = grad_norms_full.sum(dim=1).clamp(min=1e-9)
        grad_de_l1 = prompt_sum / total_sum
        prompt_sq = (grad_norms_full.pow(2) * inputs.prompt_mask.float()).sum(dim=1)
        total_sq = grad_norms_full.pow(2).sum(dim=1).clamp(min=1e-9)
        grad_de_l2 = prompt_sq / total_sq
        has_answer = inputs.answer_mask.any(dim=1)
        _nan = torch.tensor(float('nan'), device=grad_de_l1.device)
        grad_de_l1 = torch.where(has_answer, grad_de_l1, _nan)
        grad_de_l2 = torch.where(has_answer, grad_de_l2, _nan)

        # CoT necessity: CoT token contribution (same backward pass)
        cot_sum = (grad_norms_full * inputs.cot_mask.float()).sum(dim=1)
        grad_cot_nec_l1 = cot_sum / total_sum
        cot_sq = (grad_norms_full.pow(2) * inputs.cot_mask.float()).sum(dim=1)
        grad_cot_nec_l2 = cot_sq / total_sq
        grad_cot_nec_l1 = torch.where(has_answer, grad_cot_nec_l1, _nan)
        grad_cot_nec_l2 = torch.where(has_answer, grad_cot_nec_l2, _nan)

        del grad_norms_full
        torch.cuda.empty_cache()

        grad_leak_l1, grad_leak_l2 = grad_leakage(model, inputs)
        torch.cuda.empty_cache()

        if _ckpt_enabled:
            model.gradient_checkpointing_disable()
            model.eval()
        # DISABLED: grad_pathway_change is expensive (2x gradient computations)
        # grad_pathway_l1, grad_pathway_l2 = grad_pathway_change(model, inputs)
        batch_size = inputs.input_ids.size(0)
        grad_pathway_l1 = grad_pathway_l2 = torch.full((batch_size,), float('nan'))
    else:
        # Return NaN if gradient metrics skipped
        batch_size = inputs.input_ids.size(0)
        nan_tensor = torch.full((batch_size,), float('nan'))
        grad_de_l1 = grad_de_l2 = nan_tensor
        grad_cot_nec_l1 = grad_cot_nec_l2 = nan_tensor
        grad_leak_l1 = grad_leak_l2 = nan_tensor
        grad_pathway_l1 = grad_pathway_l2 = nan_tensor

    batch_size = inputs.input_ids.size(0)
    _nan_cpu = torch.full((batch_size,), float('nan'))

    return FaithfulnessMetrics(
        kl_direct_effect=kl_de.cpu(),
        kl_cot_necessity=kl_cot_nec.cpu(),
        kl_leakage=kl_leak.cpu(),
        js_direct_effect=_nan_cpu,
        js_cot_necessity=_nan_cpu,
        grad_de_l1=grad_de_l1.cpu(),
        grad_de_l2=grad_de_l2.cpu(),
        grad_cot_necessity_l1=grad_cot_nec_l1.cpu(),
        grad_cot_necessity_l2=grad_cot_nec_l2.cpu(),
        grad_leakage_l1=grad_leak_l1.cpu(),
        grad_leakage_l2=grad_leak_l2.cpu(),
        grad_pathway_change_l1=grad_pathway_l1.cpu(),
        grad_pathway_change_l2=grad_pathway_l2.cpu(),
        entropy_full=entropy_full.cpu(),
        entropy_via_cot=entropy_via_cot.cpu(),
        entropy_no_prompt=entropy_no_prompt.cpu(),
        nll_full=nll_full.cpu(),
        nll_via_cot=nll_via_cot.cpu(),
        nll_no_prompt=nll_no_prompt.cpu()
    )


# ============================================================================
# Utility: Create TokenizedInputs from raw text
# ============================================================================

def _find_first_subsequence(
    sequence: torch.Tensor,   # [seq_len]
    subseq: torch.Tensor,     # [n]
    start: int = 0,
) -> int | None:
    """Return the start index of the first occurrence of subseq in sequence[start:]."""
    n = subseq.shape[0]
    seq_len = sequence.shape[0]
    for pos in range(start, seq_len - n + 1):
        if torch.all(sequence[pos:pos + n] == subseq):
            return pos
    return None


def create_tokenized_inputs(
    tokenizer: AutoTokenizer,
    questions: list[str],
    outputs: list[str],
    device: str = "cuda",
    use_substring_matching: bool = True,
    end_think_token_id: int | None = None,
    truncate_at_assert: bool = False,
) -> TokenizedInputs:
    """
    Create TokenizedInputs from questions and outputs.

    Args:
        tokenizer: Tokenizer
        questions: List of user questions
        outputs: List of assistant outputs (format: "<think>...</think>\\nFinal answer: X")
        device: Device
        use_substring_matching: If True, use substring matching for </think> delimiter.
                                If False, use special token mode with end_think_token_id.
        end_think_token_id: Token ID for </think> (only used if use_substring_matching=False)

    Returns:
        TokenizedInputs with all masks

    Note:
        This function uses the same mask creation logic as training:
        - Mode 1 (default): Substring matching - works with any tokenizer
        - Mode 2: Special token mode - requires tokenizer with </think> special token

        The masks are created using the unified `find_special_token_positions()` function
        from `train/verl/utils/cot_masking.py` for consistency with training.

    Example:
        # Substring matching mode (default - works with any tokenizer)
        inputs = create_tokenized_inputs(tokenizer, questions, outputs)
        metrics = compute_all_metrics(model, inputs)

        # Special token mode (requires special token in tokenizer)
        inputs = create_tokenized_inputs(
            tokenizer, questions, outputs,
            use_substring_matching=False,
            end_think_token_id=151666
        )
    """
    # Apply chat template
    messages_list = [
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": output}
        ]
        for question, output in zip(questions, outputs)
    ]

    # Tokenize
    input_ids_list = []
    response_lengths = []  # Track response length for each sample
    for messages in messages_list:
        # Tokenize full conversation
        tokenized_full = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt"
        )

        # Tokenize just the prompt to compute response length
        tokenized_prompt = tokenizer.apply_chat_template(
            [messages[0]],  # Just user message
            tokenize=True,
            add_generation_prompt=True,  # Include assistant prefix
            return_tensors="pt"
        )

        input_ids_list.append(tokenized_full[0])
        response_lengths.append(tokenized_full.shape[1] - tokenized_prompt.shape[1])

    # Pad to same length
    max_len = max(ids.shape[0] for ids in input_ids_list)
    batch_size = len(input_ids_list)

    input_ids = torch.full(
        (batch_size, max_len),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=device
    )
    for i, ids in enumerate(input_ids_list):
        input_ids[i, :ids.shape[0]] = ids.to(device)

    # Use the same mask creation logic as training
    # Import from training utilities
    import sys
    import os
    # Add train directory to path if not already there
    train_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'train')
    if train_dir not in sys.path:
        sys.path.insert(0, train_dir)

    from verl.utils.cot_masking_origin import find_special_token_positions, tokenize_delimiter_string

    # Prepare delimiter detection parameters based on mode
    if use_substring_matching:
        # Substring matching mode: tokenize </think> delimiter
        delimiter_str = "</think>"
        end_delimiter_ids = tokenize_delimiter_string(tokenizer, delimiter_str, device)
        end_think_token_id_param = None
        end_think_delimiter_ids_param = end_delimiter_ids

        # DEBUG: Show tokenization (optional, can be commented out for production)
        print(f"[Eval] Substring matching mode: '{delimiter_str}' → {end_delimiter_ids.tolist()}")
        decoded_tokens = [tokenizer.decode([tid]) for tid in end_delimiter_ids.tolist()]
        print(f"[Eval] Decoded tokens: {decoded_tokens}")
    else:
        # Special token mode: use single token ID
        if end_think_token_id is None:
            raise ValueError("end_think_token_id must be provided when use_substring_matching=False")
        end_think_token_id_param = end_think_token_id
        end_think_delimiter_ids_param = None

        # DEBUG: Show special token
        print(f"[Eval] Special token mode: end_think_token_id={end_think_token_id}")

    # Create masks using unified interface (same as training)
    # We need to call find_special_token_positions for each unique response_length
    # For simplicity, process each sample individually
    prompt_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
    cot_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
    answer_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)

    for i in range(batch_size):
        # Get masks for this sample
        sample_input_ids = input_ids[i:i+1]  # Keep batch dimension
        sample_response_length = response_lengths[i]

        # Use unified mask creation (same as training)
        sample_prompt_mask, sample_cot_mask, sample_answer_mask = find_special_token_positions(
            input_ids=sample_input_ids,
            response_length=sample_response_length,
            end_think_token_id=end_think_token_id_param,
            end_think_delimiter_ids=end_think_delimiter_ids_param,
        )

        # Copy to batch masks
        prompt_mask[i] = sample_prompt_mask[0]
        cot_mask[i] = sample_cot_mask[0]
        answer_mask[i] = sample_answer_mask[0]

    # Optional: truncate answer_mask at first '\nassert' (exclude assertion block)
    # This makes KL metrics compare only the solve() function body, not the assertion lines,
    # so methods that keep more assertions aren't penalised vs methods that drop them.
    if truncate_at_assert:
        # Encode bare "assert" keyword — avoids \n encoding ambiguity
        # (\n\n and \n tokenize as different token IDs in Gemma so \nassert is unreliable)
        assert_token_ids = tokenize_delimiter_string(tokenizer, "assert", device)
        n_truncated = 0
        for i in range(batch_size):
            answer_positions = answer_mask[i].nonzero(as_tuple=True)[0]
            if len(answer_positions) == 0:
                continue
            answer_start = answer_positions[0].item()
            pos = _find_first_subsequence(input_ids[i], assert_token_ids, start=answer_start)
            if pos is not None:
                answer_mask[i, pos:] = False
                n_truncated += 1
        print(f"[Eval] truncate_at_assert: truncated {n_truncated}/{batch_size} samples at first 'assert'")

    # DEBUG: Show mask statistics (matching training output style)
    num_samples_with_delimiter = (answer_mask.any(dim=1)).sum().item()
    prompt_sum = prompt_mask.sum().item()
    cot_sum = cot_mask.sum().item()
    answer_sum = answer_mask.sum().item()
    total_tokens = prompt_mask.numel()

    print(f"[Eval] Batch size: {batch_size}")
    print(f"[Eval] Samples with </think>: {num_samples_with_delimiter}/{batch_size} ({100*num_samples_with_delimiter/batch_size:.1f}%)")
    print(f"[Eval] Token distribution:")
    print(f"  - Prompt tokens: {prompt_sum}/{total_tokens} ({100*prompt_sum/total_tokens:.1f}%)")
    print(f"  - CoT tokens: {cot_sum}/{total_tokens} ({100*cot_sum/total_tokens:.1f}%)")
    print(f"  - Answer tokens: {answer_sum}/{total_tokens} ({100*answer_sum/total_tokens:.1f}%)")

    return TokenizedInputs(
        input_ids=input_ids,
        prompt_mask=prompt_mask,
        cot_mask=cot_mask,
        answer_mask=answer_mask,
        attention_mask=None
    )
