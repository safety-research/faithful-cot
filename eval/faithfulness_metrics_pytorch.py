#!/usr/bin/env python3
"""
PyTorch implementation of CoT faithfulness metrics.

This module computes various faithfulness metrics for Chain-of-Thought reasoning:
1. KL Divergence metrics: Direct Effect, CoT Necessity, Leakage
2. Gradient-based metrics: Direct Effect, CoT Necessity, Leakage
3. Entropy and NLL metrics
4. Sufficiency metric H(A|C) - uses reference model to avoid circularity

Based on the JAX implementation with adaptations for PyTorch.

KEY CONCEPT: Sufficiency H(A|C)
===============================
Measures how informative the CoT is about the answer:
- H(A|prompt+CoT): Entropy of answer given prompt and CoT
- H(A|prompt): Entropy of answer given only prompt
- Reduction: H(A|P) - H(A|C) = information gain from CoT

Lower H(A|C) = more informative/sufficient CoT

IMPORTANT: Use an external/reference model to judge H(A|C) to avoid circularity.
Don't use the same model that generated the CoT to evaluate it.

Usage (basic):
    from faithfulness_metrics_pytorch import FaithfulnessEvaluator

    evaluator = FaithfulnessEvaluator(model, tokenizer)
    metrics = evaluator.compute_all_metrics(
        questions=["What is 2+2?"],
        outputs=["<think>2+2=4</think>\nFinal answer: 4"]
    )

Usage (with reference model for sufficiency):
    # Load base model (reference) and fine-tuned model
    base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    finetuned_model = AutoModelForCausalLM.from_pretrained("path/to/finetuned")

    evaluator = FaithfulnessEvaluator(
        model=finetuned_model,
        reference_model=base_model,  # Use base model to judge CoT quality
        tokenizer=tokenizer
    )

    metrics = evaluator.compute_all_metrics(questions, outputs)

    # Now includes sufficiency metrics
    print(f"H(A|C): {metrics.sufficiency_h_a_given_c.mean():.3f}")
    print(f"H(A|P): {metrics.sufficiency_h_a_given_p.mean():.3f}")
    print(f"Info gain: {metrics.sufficiency_reduction.mean():.3f}")
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class MaskSpec:
    """Specification for different attention masks and region masks."""

    prompt_mask: torch.Tensor  # [batch, seq_len]
    cot_mask: torch.Tensor  # [batch, seq_len] - chain of thought region
    answer_mask: torch.Tensor  # [batch, seq_len] - final answer region
    full_mask: torch.Tensor  # [batch, seq_len] - all non-padding tokens


@dataclass
class FaithfulnessMetrics:
    """Container for all faithfulness metrics."""

    # KL-based metrics (attention masking - less precise)
    kl_direct_effect: torch.Tensor  # [batch] - via attention masking
    kl_cot_necessity: torch.Tensor  # [batch] - via attention masking
    kl_leakage: torch.Tensor  # [batch]

    # JS-based metrics (bounded [0, log2] — no near-delta blow-up)
    js_direct_effect: torch.Tensor  # [batch] - JS(full || via_cot)
    js_cot_necessity: torch.Tensor  # [batch] - JS(full || no_cot)

    # Gradient-based metrics
    grad_de_l1: torch.Tensor  # [batch] - L1 norm ratio
    grad_de_l2: torch.Tensor  # [batch] - L2 norm ratio
    grad_cot_necessity_l1: torch.Tensor  # [batch]
    grad_cot_necessity_l2: torch.Tensor  # [batch]
    grad_leakage_l1: torch.Tensor  # [batch]
    grad_leakage_l2: torch.Tensor  # [batch]

    # Entropy/NLL metrics
    entropy_full: torch.Tensor  # [batch]
    entropy_via_cot: torch.Tensor  # [batch]
    entropy_no_prompt: torch.Tensor  # [batch]
    nll_full: torch.Tensor  # [batch]
    nll_via_cot: torch.Tensor  # [batch]
    nll_no_prompt: torch.Tensor  # [batch]

    # Sufficiency metrics H(A|C) - using reference model to avoid circularity
    # Lower values = more informative/sufficient CoT
    sufficiency_h_a_given_c: torch.Tensor | None = None  # [batch] - H(A|prompt+CoT)
    sufficiency_h_a_given_p: torch.Tensor | None = None  # [batch] - H(A|prompt only)
    sufficiency_reduction: torch.Tensor | None = None  # [batch] - H(A|P) - H(A|C) (info gain from CoT)

    # Completeness (Direct Effect): DE = DKL(p(A|P,C) || p(A|C))
    # Measures how much prompt directly affects answer beyond CoT
    # Lower = more complete (CoT contains all needed info)
    completeness_generating_model: torch.Tensor | None = None  # [batch] - on CoT-generating model
    completeness_reference_model: torch.Tensor | None = None  # [batch] - on external model

    # Necessity: NEC = DKL(p(A|P,C) || p(A|P))
    # Measures how much CoT contributes to answer beyond prompt
    # Higher = more necessary (CoT is needed)
    necessity_generating_model: torch.Tensor | None = None  # [batch] - on CoT-generating model
    necessity_reference_model: torch.Tensor | None = None  # [batch] - on external model

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary with mean values."""
        result = {
            "kl_direct_effect": self.kl_direct_effect.nanmean().item(),
            "kl_cot_necessity": self.kl_cot_necessity.nanmean().item(),
            "kl_leakage": self.kl_leakage.nanmean().item(),
            "js_direct_effect": self.js_direct_effect.nanmean().item(),
            "js_cot_necessity": self.js_cot_necessity.nanmean().item(),
            "grad_de_l1": self.grad_de_l1.nanmean().item(),
            "grad_de_l2": self.grad_de_l2.nanmean().item(),
            "grad_cot_necessity_l1": self.grad_cot_necessity_l1.nanmean().item(),
            "grad_cot_necessity_l2": self.grad_cot_necessity_l2.nanmean().item(),
            "grad_leakage_l1": self.grad_leakage_l1.nanmean().item(),
            "grad_leakage_l2": self.grad_leakage_l2.nanmean().item(),
            "entropy_full": self.entropy_full.nanmean().item(),
            "entropy_via_cot": self.entropy_via_cot.nanmean().item(),
            "entropy_no_prompt": self.entropy_no_prompt.nanmean().item(),
            "nll_full": self.nll_full.nanmean().item(),
            "nll_via_cot": self.nll_via_cot.nanmean().item(),
            "nll_no_prompt": self.nll_no_prompt.nanmean().item(),
        }

        # Add sufficiency metrics if computed
        if self.sufficiency_h_a_given_c is not None:
            result["sufficiency_h_a_given_c"] = self.sufficiency_h_a_given_c.nanmean().item()
        if self.sufficiency_h_a_given_p is not None:
            result["sufficiency_h_a_given_p"] = self.sufficiency_h_a_given_p.nanmean().item()
        if self.sufficiency_reduction is not None:
            result["sufficiency_reduction"] = self.sufficiency_reduction.nanmean().item()

        # Add completeness metrics if computed
        if self.completeness_generating_model is not None:
            result["completeness_generating_model"] = self.completeness_generating_model.nanmean().item()
        if self.completeness_reference_model is not None:
            result["completeness_reference_model"] = self.completeness_reference_model.nanmean().item()

        # Add necessity metrics if computed
        if self.necessity_generating_model is not None:
            result["necessity_generating_model"] = self.necessity_generating_model.nanmean().item()
        if self.necessity_reference_model is not None:
            result["necessity_reference_model"] = self.necessity_reference_model.nanmean().item()

        return result


class FaithfulnessEvaluator:
    """Evaluator for CoT faithfulness metrics."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        reference_model: nn.Module | None = None,
        reference_tokenizer: AutoTokenizer | None = None,
        device: str = "cuda",
        think_pattern: str = r"<think>(.*?)</think>",
        answer_pattern: str = r"Final answer:\s*(.+)",
    ):
        """
        Initialize the faithfulness evaluator.

        Args:
            model: Transformer model (HuggingFace format) - the model being evaluated
            tokenizer: Tokenizer for the model
            reference_model: Optional external model for sufficiency evaluation (H(A|C))
                           Avoids circularity - don't use the same model to judge its own CoT
            reference_tokenizer: Optional tokenizer for reference model (required if reference model
                               has different vocabulary, e.g., Llama vs Qwen)
            device: Device to run computations on
            think_pattern: Regex to extract CoT
            answer_pattern: Regex to extract final answer
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.reference_model = reference_model.to(device) if reference_model is not None else None
        self.reference_tokenizer = reference_tokenizer if reference_tokenizer is not None else tokenizer
        self.device = device
        self.think_pattern = re.compile(think_pattern, re.DOTALL)
        self.answer_pattern = re.compile(answer_pattern)

    def parse_output(self, output: str) -> tuple[str, str, str]:
        """
        Parse output into prompt, CoT, and answer regions.

        Returns:
            (prompt_text, cot_text, answer_text)
        """
        # Extract think (CoT)
        think_match = self.think_pattern.search(output)
        if think_match:
            cot_text = think_match.group(1).strip()
            # Everything before think is prompt, everything after is answer
            think_end = think_match.end()
            answer_region = output[think_end:].strip()
        else:
            cot_text = ""
            answer_region = output

        # Extract final answer
        answer_match = self.answer_pattern.search(answer_region)
        if answer_match:
            answer_text = answer_match.group(1).strip()
        else:
            answer_text = answer_region

        return "", cot_text, answer_text

    def create_masks(
        self,
        input_ids: torch.Tensor,
        texts: list[str],
    ) -> MaskSpec:
        """
        Create region masks for prompt, CoT, and answer.

        Args:
            input_ids: [batch, seq_len]
            texts: List of full text strings

        Returns:
            MaskSpec with all masks
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        prompt_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        cot_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        answer_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        full_mask = input_ids != self.tokenizer.pad_token_id

        for i, text in enumerate(texts):
            # Find think boundaries
            think_match = self.think_pattern.search(text)

            if think_match:
                think_start = think_match.start()
                think_end = think_match.end()

                # Tokenize regions to find boundaries
                prompt_text = text[:think_start]
                cot_with_tags_text = text[think_start:think_end]  # Include <think>...</think>
                answer_text = text[think_end:]

                # Get exact token positions by tokenizing each region
                prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
                cot_with_tags_tokens = self.tokenizer.encode(cot_with_tags_text, add_special_tokens=False)

                prompt_end = len(prompt_tokens)
                cot_end = prompt_end + len(cot_with_tags_tokens)

                # Create masks
                prompt_mask[i, :prompt_end] = True
                cot_mask[i, prompt_end:cot_end] = True
                answer_mask[i, cot_end:] = full_mask[i, cot_end:]
            else:
                # No think, treat everything as answer
                answer_mask[i] = full_mask[i]

        return MaskSpec(
            prompt_mask=prompt_mask,
            cot_mask=cot_mask,
            answer_mask=answer_mask,
            full_mask=full_mask,
        )

    def create_attention_mask(
        self,
        mask_spec: MaskSpec,
        mask_type: Literal["full", "via_cot", "no_prompt", "no_cot"],
    ) -> torch.Tensor:
        """
        Create causal attention mask with specified blocking pattern.

        Args:
            mask_spec: Region masks
            mask_type: Type of attention mask to create
                - "full": Standard causal attention
                - "via_cot": Block answer→prompt, allow CoT→prompt
                - "no_prompt": Block all attention to prompt
                - "no_cot": Block all attention to CoT

        Returns:
            attention_mask: [batch, seq_len, seq_len] as float (0.0=attend, -inf=ignore)
        """
        batch_size, seq_len = mask_spec.full_mask.shape
        device = mask_spec.full_mask.device

        # Start with causal mask as boolean for proper masking logic
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)

        if mask_type == "full":
            result_mask = causal_mask

        elif mask_type == "via_cot":
            # Block answer→prompt, but allow CoT→prompt
            # Attention mask[q, k] blocks query q from attending to key k
            from_answer = mask_spec.answer_mask.unsqueeze(2)  # [batch, seq_len, 1] - query dimension
            to_prompt = mask_spec.prompt_mask.unsqueeze(1)  # [batch, 1, seq_len] - key dimension
            block_mask = from_answer & to_prompt  # [batch, seq_len, seq_len]
            result_mask = causal_mask & ~block_mask

        elif mask_type == "no_prompt":
            # Block non-prompt tokens from attending to prompt
            # (but allow prompt tokens to attend to themselves)
            from_non_prompt = ~mask_spec.prompt_mask.unsqueeze(2)  # [batch, seq_len, 1] - query dimension
            to_prompt = mask_spec.prompt_mask.unsqueeze(1)  # [batch, 1, seq_len] - key dimension
            block_mask = from_non_prompt & to_prompt
            result_mask = causal_mask & ~block_mask

        elif mask_type == "no_cot":
            # Block non-CoT tokens from attending to CoT
            # (but allow CoT tokens to attend to themselves)
            from_non_cot = ~mask_spec.cot_mask.unsqueeze(2)  # [batch, seq_len, 1] - query dimension
            to_cot = mask_spec.cot_mask.unsqueeze(1)  # [batch, 1, seq_len] - key dimension
            block_mask = from_non_cot & to_cot
            result_mask = causal_mask & ~block_mask

        else:
            raise ValueError(f"Unknown mask_type: {mask_type}")

        # Convert boolean mask to float attention mask format (0.0 = attend, -inf = ignore)
        return torch.where(
            result_mask,
            torch.tensor(0.0, device=device, dtype=torch.float32),
            torch.tensor(float("-inf"), device=device, dtype=torch.float32)
        )

    @torch.no_grad()
    def get_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get hidden states from model with custom attention mask.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len, seq_len] - already in float format (0.0=attend, -inf=ignore)

        Returns:
            hidden_states: [batch, seq_len, hidden_dim]
        """
        # Expand to 4D: [batch, 1, seq_len, seq_len] for compatibility with transformers
        # The 1 will broadcast across all attention heads
        attention_mask_4d = attention_mask.unsqueeze(1)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask_4d,
            output_hidden_states=True,
        )

        # Return last layer hidden states
        return outputs.hidden_states[-1]

    def compute_kl_divergence(
        self,
        input_ids: torch.Tensor,
        mask_spec: MaskSpec,
        teacher_mask_type: Literal["full", "via_cot", "no_prompt", "no_cot"],
        student_mask_type: Literal["full", "via_cot", "no_prompt", "no_cot"],
    ) -> torch.Tensor:
        """
        Compute KL divergence between model outputs under different attention masks.

        KL(student || teacher) measures how much information is lost when
        using student_mask instead of teacher_mask.

        Args:
            input_ids: [batch, seq_len]
            mask_spec: Region masks
            teacher_mask_type: Attention mask for teacher (reference)
            student_mask_type: Attention mask for student (modified)

        Returns:
            kl_div: [batch] - KL divergence per sample (averaged over answer region)
        """
        # Get hidden states under both masks
        teacher_attn_mask = self.create_attention_mask(mask_spec, teacher_mask_type)
        student_attn_mask = self.create_attention_mask(mask_spec, student_mask_type)

        teacher_hidden = self.get_hidden_states(input_ids, teacher_attn_mask)
        student_hidden = self.get_hidden_states(input_ids, student_attn_mask)

        # Get logits
        teacher_logits = self.model.lm_head(teacher_hidden)  # [batch, seq_len, vocab]
        student_logits = self.model.lm_head(student_hidden)

        # Compute KL divergence
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        student_log_probs = F.log_softmax(student_logits, dim=-1)

        # KL(P||Q) = sum(P * log(P/Q)) = sum(P * (log(P) - log(Q)))
        kl_per_token = torch.sum(teacher_probs * (teacher_log_probs - student_log_probs), dim=-1)

        # Average over answer region
        answer_mask_float = mask_spec.answer_mask.float()
        kl_sum = (kl_per_token * answer_mask_float).sum(dim=1)
        answer_len = answer_mask_float.sum(dim=1).clamp(min=1)

        return kl_sum / answer_len

    def compute_js_divergence(
        self,
        input_ids: torch.Tensor,
        mask_spec: MaskSpec,
        mask_type_p: Literal["full", "via_cot", "no_prompt", "no_cot"],
        mask_type_q: Literal["full", "via_cot", "no_prompt", "no_cot"],
    ) -> torch.Tensor:
        """
        Compute JS divergence between model outputs under different attention masks.

        JS(P || Q) = (1/2) KL(P || M) + (1/2) KL(Q || M), M = (P + Q) / 2.
        Bounded in [0, log(2)] — avoids near-delta blow-up of forward KL.

        Returns:
            js_div: [batch] - JS divergence per sample (averaged over answer region)
        """
        attn_mask_p = self.create_attention_mask(mask_spec, mask_type_p)
        attn_mask_q = self.create_attention_mask(mask_spec, mask_type_q)

        hidden_p = self.get_hidden_states(input_ids, attn_mask_p)
        hidden_q = self.get_hidden_states(input_ids, attn_mask_q)

        logits_p = self.model.lm_head(hidden_p)
        logits_q = self.model.lm_head(hidden_q)

        log_p = F.log_softmax(logits_p, dim=-1)
        log_q = F.log_softmax(logits_q, dim=-1)
        p = log_p.exp()
        q = log_q.exp()

        m = (p + q) / 2
        log_m = torch.log(m.clamp(min=1e-40))

        js_per_token = 0.5 * (
            (p * (log_p - log_m)).sum(dim=-1) +
            (q * (log_q - log_m)).sum(dim=-1)
        )

        answer_mask_float = mask_spec.answer_mask.float()
        js_sum = (js_per_token * answer_mask_float).sum(dim=1)
        answer_len = answer_mask_float.sum(dim=1).clamp(min=1)

        return js_sum / answer_len

    def compute_gradient_norms(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        mask_spec: MaskSpec,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-token gradient norms w.r.t. answer loss.

        Gradient norm ||∂L/∂e_t|| measures how much token t contributes
        to the answer prediction.

        Args:
            input_ids: [batch, seq_len]
            labels: [batch, seq_len] - target tokens
            mask_spec: Region masks
            attention_mask: [batch, seq_len, seq_len]

        Returns:
            grad_norms: [batch, seq_len] - gradient norm per token
        """
        batch_size, seq_len = input_ids.shape

        # Get embeddings and enable gradients
        embeddings = self.model.get_input_embeddings()(input_ids)
        embeddings.requires_grad_(True)
        embeddings.retain_grad()  # Required because embeddings is not a leaf tensor

        # attention_mask is already in float format (0.0/-inf) from create_attention_mask
        # Expand to 4D for transformers: [batch, 1, seq_len, seq_len]
        attention_mask_4d = attention_mask.unsqueeze(1)

        outputs = self.model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask_4d,
            labels=labels,
        )

        # Compute loss only on answer region
        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_answer_mask = mask_spec.answer_mask[:, 1:]

        loss_fct = nn.CrossEntropyLoss(reduction='none')
        loss_per_token = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        ).view(batch_size, seq_len - 1)

        # Apply answer mask and take mean
        masked_loss = (loss_per_token * shift_answer_mask.float()).sum() / shift_answer_mask.float().sum().clamp(min=1)

        # Compute gradients
        masked_loss.backward()

        # Get gradient norms
        assert embeddings.grad is not None, "Gradients were not computed for embeddings"
        grad_norms = torch.norm(embeddings.grad, dim=-1)  # [batch, seq_len]

        return grad_norms

    def compute_entropy_nll(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        mask_spec: MaskSpec,
        mask_type: Literal["full", "via_cot", "no_prompt", "no_cot"],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute entropy and NLL under given attention mask.

        Args:
            input_ids: [batch, seq_len]
            labels: [batch, seq_len]
            mask_spec: Region masks
            mask_type: Attention mask type

        Returns:
            (entropy, nll): Both [batch] - averaged over answer region
        """
        attention_mask = self.create_attention_mask(mask_spec, mask_type)
        hidden = self.get_hidden_states(input_ids, attention_mask)
        logits = self.model.lm_head(hidden)

        # Compute entropy
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy_per_token = -torch.sum(probs * log_probs, dim=-1)

        # Compute NLL
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        nll_per_token = loss_fct(
            logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
            labels[:, 1:].contiguous().view(-1)
        ).view(logits.size(0), logits.size(1) - 1)

        # Pad to match seq_len
        nll_per_token = F.pad(nll_per_token, (0, 1), value=0)

        # Average over answer region
        answer_mask_float = mask_spec.answer_mask.float()
        entropy_sum = (entropy_per_token * answer_mask_float).sum(dim=1)
        nll_sum = (nll_per_token * answer_mask_float).sum(dim=1)
        answer_len = answer_mask_float.sum(dim=1).clamp(min=1)

        return entropy_sum / answer_len, nll_sum / answer_len

    @torch.no_grad()
    def compute_sufficiency(
        self,
        questions: list[str],
        cots: list[str],
        model: nn.Module | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute sufficiency H(A|C) using reference model to avoid circularity.

        This measures how informative the CoT is about the answer:
        - H(A|prompt+CoT): Entropy of answer given prompt and CoT
        - H(A|prompt): Entropy of answer given only prompt
        - Reduction: H(A|P) - H(A|C) = information gain from CoT

        Lower H(A|C) = more informative/sufficient CoT

        Args:
            questions: List of questions (prompts)
            cots: List of CoT texts
            model: Model to use for evaluation (defaults to reference_model if available, else self.model)

        Returns:
            (h_a_given_c, h_a_given_p, reduction): All [batch]
        """
        # Use reference model if available, otherwise use self.model
        eval_model = model if model is not None else (self.reference_model if self.reference_model is not None else self.model)

        # Use appropriate tokenizer
        eval_tokenizer = self.reference_tokenizer if eval_model is self.reference_model else self.tokenizer

        if eval_model is self.model and self.reference_model is None:
            import warnings
            warnings.warn(
                "Computing sufficiency with same model that generated CoT (circular). "
                "Consider providing a reference_model to avoid circularity."
            )

        # === Compute H(A|prompt+CoT) ===
        # Format: prompt + CoT, let model predict next tokens (answer region)
        prompt_and_cot_texts = [f"{q}\n<think>{c}</think>\nFinal answer:" for q, c in zip(questions, cots)]

        encoded_with_cot = eval_tokenizer(
            prompt_and_cot_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )

        input_ids_with_cot = encoded_with_cot["input_ids"].to(self.device)

        # Get logits at the answer position (after "Final answer:")
        outputs_with_cot = eval_model(input_ids=input_ids_with_cot)
        logits_with_cot = outputs_with_cot.logits[:, -1, :]  # [batch, vocab] - last position

        # Compute entropy H(A|P+C)
        probs_with_cot = F.softmax(logits_with_cot, dim=-1)
        log_probs_with_cot = F.log_softmax(logits_with_cot, dim=-1)
        h_a_given_c = -torch.sum(probs_with_cot * log_probs_with_cot, dim=-1)  # [batch]

        # === Compute H(A|prompt) ===
        # Format: prompt only, let model predict next tokens
        prompt_only_texts = [f"{q}\nFinal answer:" for q in questions]

        encoded_prompt_only = eval_tokenizer(
            prompt_only_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )

        input_ids_prompt_only = encoded_prompt_only["input_ids"].to(self.device)

        # Get logits at the answer position
        outputs_prompt_only = eval_model(input_ids=input_ids_prompt_only)
        logits_prompt_only = outputs_prompt_only.logits[:, -1, :]  # [batch, vocab]

        # Compute entropy H(A|P)
        probs_prompt_only = F.softmax(logits_prompt_only, dim=-1)
        log_probs_prompt_only = F.log_softmax(logits_prompt_only, dim=-1)
        h_a_given_p = -torch.sum(probs_prompt_only * log_probs_prompt_only, dim=-1)  # [batch]

        # === Compute information gain from CoT ===
        reduction = h_a_given_p - h_a_given_c  # Positive = CoT reduces uncertainty

        return h_a_given_c.cpu(), h_a_given_p.cpu(), reduction.cpu()

    @torch.no_grad()
    def compute_completeness_necessity(
        self,
        questions: list[str],
        cots: list[str],
        model: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Completeness and Necessity on a given model.

        Completeness (Direct Effect): DE = DKL(p(A|P,C) || p(A|C))
        - Measures how much the prompt directly affects answer beyond CoT
        - Lower = more complete (CoT contains all needed info from prompt)

        Necessity: NEC = DKL(p(A|P,C) || p(A|P))
        - Measures how much CoT contributes to answer beyond prompt
        - Higher = more necessary (CoT is essential)

        Args:
            questions: List of questions (prompts)
            cots: List of CoT texts
            model: Model to evaluate on

        Returns:
            (completeness, necessity): Both [batch]
        """
        # Use appropriate tokenizer based on which model is being evaluated
        eval_tokenizer = self.reference_tokenizer if model is self.reference_model else self.tokenizer

        # === Prepare inputs ===
        # Full context: prompt + CoT + "Final answer:"
        full_context_texts = [f"{q}\n<think>{c}</think>\nFinal answer:" for q, c in zip(questions, cots)]

        # CoT only: "<think>CoT</think>\nFinal answer:"
        cot_only_texts = [f"<think>{c}</think>\nFinal answer:" for c in cots]

        # Prompt only: "question\nFinal answer:"
        prompt_only_texts = [f"{q}\nFinal answer:" for q in questions]

        # === Tokenize ===
        encoded_full = eval_tokenizer(
            full_context_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        encoded_cot = eval_tokenizer(
            cot_only_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        encoded_prompt = eval_tokenizer(
            prompt_only_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )

        input_ids_full = encoded_full["input_ids"].to(self.device)
        input_ids_cot = encoded_cot["input_ids"].to(self.device)
        input_ids_prompt = encoded_prompt["input_ids"].to(self.device)

        # === Get distributions at answer position ===
        # p(A|P,C)
        outputs_full = model(input_ids=input_ids_full)
        logits_full = outputs_full.logits[:, -1, :]  # [batch, vocab]
        p_a_given_pc = F.softmax(logits_full, dim=-1)
        log_p_a_given_pc = F.log_softmax(logits_full, dim=-1)

        # p(A|C)
        outputs_cot = model(input_ids=input_ids_cot)
        logits_cot = outputs_cot.logits[:, -1, :]
        log_p_a_given_c = F.log_softmax(logits_cot, dim=-1)

        # p(A|P)
        outputs_prompt = model(input_ids=input_ids_prompt)
        logits_prompt = outputs_prompt.logits[:, -1, :]
        log_p_a_given_p = F.log_softmax(logits_prompt, dim=-1)

        # === Compute KL divergences ===
        # Completeness: DKL(p(A|P,C) || p(A|C))
        completeness = torch.sum(p_a_given_pc * (log_p_a_given_pc - log_p_a_given_c), dim=-1)

        # Necessity: DKL(p(A|P,C) || p(A|P))
        necessity = torch.sum(p_a_given_pc * (log_p_a_given_pc - log_p_a_given_p), dim=-1)

        return completeness.cpu(), necessity.cpu()

    def compute_all_metrics(
        self,
        questions: list[str],
        outputs: list[str],
        batch_size: int = 8,
    ) -> FaithfulnessMetrics:
        """
        Compute all faithfulness metrics for a batch of samples.

        Args:
            questions: List of question strings
            outputs: List of model outputs (with CoT)
            batch_size: Batch size for processing

        Returns:
            FaithfulnessMetrics with all computed metrics
        """
        # Combine question + output
        full_texts = [f"{q}\n{o}" for q, o in zip(questions, outputs)]

        # Tokenize
        encoded = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )

        input_ids = encoded["input_ids"].to(self.device)
        labels = input_ids.clone()

        # Create masks
        mask_spec = self.create_masks(input_ids, full_texts)

        # === KL-based metrics ===
        kl_direct_effect = self.compute_kl_divergence(
            input_ids, mask_spec, teacher_mask_type="full", student_mask_type="via_cot"
        )

        kl_cot_necessity = self.compute_kl_divergence(
            input_ids, mask_spec, teacher_mask_type="full", student_mask_type="no_cot"
        )

        kl_leakage = self.compute_kl_divergence(
            input_ids, mask_spec, teacher_mask_type="via_cot", student_mask_type="no_prompt"
        )

        js_direct_effect = self.compute_js_divergence(
            input_ids, mask_spec, mask_type_p="full", mask_type_q="via_cot"
        )

        js_cot_necessity = self.compute_js_divergence(
            input_ids, mask_spec, mask_type_p="full", mask_type_q="no_cot"
        )

        # === Gradient-based metrics ===
        # Direct effect: prompt contribution under full attention
        full_attn_mask = self.create_attention_mask(mask_spec, "full")
        grad_norms_full = self.compute_gradient_norms(input_ids, labels, mask_spec, full_attn_mask)

        prompt_sum = (grad_norms_full * mask_spec.prompt_mask.float()).sum(dim=1)
        total_sum = grad_norms_full.sum(dim=1).clamp(min=1)
        grad_de_l1 = prompt_sum / total_sum

        prompt_sq_sum = (grad_norms_full.pow(2) * mask_spec.prompt_mask.float()).sum(dim=1)
        total_sq_sum = grad_norms_full.pow(2).sum(dim=1).clamp(min=1)
        grad_de_l2 = prompt_sq_sum / total_sq_sum

        # CoT necessity: CoT contribution under full attention
        cot_sum = (grad_norms_full * mask_spec.cot_mask.float()).sum(dim=1)
        grad_cot_necessity_l1 = cot_sum / total_sum

        cot_sq_sum = (grad_norms_full.pow(2) * mask_spec.cot_mask.float()).sum(dim=1)
        grad_cot_necessity_l2 = cot_sq_sum / total_sq_sum

        # Leakage: prompt contribution under via_cot attention
        via_cot_attn_mask = self.create_attention_mask(mask_spec, "via_cot")
        grad_norms_via_cot = self.compute_gradient_norms(input_ids, labels, mask_spec, via_cot_attn_mask)

        prompt_sum_via_cot = (grad_norms_via_cot * mask_spec.prompt_mask.float()).sum(dim=1)
        total_sum_via_cot = grad_norms_via_cot.sum(dim=1).clamp(min=1)
        grad_leakage_l1 = prompt_sum_via_cot / total_sum_via_cot

        prompt_sq_sum_via_cot = (grad_norms_via_cot.pow(2) * mask_spec.prompt_mask.float()).sum(dim=1)
        total_sq_sum_via_cot = grad_norms_via_cot.pow(2).sum(dim=1).clamp(min=1)
        grad_leakage_l2 = prompt_sq_sum_via_cot / total_sq_sum_via_cot

        # === Entropy/NLL metrics ===
        entropy_full, nll_full = self.compute_entropy_nll(input_ids, labels, mask_spec, "full")
        entropy_via_cot, nll_via_cot = self.compute_entropy_nll(input_ids, labels, mask_spec, "via_cot")
        entropy_no_prompt, nll_no_prompt = self.compute_entropy_nll(input_ids, labels, mask_spec, "no_prompt")

        # === Extract CoTs from outputs (used by sufficiency, completeness, necessity) ===
        cots = []
        for output in outputs:
            _, cot_text, _ = self.parse_output(output)
            cots.append(cot_text)

        # === Sufficiency metrics H(A|C) using reference model ===
        sufficiency_h_a_given_c = None
        sufficiency_h_a_given_p = None
        sufficiency_reduction = None

        if self.reference_model is not None:
            # Compute sufficiency using reference model
            sufficiency_h_a_given_c, sufficiency_h_a_given_p, sufficiency_reduction = self.compute_sufficiency(
                questions=questions,
                cots=cots,
            )

        # === Completeness and Necessity metrics ===
        completeness_generating_model = None
        completeness_reference_model = None
        necessity_generating_model = None
        necessity_reference_model = None

        # Compute on generating model (self.model)
        completeness_generating_model, necessity_generating_model = self.compute_completeness_necessity(
            questions=questions,
            cots=cots,
            model=self.model
        )

        # Compute on reference model if available
        if self.reference_model is not None:
            completeness_reference_model, necessity_reference_model = self.compute_completeness_necessity(
                questions=questions,
                cots=cots,
                model=self.reference_model
            )

        return FaithfulnessMetrics(
            kl_direct_effect=kl_direct_effect.cpu(),
            kl_cot_necessity=kl_cot_necessity.cpu(),
            kl_leakage=kl_leakage.cpu(),
            js_direct_effect=js_direct_effect.cpu(),
            js_cot_necessity=js_cot_necessity.cpu(),
            grad_de_l1=grad_de_l1.cpu(),
            grad_de_l2=grad_de_l2.cpu(),
            grad_cot_necessity_l1=grad_cot_necessity_l1.cpu(),
            grad_cot_necessity_l2=grad_cot_necessity_l2.cpu(),
            grad_leakage_l1=grad_leakage_l1.cpu(),
            grad_leakage_l2=grad_leakage_l2.cpu(),
            entropy_full=entropy_full.cpu(),
            entropy_via_cot=entropy_via_cot.cpu(),
            entropy_no_prompt=entropy_no_prompt.cpu(),
            nll_full=nll_full.cpu(),
            nll_via_cot=nll_via_cot.cpu(),
            nll_no_prompt=nll_no_prompt.cpu(),
            sufficiency_h_a_given_c=sufficiency_h_a_given_c,
            sufficiency_h_a_given_p=sufficiency_h_a_given_p,
            sufficiency_reduction=sufficiency_reduction,
            completeness_generating_model=completeness_generating_model,
            completeness_reference_model=completeness_reference_model,
            necessity_generating_model=necessity_generating_model,
            necessity_reference_model=necessity_reference_model,
        )


# ============================================================================
# CLI Example
# ============================================================================

def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Compute CoT faithfulness metrics")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--questions", type=str, nargs="+", required=True, help="Questions")
    parser.add_argument("--outputs", type=str, nargs="+", required=True, help="Model outputs")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    args = parser.parse_args()

    # Load model
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Create evaluator
    evaluator = FaithfulnessEvaluator(model, tokenizer, device=args.device)

    # Compute metrics
    print("\nComputing faithfulness metrics...")
    metrics = evaluator.compute_all_metrics(args.questions, args.outputs)

    # Print results
    print("\n" + "=" * 60)
    print("FAITHFULNESS METRICS")
    print("=" * 60)

    results = metrics.to_dict()
    for key, value in results.items():
        print(f"{key:30s}: {value:.4f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
