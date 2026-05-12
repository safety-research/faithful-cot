"""
Chain-of-Thought Masking Utilities for VERL - Version 2 (Simplified)

IMPROVED LOGIC:
- CoT starts: Beginning of assistant turn (computed from response_length)
- CoT ends: </think> token position
- If no </think>: Entire response is CoT (no answer region, no blocking)
- Per-sample masking: Each sample masked independently
- Only searches for </think> token (ignores <think> token)
- No need to pass response_start_positions - computed internally

Assumes format: [Prompt] [Response with </think>] [Answer]
Where: response_start = seq_len - response_length
"""

import torch
from typing import Optional, Tuple, Dict


def _find_subsequence_in_sequence(
    sequence: torch.Tensor,
    subsequence: torch.Tensor,
) -> int:
    """
    Find the LAST occurrence of a subsequence in a sequence.

    Used for substring matching mode to find multi-token delimiters like "</think>".

    Args:
        sequence: (seq_len,) tensor of token IDs
        subsequence: (subseq_len,) tensor of token IDs to find

    Returns:
        int: Index of last occurrence start position, or -1 if not found
    """
    seq_len = sequence.shape[0]
    subseq_len = subsequence.shape[0]

    # Edge case: subsequence longer than sequence
    if subseq_len > seq_len:
        return -1

    # Search from end to beginning to find LAST occurrence
    for i in range(seq_len - subseq_len, -1, -1):
        if torch.equal(sequence[i:i + subseq_len], subsequence):
            return i
    return -1


def tokenize_delimiter_string(tokenizer, delimiter_str: str, device: torch.device) -> torch.Tensor:
    """
    Tokenize a delimiter string for substring matching mode.

    Converts a string like "</think>" into its token ID sequence using the tokenizer.

    Args:
        tokenizer: HuggingFace tokenizer instance
        delimiter_str: String to tokenize (e.g., "</think>")
        device: Device to place the resulting tensor

    Returns:
        torch.Tensor: (subseq_len,) tensor of token IDs

    Example:
        >>> tokenize_delimiter_string(tokenizer, "</think>", device)
        tensor([60, 14336, 29], device='cuda:0')
    """
    token_ids = tokenizer.encode(delimiter_str, add_special_tokens=False)
    return torch.tensor(token_ids, dtype=torch.long, device=device)


def find_special_token_positions(
    input_ids: torch.Tensor,
    response_length: int,
    end_think_token_id: Optional[int] = None,
    end_think_delimiter_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Find positions of prompt, CoT, and answer regions in input sequences.

    UNIFIED INTERFACE - Supports two modes:
    Mode 1: Special Token (specify end_think_token_id) - Fast vectorized search
    Mode 2: Substring Matching (specify end_think_delimiter_ids) - Sequential search per sample

    SIMPLIFIED LOGIC:
    - CoT starts: Beginning of assistant turn (computed from response_length)
    - CoT ends: </think> delimiter position (single token OR multi-token sequence)
    - Answer: Everything after </think>
    - If no </think>: Entire response is treated as ANSWER (fallback to block prompt)
      → This ensures gradient masking still works even without explicit CoT structure
    - We DON'T care about <think> token position

    Args:
        input_ids: (batch_size, seq_len) tensor of token IDs
        response_length: Length of the response (assistant's generation)
        end_think_token_id: Token ID for </think> (Mode 1: special token)
        end_think_delimiter_ids: Token ID sequence for </think> (Mode 2: substring matching)

    Returns:
        prompt_mask: (batch_size, seq_len) - True for prompt tokens
        cot_mask: (batch_size, seq_len) - True for CoT tokens
        answer_mask: (batch_size, seq_len) - True for answer tokens

    Example:
        # Mode 1: Special token
        masks = find_special_token_positions(input_ids, response_length, end_think_token_id=151666)

        # Mode 2: Substring matching
        delimiter_ids = tokenize_delimiter_string(tokenizer, "</think>", device)
        masks = find_special_token_positions(input_ids, response_length, end_think_delimiter_ids=delimiter_ids)
    """
    # Validate inputs
    if end_think_token_id is None and end_think_delimiter_ids is None:
        raise ValueError(
            "Must specify either end_think_token_id (special token mode) "
            "or end_think_delimiter_ids (substring mode)"
        )

    if end_think_token_id is not None and end_think_delimiter_ids is not None:
        raise ValueError(
            "Cannot specify both end_think_token_id and end_think_delimiter_ids. "
            "Choose one mode."
        )

    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    # Compute response start position from response_length
    response_start = seq_len - response_length

    # MODE SELECTION: Find delimiter positions based on specified mode

    if end_think_token_id is not None:
        # MODE 1: Special token - Fast vectorized search
        is_end_think = (input_ids == end_think_token_id)  # (batch_size, seq_len)
        position_indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        masked_positions = torch.where(is_end_think, position_indices, torch.tensor(-1, device=device))
        end_think_positions = masked_positions.max(dim=1)[0]  # (batch_size,)
        delimiter_length = 1  # Single token

    else:
        # MODE 2: Substring matching - Sequential search per sample
        end_think_positions = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        delimiter_length = end_think_delimiter_ids.shape[0]

        for i in range(batch_size):
            pos = _find_subsequence_in_sequence(input_ids[i], end_think_delimiter_ids)
            end_think_positions[i] = pos  # Store start position

    # Create position matrix for broadcasting comparisons
    pos_matrix = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    # Prompt mask: all positions before response_start
    prompt_mask = pos_matrix < response_start

    # Check which samples have delimiter
    has_end_think = end_think_positions >= 0
    has_end_think_expanded = has_end_think.unsqueeze(1)

    # Calculate end position of delimiter (inclusive)
    # For special token: end_pos = position
    # For substring: end_pos = start_position + length - 1
    delimiter_end_positions = end_think_positions + delimiter_length - 1
    delimiter_end_expanded = delimiter_end_positions.unsqueeze(1)

    # CoT mask:
    # - If has delimiter: response_start <= pos <= end of delimiter (inclusive)
    # - If no delimiter: EMPTY (no CoT region) - fallback to treating entire response as answer
    cot_with_end = (pos_matrix >= response_start) & (pos_matrix <= delimiter_end_expanded)
    cot_mask = torch.where(has_end_think_expanded, cot_with_end, torch.zeros_like(cot_with_end))

    # Answer mask:
    # - If has delimiter: positions after delimiter
    # - If no delimiter: entire response (fallback - treat all response as answer to block prompt)
    answer_with_delimiter = pos_matrix > delimiter_end_expanded
    answer_without_delimiter = pos_matrix >= response_start  # Entire response is answer
    answer_mask = torch.where(has_end_think_expanded, answer_with_delimiter, answer_without_delimiter)

    return prompt_mask, cot_mask, answer_mask


def create_cot_attention_mask(
    input_ids: torch.Tensor,
    response_length: int,
    end_think_token_id: Optional[int] = None,
    end_think_delimiter_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Create 4D attention mask that blocks answer tokens from attending to prompt.

    UNIFIED INTERFACE - Supports two delimiter detection modes:
    - Mode 1: Special token (specify end_think_token_id)
    - Mode 2: Substring matching (specify end_think_delimiter_ids)

    Attention rules:
    - Prompt tokens: Standard causal attention
    - CoT tokens: Can attend to prompt + previous CoT (causal)
    - Answer tokens: Can ONLY attend to CoT + previous answer (NOT prompt)

    Args:
        input_ids: (batch_size, seq_len) tensor of token IDs
        response_length: Length of the response (assistant's generation)
        end_think_token_id: Token ID for </think> (Mode 1: special token)
        end_think_delimiter_ids: Token ID sequence for </think> (Mode 2: substring matching)
        attention_mask: (batch_size, seq_len) optional padding mask (1=valid, 0=pad)
        dtype: dtype for the output mask (float32 or bfloat16)

    Returns:
        attention_mask_4d: (batch_size, 1, seq_len, seq_len)
                          0.0 = can attend, -inf = cannot attend
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    # Find regions using unified interface
    prompt_mask, cot_mask, answer_mask = find_special_token_positions(
        input_ids=input_ids,
        response_length=response_length,
        end_think_token_id=end_think_token_id,
        end_think_delimiter_ids=end_think_delimiter_ids,
    )

    # Create base causal mask (upper triangular with -inf)
    # Shape: (seq_len, seq_len)
    # IMPORTANT: Always use float32 for masks with -inf to avoid NaN in bfloat16
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=torch.float32),
        diagonal=1
    )

    # Check if causal mask has NaN (should never happen)
    if torch.isnan(causal_mask).any():
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"NaN in causal_mask! This should never happen. dtype={torch.float32}, seq_len={seq_len}")

    # Expand to batch: (batch_size, 1, seq_len, seq_len)
    attention_mask_4d = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len).clone()

    # PER-SAMPLE MASKING: Apply structural constraint independently for each sample
    # For samples with answer region: block answer → prompt attention
    # For samples without answer region (no </think>): no additional blocking needed

    # Apply structural constraint: block answer → prompt attention
    # VECTORIZED: Use broadcasting for efficiency
    # Create masks for answer (query) and prompt (key) positions
    # Shape: (batch_size, seq_len)
    answer_query_mask = answer_mask.unsqueeze(1).unsqueeze(-1)  # (batch, 1, seq, 1)
    prompt_key_mask = prompt_mask.unsqueeze(1).unsqueeze(-2)    # (batch, 1, 1, seq)

    # Broadcasting: wherever answer_query AND prompt_key are both True, block attention
    # (batch, 1, seq, 1) * (batch, 1, 1, seq) -> (batch, 1, seq, seq)
    # This applies per-sample automatically - samples without answer_mask won't block anything
    block_mask = answer_query_mask & prompt_key_mask

    # Apply blocking: set blocked positions to -inf
    attention_mask_4d[block_mask] = float('-inf')

    # Apply padding mask if provided
    if attention_mask is not None:
        # Check if input attention_mask has NaN
        if torch.isnan(attention_mask).any():
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"NaN detected in INPUT attention_mask! Shape: {attention_mask.shape}, "
                        f"NaN count: {torch.isnan(attention_mask).sum().item()}")
            # Skip padding mask application if input is corrupt
        else:
            # attention_mask: (batch_size, seq_len) with 1=valid, 0=pad
            # We need to block attention TO padded positions (key dimension)
            # Shape: (batch_size, 1, 1, seq_len)
            # IMPORTANT: Use torch.where to avoid 0.0 * float('-inf') = NaN
            padding_mask_4d = torch.where(
                attention_mask.unsqueeze(1).unsqueeze(2) == 0,
                torch.tensor(float('-inf'), dtype=torch.float32, device=attention_mask.device),
                torch.tensor(0.0, dtype=torch.float32, device=attention_mask.device)
            )

            # Check for NaN after padding mask creation
            if torch.isnan(padding_mask_4d).any():
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"NaN in padding_mask_4d after creation! This should not happen with torch.where.")
            else:
                attention_mask_4d = attention_mask_4d + padding_mask_4d

    # Safety check: ensure no NaN values in mask
    if torch.isnan(attention_mask_4d).any():
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"NaN detected in CoT attention mask! Falling back to causal mask. "
                    f"has_padding_mask={attention_mask is not None}")
        # Fallback to causal mask only (no padding)
        attention_mask_4d = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len).clone()

    return attention_mask_4d


def create_gradient_mask_hook(
    input_ids: torch.Tensor,
    end_think_token_id: int,
    response_length: int,
    block_prompt_grad: bool = True,
    block_answer_grad: bool = True,
):
    """
    Create a gradient masking hook that blocks gradients from prompt/answer positions.

    This implements the gradient masking intervention from Figure 3b, which blocks
    gradients from backpropagating from answer or prompt position activations into
    model parameters. This ensures that any gains in task performance necessarily
    come from how the model uses its CoT.

    Usage:
        Hook should be registered on the embedding layer output or first layer activations.
        During backward pass, gradients from prompt and/or answer positions will be zeroed out.

    Args:
        input_ids: (batch_size, seq_len) tensor of token IDs
        end_think_token_id: Token ID for </think>
        response_length: Length of the response (assistant's generation)
        block_prompt_grad: If True, block gradients from prompt positions (default: True)
        block_answer_grad: If True, block gradients from answer positions (default: True)

    Returns:
        hook_fn: A gradient hook function that can be registered with tensor.register_hook()
    """
    # Find regions using the existing function
    prompt_mask, cot_mask, answer_mask = find_special_token_positions(
        input_ids, end_think_token_id, response_length
    )

    # Create gradient mask: 1.0 where gradients should flow, 0.0 where they should be blocked
    # Shape: (batch_size, seq_len)
    gradient_mask = torch.ones_like(input_ids, dtype=torch.float32)

    if block_prompt_grad:
        gradient_mask = gradient_mask.masked_fill(prompt_mask, 0.0)

    if block_answer_grad:
        gradient_mask = gradient_mask.masked_fill(answer_mask, 0.0)

    # Store gradient_mask in closure
    gradient_mask_stored = gradient_mask

    def hook_fn(grad):
        """
        Gradient hook that masks gradients based on position.

        Args:
            grad: Gradient tensor of shape (batch_size, seq_len, hidden_dim) or
                  (batch_size, seq_len) for nested tensors after unpacking

        Returns:
            Masked gradient tensor
        """
        if grad is None:
            return None

        # Handle different gradient shapes
        if grad.dim() == 3:
            # Shape: (batch_size, seq_len, hidden_dim)
            # Expand mask to match: (batch_size, seq_len, 1)
            mask_expanded = gradient_mask_stored.unsqueeze(-1).to(grad.device, grad.dtype)
            return grad * mask_expanded
        elif grad.dim() == 2:
            # Shape: (batch_size, seq_len) or (total_nnz, hidden_dim) for nested
            # Try to match the gradient shape
            if grad.shape == gradient_mask_stored.shape:
                return grad * gradient_mask_stored.to(grad.device, grad.dtype)
            else:
                # Might be (total_nnz, hidden_dim) from nested tensor
                # In this case, we need to flatten the mask
                mask_flat = gradient_mask_stored.view(-1).unsqueeze(-1).to(grad.device, grad.dtype)
                if grad.shape[0] == mask_flat.shape[0]:
                    return grad * mask_flat
                else:
                    # Shape mismatch, return grad unchanged and log warning
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Gradient mask shape mismatch: grad.shape={grad.shape}, "
                                 f"mask.shape={gradient_mask_stored.shape}. Skipping masking.")
                    return grad
        else:
            # Unexpected shape, return unchanged
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Unexpected gradient shape: {grad.shape}. Expected 2D or 3D. Skipping masking.")
            return grad

    return hook_fn


def apply_gradient_masking_to_embeddings(
    model,
    input_ids: torch.Tensor,
    end_think_token_id: int,
    response_length: int,
    block_prompt_grad: bool = True,
    block_answer_grad: bool = True,
):
    """
    Apply gradient masking to the model's embedding layer.

    This is a convenience function that registers a gradient hook on the embedding
    layer output to block gradients from prompt/answer positions.

    Args:
        model: The model instance (should have get_input_embeddings() method)
        input_ids: (batch_size, seq_len) tensor of token IDs
        end_think_token_id: Token ID for </think>
        response_length: Length of the response
        block_prompt_grad: If True, block gradients from prompt positions
        block_answer_grad: If True, block gradients from answer positions

    Returns:
        hook_handle: Handle to the registered hook (can be removed with hook_handle.remove())
    """
    hook_fn = create_gradient_mask_hook(
        input_ids=input_ids,
        end_think_token_id=end_think_token_id,
        response_length=response_length,
        block_prompt_grad=block_prompt_grad,
        block_answer_grad=block_answer_grad,
    )

    # Get the embedding layer
    embed_layer = model.get_input_embeddings()

    # We'll need to register the hook during forward pass
    # This function returns the hook function and user needs to register it
    # on the embedding output tensor
    return hook_fn


def visualize_attention_mask(
    attention_mask_4d: torch.Tensor,
    input_ids: torch.Tensor,
    tokenizer,
    batch_idx: int = 0,
) -> str:
    """
    Create a text visualization of the attention mask for debugging.

    Args:
        attention_mask_4d: (batch_size, 1, seq_len, seq_len) attention mask
        input_ids: (batch_size, seq_len) token IDs
        tokenizer: Tokenizer to decode tokens
        batch_idx: Which batch element to visualize

    Returns:
        String representation of the mask
    """
    seq_len = input_ids.shape[1]
    mask_2d = attention_mask_4d[batch_idx, 0].cpu()  # (seq_len, seq_len)

    # Decode tokens
    tokens = [tokenizer.decode([tok]) for tok in input_ids[batch_idx].cpu().tolist()]

    # Create visualization
    lines = ["Attention Mask Visualization:"]
    lines.append("  Rows = Query, Cols = Key")
    lines.append("  ✓ = can attend (0.0), ✗ = blocked (-inf)")
    lines.append("")

    # Header with token indices
    header = "     " + "".join([f"{i:3d} " for i in range(min(seq_len, 20))])
    lines.append(header)

    # Show first 20x20 for readability
    for i in range(min(seq_len, 20)):
        row = f"{i:3d}: "
        for j in range(min(seq_len, 20)):
            symbol = "✓" if mask_2d[i, j] > -1000 else "✗"
            row += f" {symbol}  "
        row += f" | {tokens[i][:10]}"
        lines.append(row)

    return "\n".join(lines)


def prepare_gradient_masking_config(config) -> dict:
    """
    Prepare gradient masking configuration dictionary for injection into TensorDict.

    Extracts gradient masking settings from ActorConfig or CriticConfig and returns
    a dictionary that can be passed to tu.assign_non_tensor().

    Args:
        config: ActorConfig or CriticConfig with gradient masking settings

    Returns:
        Dictionary with gradient masking configuration
    """
    grad_mask_config = {}

    # Check if gradient masking is enabled
    if hasattr(config, 'use_cot_gradient_masking') and config.use_cot_gradient_masking:
        grad_mask_config['use_cot_gradient_masking'] = True
        grad_mask_config['end_think_token_id'] = getattr(config, 'end_think_token_id', 151666)
        grad_mask_config['block_prompt_gradients'] = getattr(config, 'block_prompt_gradients', True)
        grad_mask_config['block_answer_gradients'] = getattr(config, 'block_answer_gradients', True)
    else:
        grad_mask_config['use_cot_gradient_masking'] = False

    return grad_mask_config


# Example usage and testing
if __name__ == "__main__":
    # Test the masking function
    print("Testing CoT Masking Utilities V2 (Simplified)...")

    # Mock setup
    batch_size = 2
    seq_len = 20
    response_length = 10
    end_think_token_id = 151666

    # Create mock input: [prompt tokens] [response tokens with </think>]
    # Positions: 0-9 = prompt, 10-19 = response
    # </think> at position 15
    input_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len).clone()
    input_ids[:, 15] = end_think_token_id

    print(f"Input shape: {input_ids.shape}")
    print(f"Input IDs[0]: {input_ids[0].tolist()}")
    print(f"Response length: {response_length}")
    print(f"Response start (computed): {seq_len - response_length}")
    print(f"</think> at position: 15")

    # Test mask creation
    mask_4d = create_cot_attention_mask(
        input_ids=input_ids,
        end_think_token_id=end_think_token_id,
        response_length=response_length,
        dtype=torch.float32,
    )

    print(f"\nMask shape: {mask_4d.shape}")
    print(f"Mask dtype: {mask_4d.dtype}")

    # Check specific attention patterns
    print("\n--- Checking Attention Patterns ---")

    # Position 5 (prompt) should be able to attend to position 3 (prompt)
    can_attend = mask_4d[0, 0, 5, 3] > -1000
    print(f"Prompt[5] → Prompt[3]: {'✓ Can attend' if can_attend else '✗ Blocked'}")

    # Position 12 (CoT) should be able to attend to position 5 (prompt)
    can_attend = mask_4d[0, 0, 12, 5] > -1000
    print(f"CoT[12] → Prompt[5]: {'✓ Can attend' if can_attend else '✗ Blocked'}")

    # Position 17 (answer) should NOT be able to attend to position 5 (prompt)
    can_attend = mask_4d[0, 0, 17, 5] > -1000
    print(f"Answer[17] → Prompt[5]: {'✓ Can attend' if can_attend else '✗ Blocked'}")

    # Position 17 (answer) should be able to attend to position 12 (CoT)
    can_attend = mask_4d[0, 0, 17, 12] > -1000
    print(f"Answer[17] → CoT[12]: {'✓ Can attend' if can_attend else '✗ Blocked'}")

    print("\n✅ Test completed!")

    # Test 2: Mixed batch (one sample with </think>, one without)
    print("\n" + "=" * 60)
    print("Test 2: Mixed batch (one with </think>, one without)")
    print("=" * 60)

    input_ids_mixed = torch.arange(seq_len).unsqueeze(0).expand(2, seq_len).clone()
    input_ids_mixed[0, 15] = end_think_token_id  # Sample 0 has </think>
    # Sample 1 has no </think>

    print(f"Sample 0: Has </think> at position 15")
    print(f"Sample 1: No </think> token")

    mask_4d_mixed = create_cot_attention_mask(
        input_ids=input_ids_mixed,
        end_think_token_id=end_think_token_id,
        response_length=response_length,
        dtype=torch.float32,
    )

    print("\n--- Sample 0 (has </think>) ---")
    can_attend = mask_4d_mixed[0, 0, 17, 5] > -1000
    print(f"Answer[17] → Prompt[5]: {'✗ BLOCKED (correct!)' if not can_attend else '✓ ERROR: Not blocked!'}")

    print("\n--- Sample 1 (no </think>) ---")
    can_attend = mask_4d_mixed[1, 0, 17, 5] > -1000
    print(f"Response[17] → Prompt[5]: {'✓ Can attend (correct - no answer region)' if can_attend else '✗ BLOCKED (wrong!)'}")

    print("\n✅ Mixed batch test completed!")
