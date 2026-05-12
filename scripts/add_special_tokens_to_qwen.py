#!/usr/bin/env python3
"""
Add special tokens (<think>, </think>) to Qwen2.5-1.5B-Instruct model.

Usage:
    python scripts/add_special_tokens_to_qwen.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import argparse


def add_special_tokens_to_model(
    model_path: str = "Qwen/Qwen2.5-1.5B-Instruct",
    output_path: str = "./models/qwen2.5-1.5b-instruct-think",
    special_tokens: list = None,
    dtype: torch.dtype = torch.bfloat16,
):
    """
    Add special tokens to a pretrained model and save.

    Args:
        model_path: HuggingFace model path or local path
        output_path: Where to save the modified model
        special_tokens: List of special tokens to add
        dtype: Model dtype (bfloat16, float16, or float32)
    """
    if special_tokens is None:
        special_tokens = ["<think>", "</think>"]

    print("="*60)
    print("Adding Special Tokens to Model")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Output: {output_path}")
    print(f"Special tokens: {special_tokens}")
    print(f"Dtype: {dtype}")
    print()

    # ========================================
    # Step 1: Load tokenizer
    # ========================================
    print("[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    original_vocab_size = len(tokenizer)
    print(f"  Original vocab size: {original_vocab_size}")

    # ========================================
    # Step 2: Add special tokens
    # ========================================
    print("\n[2/6] Adding special tokens...")

    # Add as special tokens (ensures single-token encoding)
    num_added = tokenizer.add_tokens(special_tokens, special_tokens=True)

    if num_added == 0:
        print("  ⚠️  Warning: No tokens were added (already exist)")
    else:
        print(f"  ✅ Added {num_added} tokens")

    new_vocab_size = len(tokenizer)
    print(f"  New vocab size: {new_vocab_size}")

    # Get token IDs
    token_ids = {token: tokenizer.convert_tokens_to_ids(token) for token in special_tokens}
    for token, token_id in token_ids.items():
        print(f"  {token:15s} → ID: {token_id}")

    # ========================================
    # Step 3: Verify tokenization consistency
    # ========================================
    print("\n[3/6] Verifying tokenization consistency...")

    test_texts = [
        "<think>",
        " <think>",
        "\n<think>",
        "Question <think>",
        "<think>reasoning</think>",
    ]

    all_consistent = True
    for text in test_texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        tokens_str = tokenizer.convert_ids_to_tokens(tokens)

        # Check if <think> appears as single token
        think_count = tokens.count(token_ids['<think>'])

        print(f"  '{text:25s}' → {tokens}")

        if '<think>' in text and think_count != text.count('<think>'):
            print(f"    ⚠️  Warning: <think> not consistently tokenized")
            all_consistent = False

    if all_consistent:
        print("  ✅ All special tokens tokenize consistently")

    # ========================================
    # Step 4: Load model
    # ========================================
    print("\n[4/6] Loading model...")
    print(f"  This may take a few minutes...")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    )

    original_embed_size = model.get_input_embeddings().weight.shape[0]
    print(f"  Original embedding size: {original_embed_size}")
    print(f"  Model dtype: {model.dtype}")
    print(f"  Model device: {model.device}")

    # ========================================
    # Step 5: Resize embeddings and initialize
    # ========================================
    print("\n[5/6] Resizing model embeddings...")

    if original_embed_size == new_vocab_size:
        print("  ℹ️  No resize needed (embeddings already match)")
    else:
        # Resize
        model.resize_token_embeddings(new_vocab_size)
        print(f"  ✅ Resized embeddings: {original_embed_size} → {new_vocab_size}")

        # Initialize new token embeddings
        print("\n  Initializing new token embeddings...")

        with torch.no_grad():
            input_embeddings = model.get_input_embeddings()
            output_embeddings = model.get_output_embeddings()

            # Strategy: Initialize from similar existing tokens
            # For <think>, use embedding of "think" token(s)

            think_word_ids = tokenizer.encode("think", add_special_tokens=False)
            print(f"    'think' token IDs: {think_word_ids}")

            if len(think_word_ids) > 0:
                # Use mean of "think" token embeddings
                related_embeddings = input_embeddings.weight[think_word_ids]
                init_embedding = related_embeddings.mean(dim=0)
                print(f"    Using mean of 'think' token embeddings")
            else:
                # Fallback: use mean of all existing embeddings
                init_embedding = input_embeddings.weight[:original_vocab_size].mean(dim=0)
                print(f"    Using mean of all existing embeddings (fallback)")

            # Initialize all new tokens
            for token, token_id in token_ids.items():
                if token_id >= original_vocab_size:
                    input_embeddings.weight[token_id] = init_embedding.clone()
                    if output_embeddings is not None:
                        output_embeddings.weight[token_id] = init_embedding.clone()
                    print(f"    ✅ Initialized embeddings for {token} (ID: {token_id})")

        # Verify initialization
        new_token_id = token_ids[special_tokens[0]]
        new_embedding = input_embeddings.weight[new_token_id]
        is_zero = (new_embedding == 0).all().item()
        is_nan = torch.isnan(new_embedding).any().item()

        if is_zero:
            print("    ⚠️  Warning: New embeddings are all zeros!")
        elif is_nan:
            print("    ⚠️  Warning: New embeddings contain NaN!")
        else:
            print(f"    ✅ New embeddings initialized properly (norm: {new_embedding.norm().item():.4f})")

    # ========================================
    # Step 6: Save model and tokenizer
    # ========================================
    print("\n[6/6] Saving model and tokenizer...")

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save tokenizer
    tokenizer.save_pretrained(output_path)
    print(f"  ✅ Saved tokenizer to {output_path}")

    # Save model
    model.save_pretrained(output_path)
    print(f"  ✅ Saved model to {output_path}")

    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*60)
    print("✅ SUCCESS!")
    print("="*60)
    print(f"Model with special tokens saved to: {output_path}")
    print()
    print("Special Token IDs:")
    for token, token_id in token_ids.items():
        print(f"  {token:15s} = {token_id}")
    print()
    print("Usage in your training config:")
    print(f"  actor_rollout_ref:")
    print(f"    model:")
    print(f"      path: {output_path}")
    print(f"      tokenizer_path: {output_path}")
    print("="*60)

    return model, tokenizer


def verify_model(model_path: str):
    """
    Verify that the saved model works correctly.
    """
    print("\n" + "="*60)
    print("Verification Test")
    print("="*60)

    # Load saved model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Loaded model from: {model_path}")
    print(f"Vocab size: {len(tokenizer)}")

    # Test generation with special tokens
    test_prompt = "What is 2+2? <think>"

    print(f"\nTest prompt: '{test_prompt}'")

    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    print(f"Input IDs: {inputs['input_ids'][0].tolist()}")

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print(f"Generated: {generated_text}")

    # Check if special tokens appear in generation
    has_think = '<think>' in generated_text
    has_end_think = '</think>' in generated_text

    if has_think or has_end_think:
        print("✅ Model can generate special tokens")
    else:
        print("ℹ️  Model did not generate special tokens (may need training)")

    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Add special tokens to Qwen2.5-1.5B-Instruct model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Source model path (default: Qwen/Qwen2.5-1.5B-Instruct)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./models/qwen2.5-1.5b-instruct-think",
        help="Output path for modified model (default: ./models/qwen2.5-1.5b-instruct-think)",
    )
    parser.add_argument(
        "--tokens",
        type=str,
        nargs="+",
        default=["<think>", "</think>"],
        help="Special tokens to add (default: <think> </think>)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run verification test after saving",
    )
    parser.add_argument(
        "--skip_save",
        action="store_true",
        help="Skip saving (dry run)",
    )

    args = parser.parse_args()

    # Convert dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # Add special tokens
    model, tokenizer = add_special_tokens_to_model(
        model_path=args.model_path,
        output_path=args.output_path,
        special_tokens=args.tokens,
        dtype=dtype,
    )

    # Verification test
    if args.verify:
        verify_model(args.output_path)

    print("\n✅ All done!")


if __name__ == "__main__":
    main()
