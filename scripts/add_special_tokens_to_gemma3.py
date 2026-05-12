import torch
import json
import re
from transformers import AutoTokenizer, Gemma3ForCausalLM

def prepare_gemma3_1b_with_think_tokens(
    model_name: str = "google/gemma-3-1b-it",
    output_path: str = "/workspace-vast/jinghanj/workspace/Structural_RL/models/checkpoints/gemma3-1b-think",
    new_tokens: list = ["<think>", "</think>"],
):
    """
    Load Gemma 3 1B and add custom tokens for reasoning.
    """
    print(f"Loading model and tokenizer from {model_name}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add custom tokens
    print(f"Adding custom tokens: {new_tokens}")
    num_added = tokenizer.add_tokens(new_tokens, special_tokens=False)
    print(f"Added {num_added} new tokens. New vocab size: {len(tokenizer)}")
    
    # Load model
    model = Gemma3ForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Resize embeddings to accommodate new tokens
    print("Resizing token embeddings...")
    model.resize_token_embeddings(len(tokenizer))
    
    # Save the model and tokenizer
    print(f"Saving to {output_path}...")
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    
    # Print token IDs for reference
    for token in new_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"Token '{token}' -> ID {token_id}")
    
    print("Done!")


def test_generation(
    model_path: str,
    test_prompts: list = None,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
):
    """
    Test generation with the model to verify it works correctly.

    Args:
        model_path: Path to the model checkpoint
        test_prompts: List of test problems (default: simple math problems)
        temperature: Sampling temperature
        max_new_tokens: Maximum tokens to generate
    """
    if test_prompts is None:
        test_prompts = [
            "What is 15 + 27?",
            "Calculate 8 × 9.",
            "If a rectangle has length 12 and width 5, what is its area?",
            "Solve for x: 2x + 5 = 15",
        ]

    print(f"\nLoading model for generation test from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = Gemma3ForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    print("\n" + "="*80)
    print("GENERATION TESTS")
    print("="*80)

    # Check if think tokens are present
    has_think = "<think>" in tokenizer.get_vocab()
    has_end_think = "</think>" in tokenizer.get_vocab()
    print(f"\nThink tokens present: <think>={has_think}, </think>={has_end_think}")

    if has_think:
        think_id = tokenizer.convert_tokens_to_ids("<think>")
        print(f"  <think> token ID: {think_id}")
    if has_end_think:
        end_think_id = tokenizer.convert_tokens_to_ids("</think>")
        print(f"  </think> token ID: {end_think_id}")

    # Test each prompt
    results = []
    for idx, problem in enumerate(test_prompts, 1):
        print(f"\n{'='*80}")
        print(f"Test {idx}/{len(test_prompts)}: {problem}")
        print("="*80)
        new_prompt = """Solve the following math problem step by step.

Format your response as follows:
1. Start with <think> tag
2. Show your step-by-step reasoning inside the <think> tags
3. Close with </think> tag
4. Provide your final numerical answer after "Final answer: "

Example format:
<think>
Your step-by-step reasoning here
</think>
Final answer: your numerical answer

Problem: {problem}""" 
        # Format as chat
        messages = [
            {"role": "system", "content": "You are a helpful math tutor. Use <think> tags to show your reasoning, then provide the final answer."},
            {"role": "user", "content": new_prompt},
        ]

        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Generate
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode
        generated_text = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        print("\nGenerated response:")
        print("-" * 80)
        print(generated_text)
        print("-" * 80)

        # Check format
        has_open_think = "<think>" in generated_text
        has_close_think = "</think>" in generated_text
        has_answer = bool(re.search(r"final\s+answer\s*:", generated_text, re.IGNORECASE))

        print(f"\nFormat check:")
        print(f"  Has <think>: {has_open_think}")
        print(f"  Has </think>: {has_close_think}")
        print(f"  Has 'Final answer:': {has_answer}")

        # Extract answer
        answer = ""
        if has_answer:
            match = re.search(r"final\s+answer\s*:\s*(.+?)(?:\n|$)", generated_text, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                print(f"  Extracted answer: {answer}")

        results.append({
            "problem": problem,
            "generated_text": generated_text,
            "has_think_tags": has_open_think and has_close_think,
            "has_answer": has_answer,
            "extracted_answer": answer,
        })

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    total = len(results)
    with_think_tags = sum(1 for r in results if r["has_think_tags"])
    with_answer = sum(1 for r in results if r["has_answer"])

    print(f"Total tests: {total}")
    print(f"With think tags: {with_think_tags}/{total} ({100*with_think_tags/total:.1f}%)")
    print(f"With answer format: {with_answer}/{total} ({100*with_answer/total:.1f}%)")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["prepare", "test", "both"], default="prepare",
                        help="Mode: prepare (add tokens), test (generation test), or both")
    parser.add_argument("--model", default="google/gemma-3-1b-it",
                        help="Model name or path")
    parser.add_argument("--output", default="./gemma3-1b-think",
                        help="Output path for prepared model")
    args = parser.parse_args()

    if args.mode in ["prepare", "both"]:
        print("="*80)
        print("PREPARING MODEL WITH THINK TOKENS")
        print("="*80)
        prepare_gemma3_1b_with_think_tokens(
            model_name=args.model,
            output_path=args.output,
            new_tokens=["<think>", "</think>"],
        )

    if args.mode in ["test", "both"]:
        # If we just prepared, test the output model
        # Otherwise test the input model
        test_path = args.output if args.mode == "both" else args.model

        print("\n" + "="*80)
        print("TESTING MODEL GENERATION")
        print("="*80)
        test_generation(model_path=test_path)