#!/usr/bin/env python3
"""
Compute CoT faithfulness metrics from pre-generated outputs using clean implementation.

This script:
1. Loads questions from test parquet file
2. Loads generated outputs from hint_analysis.json
3. Computes faithfulness metrics using faithfulness_metrics_clean.py
4. Saves results to output file

Delimiter Detection:
- Default: Substring matching mode (works with any tokenizer)
- Optional: Special token mode (requires tokenizer with </think> special token)

Usage (Substring Matching - Default):
    python eval/compute_faithfulness_from_generations_clean.py \
        --test_file data/deepmindmath/test.parquet \
        --generations_file results/RL_deep_math_hint/checkpoint10/hint_analysis.json \
        --finetuned_model results/checkpoints/qwen2.5-1.5b-instruct/global_step_10_hf \
        --reference_model Qwen/Qwen2.5-1.5B-Instruct \
        --output_file results/RL_deep_math_hint/checkpoint10/faithfulness_metrics_clean.json \
        --num_samples 100 \
        --batch_size 8 \
        --device cuda

Usage (Special Token Mode):
    python eval/compute_faithfulness_from_generations_clean.py \
        --test_file data/deepmindmath/test.parquet \
        --generations_file results/RL_deep_math_hint/checkpoint10/hint_analysis.json \
        --finetuned_model results/checkpoints/qwen2.5-1.5b-instruct/global_step_10_hf \
        --reference_model Qwen/Qwen2.5-1.5B-Instruct \
        --output_file results/RL_deep_math_hint/checkpoint10/faithfulness_metrics_clean.json \
        --use-special-token \
        --end-think-token-id 151666
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from eval.faithfulness_metrics_clean_qwen import (
    create_tokenized_inputs,
    compute_all_metrics,
    FaithfulnessMetrics
)
from tqdm import tqdm


def load_test_data(test_file: str, num_samples: int = -1) -> list[str]:
    """Load questions from test parquet file."""
    print(f"Loading test data from {test_file}...")
    df = pd.read_parquet(test_file)

    if num_samples > 0:
        df = df.head(num_samples)

    # Extract questions - adjust based on your data structure
    questions = []
    for _, row in df.iterrows():
        if 'question' in row:
            questions.append(row['question'])
        elif 'messages' in row:
            # Messages can be a numpy array or list
            messages = row['messages']

            # Handle numpy array
            if hasattr(messages, 'tolist'):
                messages = messages.tolist()

            # If messages is a list of dicts with role/content
            if isinstance(messages, list) and len(messages) > 0:
                # Get the user message
                for msg in messages:
                    if isinstance(msg, dict) and msg.get('role') == 'user':
                        questions.append(msg.get('content', ''))
                        break
            elif isinstance(messages, str):
                questions.append(messages)
            else:
                questions.append(str(messages))
        else:
            # Try to find any text field
            questions.append(str(row.values[0]))

    print(f"Loaded {len(questions)} questions")
    return questions


def load_generations(generations_file: str, num_samples: int = -1) -> list[str]:
    """
    Load generated outputs from hint_analysis.json.

    Returns:
        generated_texts: List of generated outputs (format: "<think>...</think>\\nFinal answer: X")
    """
    print(f"Loading generations from {generations_file}...")
    with open(generations_file, 'r') as f:
        data = json.load(f)

    per_sample = data['per_sample_analyses']

    if num_samples > 0:
        per_sample = per_sample[:num_samples]

    generated_texts = [sample['generated_text'] for sample in per_sample]

    print(f"Loaded {len(generated_texts)} generated outputs")
    return generated_texts


def compute_faithfulness_metrics(
    questions: list[str],
    outputs: list[str],
    finetuned_model_path: str,
    reference_model_path: str,
    device: str = "cuda",
    batch_size: int = 8,
    compute_gradients: bool = True,
    use_substring_matching: bool = True,
    end_think_token_id: int | None = None,
    truncate_at_assert: bool = False,
    hint_prefix: str | None = None,
) -> dict:
    """
    Compute faithfulness metrics for the generated outputs.

    Uses optimized model-by-model processing:
    - Phase 1: Compute all metrics on generating model (with gradients if requested)
    - Phase 2: Compute only distributional metrics on reference model (KL, entropy, NLL - no gradients)

    This saves memory (only one model on GPU at a time) and is faster
    (uses hidden state caching).

    Args:
        questions: List of input questions
        outputs: List of generated outputs (with <think> tags)
        finetuned_model_path: Path to fine-tuned model checkpoint
        reference_model_path: Path to reference (base) model
        device: Device to run on
        batch_size: Batch size for processing
        compute_gradients: Whether to compute gradient metrics (set False to save ~2GB GPU memory)
        use_substring_matching: If True (default), use substring matching for </think> delimiter.
                                If False, use special token mode.
        end_think_token_id: Token ID for </think> (only used if use_substring_matching=False)

    Returns:
        Dictionary with all faithfulness metrics
    """
    # Print delimiter detection mode
    print(f"\n{'='*70}")
    print("DELIMITER DETECTION MODE")
    print(f"{'='*70}")
    if use_substring_matching:
        print("✓ Using substring matching mode (works with any tokenizer)")
        print("  Delimiter: '</think>' will be tokenized as multi-token sequence")
    else:
        print(f"✓ Using special token mode")
        print(f"  Token ID: {end_think_token_id}")
    print(f"{'='*70}")

    # ============== PHASE 1: Generating Model (Full Metrics) ==============
    print(f"\n{'='*70}")
    print("PHASE 1: GENERATING MODEL (all metrics)")
    print(f"{'='*70}")

    print(f"\nLoading fine-tuned model from {finetuned_model_path}...")
    finetuned_model = AutoModelForCausalLM.from_pretrained(
        finetuned_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )

    print(f"Loading fine-tuned tokenizer...")
    finetuned_tokenizer = AutoTokenizer.from_pretrained(
        finetuned_model_path,
        trust_remote_code=True
    )
    if finetuned_tokenizer.pad_token is None:
        finetuned_tokenizer.pad_token = finetuned_tokenizer.eos_token

    # Handle special tokens based on mode
    if not use_substring_matching:
        # Special token mode: ensure special tokens are present
        if "<think>" not in finetuned_tokenizer.get_vocab():
            finetuned_tokenizer.add_tokens(["<think>", "</think>"], special_tokens=True)
            finetuned_model.resize_token_embeddings(len(finetuned_tokenizer))
            print(f"[Special Token Mode] Added <think> and </think> tokens to fine-tuned model")
        else:
            print(f"[Special Token Mode] Using existing <think> and </think> special tokens")
    else:
        # Substring matching mode: do NOT add special tokens
        if "<think>" in finetuned_tokenizer.get_vocab():
            print(f"[Substring Mode] Model has <think> special tokens, but using substring matching anyway")
        else:
            print(f"[Substring Mode] Model does not have special tokens (as expected for substring matching)")

    # Process all batches on generating model
    all_metrics_gen = []
    print(f"\nProcessing {len(questions)} samples in batches of {batch_size}...")

    # Debug: print first sample
    if len(questions) > 0:
        print(f"\n=== First sample ===")
        print(f"Question: {questions[0][:200]}...")
        print(f"Output: {outputs[0][:200]}...")
        print(f"Has </think>: {'</think>' in outputs[0]}")
        print("=" * 50)

    for i in tqdm(range(0, len(questions), batch_size), desc="Generating model"):
        batch_questions = questions[i:i+batch_size]
        batch_outputs = outputs[i:i+batch_size]

        try:
            # Create tokenized inputs for fine-tuned model
            inputs_gen = create_tokenized_inputs(
                finetuned_tokenizer,
                batch_questions,
                batch_outputs,
                device=device,
                use_substring_matching=use_substring_matching,
                end_think_token_id=end_think_token_id,
                truncate_at_assert=truncate_at_assert,
                hint_prefix=hint_prefix,
            )

            # Compute ALL metrics on fine-tuned model
            metrics_gen = compute_all_metrics(
                finetuned_model, inputs_gen, device=device, compute_gradients=compute_gradients
            )

            # Convert to dict for storage
            batch_metrics_gen = {
                'kl_direct_effect': metrics_gen.kl_direct_effect.tolist(),
                'kl_cot_necessity': metrics_gen.kl_cot_necessity.tolist(),
                'kl_leakage': metrics_gen.kl_leakage.tolist(),
                'js_direct_effect': metrics_gen.js_direct_effect.tolist(),
                'js_cot_necessity': metrics_gen.js_cot_necessity.tolist(),
                'grad_de_l1': metrics_gen.grad_de_l1.tolist(),
                'grad_de_l2': metrics_gen.grad_de_l2.tolist(),
                'grad_cot_necessity_l1': metrics_gen.grad_cot_necessity_l1.tolist(),
                'grad_cot_necessity_l2': metrics_gen.grad_cot_necessity_l2.tolist(),
                'grad_leakage_l1': metrics_gen.grad_leakage_l1.tolist(),
                'grad_leakage_l2': metrics_gen.grad_leakage_l2.tolist(),
                'entropy_full': metrics_gen.entropy_full.tolist(),
                'entropy_via_cot': metrics_gen.entropy_via_cot.tolist(),
                'entropy_no_prompt': metrics_gen.entropy_no_prompt.tolist(),
                'nll_full': metrics_gen.nll_full.tolist(),
                'nll_via_cot': metrics_gen.nll_via_cot.tolist(),
                'nll_no_prompt': metrics_gen.nll_no_prompt.tolist(),
            }

            all_metrics_gen.append(batch_metrics_gen)

            del inputs_gen
            del metrics_gen
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            import traceback
            traceback.print_exc()
            torch.cuda.empty_cache()
            continue

    # Free generating model
    print("\n✓ Generating model done. Freeing GPU memory...")
    del finetuned_model
    del finetuned_tokenizer
    torch.cuda.empty_cache()

    # ============== PHASE 2: Reference Model ==============
    print(f"\n{'='*70}")
    print("PHASE 2: REFERENCE MODEL (distributional metrics only - no gradients)")
    print(f"{'='*70}")

    print(f"\nLoading reference model from {reference_model_path}...")
    reference_model = AutoModelForCausalLM.from_pretrained(
        reference_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )

    print(f"Loading reference tokenizer...")
    reference_tokenizer = AutoTokenizer.from_pretrained(
        reference_model_path,
        trust_remote_code=True
    )
    if reference_tokenizer.pad_token is None:
        reference_tokenizer.pad_token = reference_tokenizer.eos_token

    # Handle special tokens based on mode (same as finetuned model)
    if not use_substring_matching:
        # Special token mode: ensure special tokens are present
        if "<think>" not in reference_tokenizer.get_vocab():
            reference_tokenizer.add_tokens(["<think>", "</think>"], special_tokens=True)
            reference_model.resize_token_embeddings(len(reference_tokenizer))
            print(f"[Special Token Mode] Added <think> and </think> tokens to reference model")
        else:
            print(f"[Special Token Mode] Using existing <think> and </think> special tokens")
    else:
        # Substring matching mode: do NOT add special tokens
        if "<think>" in reference_tokenizer.get_vocab():
            print(f"[Substring Mode] Reference model has <think> special tokens, but using substring matching anyway")
        else:
            print(f"[Substring Mode] Reference model does not have special tokens (as expected for substring matching)")

    # Process all batches on reference model (lightweight - only S/C/N)
    all_metrics_ref = []
    print(f"\nProcessing {len(questions)} samples in batches of {batch_size}...")

    for i in tqdm(range(0, len(questions), batch_size), desc="Reference model"):
        batch_questions = questions[i:i+batch_size]
        batch_outputs = outputs[i:i+batch_size]

        try:
            # Create tokenized inputs for reference model
            inputs_ref = create_tokenized_inputs(
                reference_tokenizer,
                batch_questions,
                batch_outputs,
                device=device,
                use_substring_matching=use_substring_matching,
                end_think_token_id=end_think_token_id,
                truncate_at_assert=truncate_at_assert,
                hint_prefix=hint_prefix,
            )

            # Compute metrics on reference model (no gradients - only distributional metrics)
            metrics_ref = compute_all_metrics(
                reference_model, inputs_ref, device=device, compute_gradients=False
            )

            # Convert to dict for storage
            # Note: gradient metrics will be NaN (not computed for reference model)
            batch_metrics_ref = {
                'kl_direct_effect': metrics_ref.kl_direct_effect.tolist(),
                'kl_cot_necessity': metrics_ref.kl_cot_necessity.tolist(),
                'kl_leakage': metrics_ref.kl_leakage.tolist(),
                'js_direct_effect': metrics_ref.js_direct_effect.tolist(),
                'js_cot_necessity': metrics_ref.js_cot_necessity.tolist(),
                'entropy_full': metrics_ref.entropy_full.tolist(),
                'entropy_via_cot': metrics_ref.entropy_via_cot.tolist(),
                'entropy_no_prompt': metrics_ref.entropy_no_prompt.tolist(),
                'nll_full': metrics_ref.nll_full.tolist(),
                'nll_via_cot': metrics_ref.nll_via_cot.tolist(),
                'nll_no_prompt': metrics_ref.nll_no_prompt.tolist(),
            }

            all_metrics_ref.append(batch_metrics_ref)

            del inputs_ref
            del metrics_ref
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            import traceback
            traceback.print_exc()
            torch.cuda.empty_cache()
            continue

    # Free reference model
    print("\n✓ Reference model done. Freeing GPU memory...")
    del reference_model
    del reference_tokenizer
    torch.cuda.empty_cache()

    # Check if we have any successful batches
    if len(all_metrics_gen) == 0:
        raise RuntimeError("No batches were successfully processed. Check errors above.")

    # Aggregate all batches
    print("\nAggregating results...")

    def aggregate_batches(all_metrics):
        aggregated = {}
        for key in all_metrics[0].keys():
            values = []
            for batch in all_metrics:
                values.extend(batch[key])
            aggregated[key] = values
        return aggregated

    aggregated_gen = aggregate_batches(all_metrics_gen)
    aggregated_ref = aggregate_batches(all_metrics_ref)

    # Compute statistics
    print("\nComputing summary statistics...")

    def compute_statistics(aggregated):
        summary = {}
        for key, values in aggregated.items():
            values_tensor = torch.tensor(values)
            # Filter out NaNs for statistics
            valid_values = values_tensor[~torch.isnan(values_tensor)]
            if len(valid_values) > 0:
                summary[key] = {
                    'mean': float(valid_values.mean()),
                    'std': float(valid_values.std()),
                    'min': float(valid_values.min()),
                    'max': float(valid_values.max()),
                    'median': float(valid_values.median()),
                    'num_valid': int(len(valid_values)),
                    'num_nan': int(len(values_tensor) - len(valid_values)),
                }
            else:
                summary[key] = {
                    'mean': float('nan'),
                    'std': float('nan'),
                    'min': float('nan'),
                    'max': float('nan'),
                    'median': float('nan'),
                    'num_valid': 0,
                    'num_nan': int(len(values_tensor)),
                }
        return summary

    summary_gen = compute_statistics(aggregated_gen)
    summary_ref = compute_statistics(aggregated_ref)

    return {
        'generating_model': {
            'per_sample_metrics': aggregated_gen,
            'summary_statistics': summary_gen,
        },
        'reference_model': {
            'per_sample_metrics': aggregated_ref,
            'summary_statistics': summary_ref,
        },
        'num_samples': len(questions),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute faithfulness metrics from generated outputs")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test parquet file")
    parser.add_argument("--generations_file", type=str, required=True, help="Path to hint_analysis.json")
    parser.add_argument("--finetuned_model", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--reference_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                       help="Path to reference model")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save faithfulness metrics")
    parser.add_argument("--num_samples", type=int, default=-1,
                       help="Number of samples to process (-1 for all)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing (1 avoids left-padding during metric computation)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("--no-gradients", action="store_true",
                       help="Skip gradient metrics to save GPU memory (~2GB)")

    # Delimiter detection mode
    parser.add_argument("--use-substring-matching", action="store_true", default=True,
                       help="Use substring matching for </think> delimiter (default, works with any tokenizer)")
    parser.add_argument("--use-special-token", action="store_true",
                       help="Use special token mode instead of substring matching")
    parser.add_argument("--end-think-token-id", type=int, default=None,
                       help="Token ID for </think> (only used with --use-special-token)")
    parser.add_argument("--truncate-at-assert", action="store_true",
                       help="Exclude assertion block from answer_mask; measure only solve() body")
    parser.add_argument("--hint-prefix", type=str, default=None,
                       help="If set, narrow prompt_mask to start from this substring (e.g. 'A Stanford professor guessed'). "
                            "Gradient/KL metrics then measure only the hint-specific portion of the prompt.")

    args = parser.parse_args()

    # Handle delimiter mode flags
    if args.use_special_token:
        use_substring_matching = False
        if args.end_think_token_id is None:
            parser.error("--end-think-token-id is required when using --use-special-token")
    else:
        use_substring_matching = True
        end_think_token_id = args.end_think_token_id  # Can be None for substring mode

    # Create output directory
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    questions = load_test_data(args.test_file, args.num_samples)
    generated_texts = load_generations(args.generations_file, args.num_samples)

    # Ensure lengths match
    min_len = min(len(questions), len(generated_texts))
    if len(questions) != len(generated_texts):
        print(f"Warning: Question count ({len(questions)}) != generation count ({len(generated_texts)})")
        print(f"Using first {min_len} samples")
        questions = questions[:min_len]
        generated_texts = generated_texts[:min_len]

    # Compute metrics
    results = compute_faithfulness_metrics(
        questions=questions,
        outputs=generated_texts,
        finetuned_model_path=args.finetuned_model,
        reference_model_path=args.reference_model,
        device=args.device,
        batch_size=args.batch_size,
        compute_gradients=not args.no_gradients,
        use_substring_matching=use_substring_matching,
        end_think_token_id=end_think_token_id,
        truncate_at_assert=args.truncate_at_assert,
        hint_prefix=args.hint_prefix,
    )

    # Add metadata
    results['config'] = {
        'test_file': args.test_file,
        'generations_file': args.generations_file,
        'finetuned_model': args.finetuned_model,
        'reference_model': args.reference_model,
        'num_samples': args.num_samples,
        'batch_size': args.batch_size,
        'device': args.device,
        'use_substring_matching': use_substring_matching,
        'end_think_token_id': end_think_token_id,
        'delimiter_mode': 'substring' if use_substring_matching else 'special_token',
        'hint_prefix': args.hint_prefix,
    }

    # Save results
    print(f"\nSaving results to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Faithfulness metrics computed successfully!")
    print(f"Results saved to: {args.output_file}")

    # Print summary
    print("\n=== Summary Statistics (Generating Model) ===")
    for metric_name, stats in results['generating_model']['summary_statistics'].items():
        print(f"\n{metric_name}:")
        print(f"  Mean:   {stats['mean']:.4f}")
        print(f"  Std:    {stats['std']:.4f}")
        print(f"  Min:    {stats['min']:.4f}")
        print(f"  Max:    {stats['max']:.4f}")
        print(f"  Valid:  {stats['num_valid']}/{stats['num_valid'] + stats['num_nan']}")

    print("\n=== Summary Statistics (Reference Model) ===")
    for metric_name, stats in results['reference_model']['summary_statistics'].items():
        print(f"\n{metric_name}:")
        print(f"  Mean:   {stats['mean']:.4f}")
        print(f"  Std:    {stats['std']:.4f}")
        print(f"  Min:    {stats['min']:.4f}")
        print(f"  Max:    {stats['max']:.4f}")
        print(f"  Valid:  {stats['num_valid']}/{stats['num_valid'] + stats['num_nan']}")


if __name__ == "__main__":
    main()
