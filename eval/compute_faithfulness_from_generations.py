#!/usr/bin/env python3
"""
Compute CoT faithfulness metrics from pre-generated outputs.

This script:
1. Loads questions from test parquet file
2. Loads generated outputs from hint_analysis.json
3. Computes faithfulness metrics using faithfulness_metrics_pytorch.py
4. Saves results to output file

Usage:
    python compute_faithfulness_from_generations.py \
        --test_file data/deepmindmath/test.parquet \
        --generations_file results/RL_deep_math_hint/checkpoint10/hint_analysis.json \
        --finetuned_model results/checkpoints/qwen2.5-1.5b-instruct/global_step_10_hf \
        --reference_model Qwen/Qwen2.5-1.5B-Instruct \
        --output_file results/RL_deep_math_hint/checkpoint10/faithfulness_metrics.json \
        --num_samples 100
"""

import argparse
import json
import torch
import pandas as pd
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from faithfulness_metrics_pytorch import FaithfulnessEvaluator
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


def load_generations(generations_file: str, num_samples: int = -1) -> tuple[list[str], list[str]]:
    """
    Load generated outputs from hint_analysis.json.

    Returns:
        (generated_texts, cot_texts): Lists of generated outputs and extracted CoTs
    """
    print(f"Loading generations from {generations_file}...")
    with open(generations_file, 'r') as f:
        data = json.load(f)

    per_sample = data['per_sample_analyses']

    if num_samples > 0:
        per_sample = per_sample[:num_samples]

    generated_texts = [sample['generated_text'] for sample in per_sample]
    cot_texts = [sample['cot_text'] for sample in per_sample]

    print(f"Loaded {len(generated_texts)} generated outputs")
    return generated_texts, cot_texts


def compute_faithfulness_metrics(
    questions: list[str],
    outputs: list[str],
    finetuned_model_path: str,
    reference_model_path: str,
    device: str = "cuda",
    batch_size: int = 8,
) -> dict:
    """
    Compute faithfulness metrics for the generated outputs.

    Args:
        questions: List of input questions
        outputs: List of generated outputs (with <think> tags)
        finetuned_model_path: Path to fine-tuned model checkpoint
        reference_model_path: Path to reference (base) model
        device: Device to run on
        batch_size: Batch size for processing

    Returns:
        Dictionary with all faithfulness metrics
    """
    print(f"\nLoading fine-tuned model from {finetuned_model_path}...")
    finetuned_model = AutoModelForCausalLM.from_pretrained(
        finetuned_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",  # Avoid SDPA bitwise operation bug in transformers 4.57.6
    )

    print(f"Loading reference model from {reference_model_path}...")
    reference_model = AutoModelForCausalLM.from_pretrained(
        reference_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",  # Avoid SDPA bitwise operation bug in transformers 4.57.6
    )

    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\nInitializing FaithfulnessEvaluator...")
    evaluator = FaithfulnessEvaluator(
        model=finetuned_model,
        tokenizer=tokenizer,
        reference_model=reference_model,
        device=device,
    )

    # Process in batches to avoid OOM
    all_metrics = []

    print(f"\nComputing faithfulness metrics for {len(questions)} samples...")
    print(f"Processing in batches of {batch_size}...")

    # Debug: print first sample
    if len(questions) > 0:
        print(f"\n=== DEBUG: First sample ===")
        print(f"Question: {questions[0][:200]}...")
        print(f"Output: {outputs[0][:200]}...")
        print("=" * 50)

    for i in tqdm(range(0, len(questions), batch_size), desc="Processing batches"):
        batch_questions = questions[i:i+batch_size]
        batch_outputs = outputs[i:i+batch_size]

        print(f"\nProcessing batch {i//batch_size}, samples {i} to {i+len(batch_questions)}")

        try:
            metrics = evaluator.compute_all_metrics(batch_questions, batch_outputs)

            # Convert tensors to lists for JSON serialization
            batch_metrics = {
                'kl_direct_effect': metrics.kl_direct_effect.tolist(),
                'kl_cot_necessity': metrics.kl_cot_necessity.tolist(),
                'kl_leakage': metrics.kl_leakage.tolist(),
                'grad_de_l1': metrics.grad_de_l1.tolist(),
                'grad_de_l2': metrics.grad_de_l2.tolist(),
                'grad_cot_necessity_l1': metrics.grad_cot_necessity_l1.tolist(),
                'grad_cot_necessity_l2': metrics.grad_cot_necessity_l2.tolist(),
                'grad_leakage_l1': metrics.grad_leakage_l1.tolist(),
                'grad_leakage_l2': metrics.grad_leakage_l2.tolist(),
                'entropy_full': metrics.entropy_full.tolist(),
                'entropy_via_cot': metrics.entropy_via_cot.tolist(),
                'entropy_no_prompt': metrics.entropy_no_prompt.tolist(),
                'nll_full': metrics.nll_full.tolist(),
                'nll_via_cot': metrics.nll_via_cot.tolist(),
                'nll_no_prompt': metrics.nll_no_prompt.tolist(),
            }

            # Add sufficiency metrics if available
            if metrics.sufficiency_h_a_given_c is not None:
                batch_metrics['sufficiency_h_a_given_c'] = metrics.sufficiency_h_a_given_c.tolist()
                batch_metrics['sufficiency_h_a_given_p'] = metrics.sufficiency_h_a_given_p.tolist()
                batch_metrics['sufficiency_reduction'] = metrics.sufficiency_reduction.tolist()

            # Add completeness and necessity metrics if available
            if metrics.completeness_generating_model is not None:
                batch_metrics['completeness_generating_model'] = metrics.completeness_generating_model.tolist()
                batch_metrics['necessity_generating_model'] = metrics.necessity_generating_model.tolist()

            if metrics.completeness_reference_model is not None:
                batch_metrics['completeness_reference_model'] = metrics.completeness_reference_model.tolist()
                batch_metrics['necessity_reference_model'] = metrics.necessity_reference_model.tolist()

            all_metrics.append(batch_metrics)

        except Exception as e:
            print(f"Error processing batch {i//batch_size}: {e}")
            import traceback
            traceback.print_exc()
            continue
    print(f"DEBUG: Collected {len(all_metrics)} batches")

    # Check if we have any successful batches
    if len(all_metrics) == 0:
        raise RuntimeError("No batches were successfully processed. Check errors above.")

    # Aggregate all batches
    print("\nAggregating results...")
    aggregated = {}

    for key in all_metrics[0].keys():
        values = []
        for batch in all_metrics:
            values.extend(batch[key])
        aggregated[key] = values

    # Compute statistics
    print("\nComputing summary statistics...")
    summary = {}
    for key, values in aggregated.items():
        values_tensor = torch.tensor(values)
        summary[key] = {
            'mean': float(values_tensor.mean()),
            'std': float(values_tensor.std()),
            'min': float(values_tensor.min()),
            'max': float(values_tensor.max()),
            'median': float(values_tensor.median()),
        }

    return {
        'per_sample_metrics': aggregated,
        'summary_statistics': summary,
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
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    questions = load_test_data(args.test_file, args.num_samples)
    generated_texts, cot_texts = load_generations(args.generations_file, args.num_samples)

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
    }

    # Save results
    print(f"\nSaving results to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Faithfulness metrics computed successfully!")
    print(f"Results saved to: {args.output_file}")

    # Print summary
    print("\n=== Summary Statistics ===")
    for metric_name, stats in results['summary_statistics'].items():
        print(f"\n{metric_name}:")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std:  {stats['std']:.4f}")
        print(f"  Min:  {stats['min']:.4f}")
        print(f"  Max:  {stats['max']:.4f}")


if __name__ == "__main__":
    main()
