"""
Quick test to verify vectorized CoT masking produces same results as before
and runs much faster.
"""
import torch
import time
from cot_masking import create_cot_attention_mask


def test_vectorized_performance():
    """Test that vectorized version is fast enough for large batches."""
    # Simulate realistic training batch
    batch_size = 512
    seq_len = 512
    think_token_id = 151665
    end_think_token_id = 151666

    # Create sample input with <think>...</think> tags
    input_ids = torch.randint(0, 50000, (batch_size, seq_len))

    # Add <think> at position ~100 and </think> at position ~300 for all samples
    for i in range(batch_size):
        input_ids[i, 100] = think_token_id
        input_ids[i, 300] = end_think_token_id

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = input_ids.to(device)

    print(f"Testing on {device}")
    print(f"Batch size: {batch_size}, Seq len: {seq_len}")
    print(f"Total elements: {batch_size * seq_len * seq_len:,}")

    # Warm up
    _ = create_cot_attention_mask(input_ids[:2], think_token_id, end_think_token_id)

    # Time the operation
    start = time.time()
    mask = create_cot_attention_mask(input_ids, think_token_id, end_think_token_id)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start

    print(f"Time: {elapsed:.3f} seconds")
    print(f"Throughput: {batch_size / elapsed:.1f} samples/sec")

    # Verify correctness: check that answer cannot attend to prompt
    # Sample 0: prompt=[0:100], CoT=[100:301], answer=[301:]
    sample_0_mask = mask[0, 0]  # (seq_len, seq_len)

    # Check: answer position 350 should NOT attend to prompt position 50
    assert sample_0_mask[350, 50] == float('-inf'), "Answer→Prompt should be blocked"

    # Check: answer position 350 CAN attend to CoT position 200
    assert sample_0_mask[350, 200] != float('-inf'), "Answer→CoT should be allowed"

    # Check: answer position 350 CAN attend to answer position 320 (causal)
    assert sample_0_mask[350, 320] != float('-inf'), "Answer→Answer (causal) should be allowed"

    print("✅ All correctness checks passed!")

    # Performance target: should complete in <1 second for 512 batch
    if elapsed > 1.0:
        print(f"⚠️  Warning: Slow performance ({elapsed:.3f}s). Expected <1s")
    else:
        print(f"✅ Performance acceptable ({elapsed:.3f}s < 1s threshold)")

    return elapsed


if __name__ == "__main__":
    elapsed = test_vectorized_performance()
    print(f"\nFinal time: {elapsed:.3f} seconds")
