"""
Test and visualize CoT attention masking with a concrete example.
"""
import torch
import sys
sys.path.insert(0, '/workspace-vast/jinghanj/workspace/Structural_RL/train')

from verl.utils.cot_masking import create_cot_attention_mask, find_special_token_positions


def visualize_attention_mask():
    """Create and visualize a simple CoT attention mask example."""

    # Setup
    think_token_id = 151665
    end_think_token_id = 151666

    # Create a simple example sequence
    # Structure: [PROMPT tokens] <think> [COT tokens] </think> [ANSWER tokens]
    seq_len = 20
    input_ids = torch.zeros((1, seq_len), dtype=torch.long)

    # Positions:
    # 0-4: Prompt (5 tokens)
    # 5: <think>
    # 6-9: CoT reasoning (4 tokens)
    # 10: </think>
    # 11-19: Answer (9 tokens)

    input_ids[0, 0:5] = torch.tensor([100, 101, 102, 103, 104])  # Prompt
    input_ids[0, 5] = think_token_id  # <think>
    input_ids[0, 6:10] = torch.tensor([200, 201, 202, 203])  # CoT
    input_ids[0, 10] = end_think_token_id  # </think>
    input_ids[0, 11:20] = torch.tensor([300, 301, 302, 303, 304, 305, 306, 307, 308])  # Answer

    print("=" * 80)
    print("CoT Attention Mask Visualization")
    print("=" * 80)
    print()

    # Print the sequence structure
    print("Input Sequence Structure:")
    print("-" * 80)
    for i in range(seq_len):
        token_id = input_ids[0, i].item()
        if token_id == think_token_id:
            region = "THINK_START"
            symbol = "<think>"
        elif token_id == end_think_token_id:
            region = "THINK_END"
            symbol = "</think>"
        elif i < 5:
            region = "PROMPT"
            symbol = f"tok_{token_id}"
        elif i < 10:
            region = "COT"
            symbol = f"tok_{token_id}"
        else:
            region = "ANSWER"
            symbol = f"tok_{token_id}"

        print(f"  Position {i:2d}: {symbol:15s} [{region}]")
    print()

    # Find regions
    prompt_mask, cot_mask, answer_mask = find_special_token_positions(
        input_ids, think_token_id, end_think_token_id
    )

    print("Region Masks:")
    print("-" * 80)
    print(f"Prompt positions:  {prompt_mask[0].nonzero(as_tuple=True)[0].tolist()}")
    print(f"CoT positions:     {cot_mask[0].nonzero(as_tuple=True)[0].tolist()}")
    print(f"Answer positions:  {answer_mask[0].nonzero(as_tuple=True)[0].tolist()}")
    print()

    # Create attention mask
    attention_mask_4d = create_cot_attention_mask(
        input_ids, think_token_id, end_think_token_id, dtype=torch.float32
    )

    # Extract 2D mask for visualization (batch=0, head=0)
    mask_2d = attention_mask_4d[0, 0].cpu()  # (seq_len, seq_len)

    # Print attention mask as a grid
    print("Attention Mask (Query → Key):")
    print("-" * 80)
    print("Legend: ✓ = can attend (0.0), ✗ = blocked (-inf), △ = causal block")
    print()

    # Header
    print("     Key→ ", end="")
    for j in range(seq_len):
        if j < 5:
            print("P", end="")
        elif j == 5:
            print("<", end="")
        elif j < 10:
            print("C", end="")
        elif j == 10:
            print(">", end="")
        else:
            print("A", end="")
    print(" (P=Prompt, C=CoT, A=Answer)")
    print("     " + "-" * (seq_len + 7))

    # Rows
    for i in range(seq_len):
        # Row label
        if i < 5:
            row_label = f"P{i}"
        elif i == 5:
            row_label = "<t"
        elif i < 10:
            row_label = f"C{i-6}"
        elif i == 10:
            row_label = ">t"
        else:
            row_label = f"A{i-11}"

        print(f"Query {row_label:2s} |", end="")

        # Columns
        for j in range(seq_len):
            val = mask_2d[i, j].item()
            if val == 0.0:
                print("✓", end="")
            elif val == float('-inf'):
                if j > i:  # Causal blocking (future)
                    print("△", end="")
                else:  # CoT structural blocking (answer→prompt)
                    print("✗", end="")
            else:
                print("?", end="")
        print()
    print()

    # Test specific attention patterns
    print("Attention Rules Verification:")
    print("-" * 80)

    # Test cases
    test_cases = [
        # (query_pos, key_pos, expected_result, description)
        (2, 1, True, "Prompt[2] → Prompt[1] (causal past)"),
        (2, 3, False, "Prompt[2] → Prompt[3] (causal future)"),
        (7, 2, True, "CoT[7] → Prompt[2] (should attend)"),
        (7, 8, False, "CoT[7] → CoT[8] (causal future)"),
        (7, 6, True, "CoT[7] → CoT[6] (causal past)"),
        (15, 2, False, "Answer[15] → Prompt[2] (BLOCKED by CoT mask)"),
        (15, 7, True, "Answer[15] → CoT[7] (should attend)"),
        (15, 12, True, "Answer[15] → Answer[12] (causal past)"),
        (15, 16, False, "Answer[15] → Answer[16] (causal future)"),
    ]

    all_passed = True
    for query_pos, key_pos, should_attend, description in test_cases:
        val = mask_2d[query_pos, key_pos].item()
        can_attend = (val == 0.0)

        if can_attend == should_attend:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
            all_passed = False

        attend_str = "CAN attend" if can_attend else "BLOCKED"
        expected_str = "should attend" if should_attend else "should block"
        print(f"{status}: {description}")
        print(f"       Result: {attend_str}, Expected: {expected_str}")
        print()

    if all_passed:
        print("=" * 80)
        print("🎉 ALL TESTS PASSED! CoT masking is working correctly.")
        print("=" * 80)
    else:
        print("=" * 80)
        print("❌ SOME TESTS FAILED! Please review the mask logic.")
        print("=" * 80)

    print()
    print("Summary:")
    print("-" * 80)
    print("✓ Prompt tokens can attend to: previous prompt (causal)")
    print("✓ CoT tokens can attend to: all prompt + previous CoT (causal)")
    print("✓ Answer tokens can attend to: all CoT + previous answer (causal)")
    print("✗ Answer tokens CANNOT attend to: any prompt (CoT masking)")
    print()

    return all_passed


if __name__ == "__main__":
    success = visualize_attention_mask()
    sys.exit(0 if success else 1)
