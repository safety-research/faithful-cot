# Copyright 2025 Bytedance Ltd. and/or its affiliates
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def initialize_system_prompt(tokenizer, **apply_chat_template_kwargs) -> list[int]:
    """
    Initialize system prompt tokens for chat templates that support them.

    Args:
        tokenizer: The tokenizer with a chat template
        **apply_chat_template_kwargs: Additional arguments for apply_chat_template

    Returns:
        List of token IDs for the system prompt, or empty list if not supported
    """
    token1 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True
    )

    # Try consecutive user messages first (simpler, works for most models)
    try:
        token2 = tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}] * 2, add_generation_prompt=False, tokenize=True
        )
    except Exception as e:
        # Some models (like Gemma-3) require strict role alternation (user/assistant/user/...)
        # Fall back to alternating roles
        logger.warning(f"Failed with consecutive user messages: {e}. Trying alternating roles.")
        token2 = tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}, {"role": "user", "content": ""}],
            add_generation_prompt=False,
            tokenize=True
        )

    # get system prompt tokens
    system_prompt = token1[: -(len(token2) - len(token1))]
    return system_prompt


def extract_system_prompt_and_generation(tokenizer):
    token1 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True
    )

    # Try consecutive user messages first (simpler, works for most models)
    try:
        token2 = tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}] * 2, add_generation_prompt=False, tokenize=True
        )
    except Exception as e:
        # Some models (like Gemma-3) require strict role alternation (user/assistant/user/...)
        # Fall back to alternating roles
        logger.warning(f"Failed with consecutive user messages: {e}. Trying alternating roles.")
        token2 = tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}, {"role": "user", "content": ""}],
            add_generation_prompt=False,
            tokenize=True
        )

    # get system prompt tokens
    system_prompt = token1[: -(len(token2) - len(token1))]
    # get generate prompt tokens
    token3 = tokenizer.apply_chat_template([{"role": "user", "content": ""}], add_generation_prompt=True, tokenize=True)
    generate_prompt = token3[len(token1) :]

    return system_prompt, generate_prompt
