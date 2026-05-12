# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.models.transformers.cot_parameter_gradient_masking_patch import cot_parameter_gradient_mask_context
from verl.utils.cot_masking import find_special_token_positions
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.utils.cot_masking import create_cot_attention_mask
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None, tokenizer=None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.tokenizer = tokenizer  # Store tokenizer for substring delimiter tokenization
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.use_dynamic_bsz = self.config.get("use_dynamic_bsz", False)

        self.use_prefix_grouper = self.config.get("use_prefix_grouper", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_prefix_grouper={self.use_prefix_grouper}")

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

        # Sum of squared probabilities computation (for optimal_token_baseline)
        # Only initialize if calculate_sum_pi_squared config is enabled
        if self.config.get("calculate_sum_pi_squared", False):
            self.calculate_sum_pi_squared_from_logits = (
                torch.compile(verl_F.calculate_sum_pi_squared_from_logits, dynamic=True)
                if self.config.get("use_torch_compile", True)
                else verl_F.calculate_sum_pi_squared_from_logits
            )
            assert not (self.use_fused_kernels or self.use_prefix_grouper), (
                "calculate_sum_pi_squared is not supported with "
                f"{self.use_fused_kernels=} or {self.use_prefix_grouper=} for now."
            )

    def _forward_micro_batch(
        self, micro_batch: dict[str, torch.Tensor], temperature: float, calculate_entropy: bool = False
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            dict[str, torch.Tensor]:
                log_probs: (bs, response_len)
                if calculate_entropy is True:
                    entropys: (bs, response_len)
                if calculate_sum_pi_squared is False:
                    sum_pi_squared: (bs, response_len)
        """
        logger = logging.getLogger(__name__)
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)
        sum_pi_squared_checkpointing = self.config.get("sum_pi_squared_checkpointing", False)
        # PrefixGrouper path for shared-prefix optimization
        if self.use_prefix_grouper:
            can_use_pg = (
                not self.use_remove_padding
                and not self.use_ulysses_sp
                and not self.use_fused_kernels
                and not self.use_dynamic_bsz
            )
            if can_use_pg and "response_mask" in micro_batch and "uid" in micro_batch:
                from verl.trainer.ppo.prefix_grouper_utils import forward_micro_batch_with_prefix_grouper

                return forward_micro_batch_with_prefix_grouper(
                    micro_batch=micro_batch,
                    model=self.actor_module,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    device_name=self.device_name,
                    param_dtype=self.param_dtype,
                    use_chunking_entropy=self.config.get("entropy_from_logits_with_chunking", False),
                )

        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            _pep_aux_loss = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            # Prepare gradient mask for CoT parameter gradient masking
            gradient_mask_preprocessed = None
            if "gradient_mask" in micro_batch:
                gradient_mask_original = micro_batch["gradient_mask"]  # (batch, seq_len)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                # Unpad the gradient mask using the same indices
                if "gradient_mask" in micro_batch:
                    # Flatten gradient mask and index with same indices as input_ids
                    gradient_mask_flat = rearrange(
                        gradient_mask_original.unsqueeze(-1), "b s ... -> (b s) ..."
                    )  # (batch * seq_len, 1)
                    gradient_mask_rmpad = index_first_axis(gradient_mask_flat, indices)  # (total_nnz, 1)
                    gradient_mask_rmpad = gradient_mask_rmpad.transpose(0, 1)  # (1, total_nnz)
                    gradient_mask_preprocessed = gradient_mask_rmpad.squeeze(0)  # (total_nnz,)

                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    else:
                        position_ids_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                # Set up gradient mask context for CoT parameter gradient masking
                from verl.models.transformers.cot_parameter_gradient_masking_patch import cot_parameter_gradient_mask_context
                from contextlib import nullcontext
                grad_mask_ctx = (
                    cot_parameter_gradient_mask_context(gradient_mask_preprocessed)
                    if gradient_mask_preprocessed is not None
                    else nullcontext()
                )

                with grad_mask_ctx:
                    output = self.actor_module(
                        input_ids=input_ids_rmpad,
                        attention_mask=None,
                        position_ids=position_ids_rmpad,
                        **multi_modal_inputs,
                        use_cache=False,
                        **extra_args,
                    )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        # ((total_nnz / sp) + pad)
                        entropy_rmpad = (
                            self.compute_entropy_from_logits(logits_rmpad)
                            if not self.config.entropy_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)
                        )

                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = (
                            self.calculate_sum_pi_squared_from_logits(logits_rmpad)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(
                                self.calculate_sum_pi_squared_from_logits, logits_rmpad
                            )
                        )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = gather_outputs_and_unpad(
                            sum_pi_squared_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )

                if is_mask_all_zero:
                    log_probs = log_probs[:0]
                    if calculate_entropy:
                        entropy_rmpad = entropy_rmpad[:0]

                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if calculate_sum_pi_squared:
                    full_sum_pi_squared = pad_input(
                        hidden_states=sum_pi_squared_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                if calculate_sum_pi_squared:
                    # (bsz, response_length)
                    sum_pi_squared = full_sum_pi_squared.squeeze(-1)[:, -response_length - 1 : -1]
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                # For non-remove-padding case, use original gradient mask (no unpacking needed)
                if "gradient_mask" in micro_batch:
                    gradient_mask_preprocessed = gradient_mask_original  # (batch, seq_len)

                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                # Apply CoT masking if enabled (respecting any override)
                custom_attention_mask = attention_mask
                attention_gradient_mask = None
                use_cot_masking = getattr(self, '_cot_masking_override', self.config.get("use_cot_masking", False))

                # Check if masking was overridden
                has_override = hasattr(self, '_cot_masking_override')
                config_masking = self.config.get("use_cot_masking", False)
                if has_override and not use_cot_masking and config_masking:
                    print(f"✓ CoT masking OVERRIDE active: config={config_masking}, effective={use_cot_masking}", flush=True)

                if use_cot_masking:
                    # Get delimiter config
                    use_substring_matching = self.config.get('use_substring_delimiter_matching', False)

                    # Prepare delimiter parameters based on mode
                    if use_substring_matching:
                        # Substring matching mode
                        from verl.utils.cot_masking import tokenize_delimiter_string
                        delimiter_str = self.config.get('end_think_delimiter_str', '</think>')
                        end_think_delimiter_ids = tokenize_delimiter_string(
                            tokenizer=self.tokenizer,
                            delimiter_str=delimiter_str,
                            device=input_ids.device
                        )
                        end_think_token_id = None
                    else:
                        # Special token mode
                        end_think_token_id = self.config.get("end_think_token_id", 151666)
                        end_think_delimiter_ids = None

                    # Create 4D attention mask using unified interface
                    # - Supports both special token and substring matching modes
                    # - Uses response_length to compute response_start
                    # - Per-sample masking (each sample independent)
                    # CRITICAL: Use float32 for mask to avoid numerical instabilities with bfloat16
                    custom_attention_mask = create_cot_attention_mask(
                        input_ids=input_ids,
                        response_length=response_length,
                        end_think_token_id=end_think_token_id,
                        end_think_delimiter_ids=end_think_delimiter_ids,
                        attention_mask=attention_mask,
                        dtype=torch.float32,
                    )

                    # Log CoT masking application
                    if logger.isEnabledFor(logging.INFO):
                        batch_size_log = input_ids.shape[0]
                        if use_substring_matching:
                            from verl.utils.cot_masking import _find_subsequence_in_sequence
                            response_start_log = input_ids.shape[1] - response_length
                            num_with_end_think = sum(
                                any(_find_subsequence_in_sequence(input_ids[i, response_start_log:], delim) >= 0
                                    for delim in end_think_delimiter_ids)
                                for i in range(batch_size_log)
                            )
                        else:
                            num_with_end_think = (input_ids == end_think_token_id).any(dim=1).sum().item()
                        logger.info(f"CoT attention masking: {num_with_end_think}/{batch_size_log} have </think>")

                # Extract attention gradient mask from micro_batch if provided (only during update_policy)
                attention_gradient_mask = micro_batch.get("attention_gradient_mask", None)

                # ---- Prompt Embedding Perturbation (PEP / FACT) ----
                # Active only during policy updates (training mode, not compute_log_prob).
                # Perturbs prompt token hidden states at a chosen transformer layer,
                # leaving CoT and answer positions untouched.
                #   random    → Gaussian noise injected at the target layer
                #   worst_case → FGSM/PGD adversarial perturbation that maximises NLL of (C,A)
                # dual_loss  → also compute perturbed auxiliary NLL added to PPO loss
                _pep_hook_handle = None

                if (
                    self.config.get("use_prompt_embedding_perturbation", False)
                    and self.actor_module.training
                ):
                    _pep_epsilon          = self.config.get("pep_epsilon", 0.05)
                    _pep_normalize        = self.config.get("pep_normalize", True)
                    _pep_layer            = self.config.get("pep_layer", 8)   # -1 = embed_tokens
                    _pep_mode             = self.config.get("pep_mode", "random")
                    _pep_pgd_steps        = self.config.get("pep_pgd_steps", 1)
                    _pep_dual_loss        = self.config.get("pep_dual_loss", False)
                    _pep_attack_ans_only  = self.config.get("pep_attack_answer_only", False)

                    _seq_len = input_ids.shape[1]
                    _response_start = _seq_len - response_length

                    # (1, seq_len, 1): 1 for prompt positions, 0 for CoT+answer
                    _prompt_pos_mask = torch.zeros(
                        1, _seq_len, 1, device=input_ids.device, dtype=torch.float32
                    )
                    _prompt_pos_mask[0, :_response_start, 0] = 1.0

                    _pep_debug = {"fired": False, "hidden_norm": 0.0, "noise_norm": 0.0}

                    def _get_pep_target():
                        if _pep_layer == -1:
                            return self.actor_module.model.embed_tokens
                        return self.actor_module.model.layers[_pep_layer]

                    def _apply_hook(module, inp, out, perturbation):
                        h = out[0] if isinstance(out, tuple) else out
                        if not _pep_debug["fired"]:
                            _pep_debug["hidden_norm"] = h.detach()[:, :_response_start, :].norm(dim=-1).mean().item()
                            _pep_debug["noise_norm"]  = perturbation.detach()[:, :_response_start, :].norm(dim=-1).mean().item()
                            _pep_debug["fired"] = True
                        ph = h + perturbation.to(h.dtype)
                        return (ph,) + out[1:] if isinstance(out, tuple) else ph

                    if _pep_mode == "worst_case":
                        hidden_size = self.actor_module.config.hidden_size
                        delta = torch.zeros(
                            batch_size, _seq_len, hidden_size,
                            device=input_ids.device, dtype=torch.float32,
                            requires_grad=True,
                        )
                        _responses  = micro_batch["responses"]
                        _resp_mask  = micro_batch.get(
                            "response_mask",
                            torch.ones_like(_responses, dtype=torch.float32),
                        )

                        for _step in range(_pep_pgd_steps):
                            _cur_delta = delta

                            def _atk_hook(module, inp, out, _d=_cur_delta):
                                h = out[0] if isinstance(out, tuple) else out
                                ph = h + (_d * _prompt_pos_mask).to(h.dtype)
                                return (ph,) + out[1:] if isinstance(out, tuple) else ph

                            _atk_handle = _get_pep_target().register_forward_hook(_atk_hook)
                            try:
                                _atk_out = self.actor_module(
                                    input_ids=input_ids,
                                    attention_mask=custom_attention_mask,
                                    position_ids=position_ids,
                                    use_cache=False,
                                    **extra_args,
                                )
                            finally:
                                _atk_handle.remove()

                            _logits_atk = _atk_out.logits[:, -response_length - 1 : -1, :].float() / temperature
                            _lp_atk     = logprobs_from_logits(_logits_atk, _responses)
                            if _pep_attack_ans_only:
                                _end_think_id = self.config.get("end_think_token_id", 151666)
                                _ans_mask = torch.zeros_like(_resp_mask)
                                for _bi in range(_responses.shape[0]):
                                    _pos = (_responses[_bi] == _end_think_id).nonzero(as_tuple=True)[0]
                                    if len(_pos) > 0:
                                        _ans_mask[_bi, _pos[0].item() + 1:] = _resp_mask[_bi, _pos[0].item() + 1:]
                                _atk_loss = -(_lp_atk * _ans_mask.float()).sum()
                            else:
                                _atk_loss = -(_lp_atk * _resp_mask.float()).sum()

                            _delta_grad = torch.autograd.grad(_atk_loss, delta, create_graph=False)[0]

                            with torch.no_grad():
                                _g    = _delta_grad * _prompt_pos_mask
                                _gnorm = _g.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
                                _delta_adv = _pep_epsilon * _g / _gnorm

                                if _step < _pep_pgd_steps - 1:
                                    _d_new  = delta.detach() + _delta_adv
                                    _dnorm  = _d_new.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
                                    _d_new  = _pep_epsilon * _d_new / _dnorm.clamp(min=_pep_epsilon)
                                    delta   = (_d_new * _prompt_pos_mask).requires_grad_(True)

                        _perturbation = _delta_adv.detach()

                    else:  # random Gaussian
                        with torch.no_grad():
                            _noise = torch.randn(
                                batch_size, _seq_len,
                                self.actor_module.config.hidden_size,
                                device=input_ids.device, dtype=torch.float32,
                            )
                            _perturbation = None

                    if _pep_mode == "worst_case":
                        _pert_static = _perturbation

                        def _pep_perturb_hook(module, inp, out):
                            return _apply_hook(module, inp, out, _pert_static)

                    else:
                        _noise_static = _noise

                        def _pep_perturb_hook(module, inp, out):
                            h = out[0] if isinstance(out, tuple) else out
                            noise = _noise_static
                            if _pep_normalize:
                                h_norm = h.detach().norm(dim=-1, keepdim=True).clamp(min=1e-6)
                                noise  = noise * h_norm
                            pert = (_pep_epsilon * noise * _prompt_pos_mask).to(h.dtype)
                            return _apply_hook(module, inp, out, pert)

                    if _pep_dual_loss:
                        _pert_handle = _get_pep_target().register_forward_hook(_pep_perturb_hook)
                        try:
                            _pert_out = self.actor_module(
                                input_ids=input_ids,
                                attention_mask=custom_attention_mask,
                                position_ids=position_ids,
                                use_cache=False,
                                **extra_args,
                            )
                        finally:
                            _pert_handle.remove()

                        _logits_pert  = _pert_out.logits[:, -response_length - 1 : -1, :].float() / temperature
                        _lp_pert      = logprobs_from_logits(_logits_pert, micro_batch["responses"])
                        _resp_mask_d  = micro_batch.get(
                            "response_mask",
                            torch.ones_like(_lp_pert, dtype=torch.float32),
                        )
                        _pep_aux_loss = (-_lp_pert * _resp_mask_d.float()).mean()

                    else:
                        _pep_hook_handle = _get_pep_target().register_forward_hook(_pep_perturb_hook)

                # Set up gradient mask contexts
                from verl.models.transformers.cot_parameter_gradient_masking_patch import cot_parameter_gradient_mask_context
                from verl.models.transformers.attention_gradient_masking_patch import attention_gradient_mask_context
                from contextlib import nullcontext

                # Context for CoT parameter gradient masking
                param_grad_mask_ctx = (
                    cot_parameter_gradient_mask_context(gradient_mask_preprocessed)
                    if gradient_mask_preprocessed is not None
                    else nullcontext()
                )

                # Context for attention gradient masking
                attn_grad_mask_ctx = (
                    attention_gradient_mask_context(attention_gradient_mask)
                    if attention_gradient_mask is not None
                    else nullcontext()
                )

                try:
                  with param_grad_mask_ctx, attn_grad_mask_ctx:
                    output = self.actor_module(
                        input_ids=input_ids,
                        attention_mask=custom_attention_mask,
                        position_ids=position_ids,
                        **multi_modal_inputs,
                        use_cache=False,
                        **extra_args,
                    )  # prevent model thinks we are generating
                finally:
                    if _pep_hook_handle is not None:
                        _pep_hook_handle.remove()

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    # Diagnostic: Check for NaN in model output logits
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        logger.error(f"❌ NaN/Inf in model output logits!")
                        logger.error(f"   Shape: {logits.shape}, Dtype: {logits.dtype}")
                        logger.error(f"   NaN count: {torch.isnan(logits).sum().item()}, Inf count: {torch.isinf(logits).sum().item()}")
                        logger.error(f"   Min: {logits[~torch.isnan(logits) & ~torch.isinf(logits)].min().item() if (~torch.isnan(logits) & ~torch.isinf(logits)).any() else 'N/A'}")
                        logger.error(f"   Max: {logits[~torch.isnan(logits) & ~torch.isinf(logits)].max().item() if (~torch.isnan(logits) & ~torch.isinf(logits)).any() else 'N/A'}")

                    logits.div_(temperature)

                    # Diagnostic: Check for NaN after temperature division
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        logger.error(f"❌ NaN/Inf in logits after temperature division! (temp={temperature})")

                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])

                    # Diagnostic: Check for NaN in log_probs
                    if torch.isnan(log_probs).any():
                        logger.error(f"❌ NaN in log_probs after logprobs_from_logits!")

                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

                        # Diagnostic: Check for NaN in entropy
                        if torch.isnan(entropy).any():
                            logger.error(f"❌ NaN in entropy after entropy_from_logits!")
                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared = (
                            self.calculate_sum_pi_squared_from_logits(logits)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.calculate_sum_pi_squared_from_logits, logits)
                        )

            outputs = {"log_probs": log_probs}
            if calculate_entropy:
                outputs["entropys"] = entropy
            if calculate_sum_pi_squared:
                outputs["sum_pi_squared"] = sum_pi_squared
            if _pep_aux_loss is not None:
                outputs["pep_aux_loss"] = _pep_aux_loss
            return outputs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            if not torch.isfinite(grad_norm):
                print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
                self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy: bool = False) -> dict[str, torch.Tensor]:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            dict[str, torch.Tensor]: a dict containing keys
                - ``log_probs``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``entropys``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``sum_pi_squared``: tensor of shape [batch_size, response_length]. torch.float32.
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)

        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Check if CoT masking should be temporarily disabled (e.g., for old_log_prob computation)
        # We can't modify the frozen config, so use an instance variable to override
        disable_cot_masking = data.meta_info.get("disable_cot_masking", False)
        old_override = getattr(self, '_cot_masking_override', None)

        if disable_cot_masking and self.config.get("use_cot_masking", False):
            # Set override flag to disable masking for this computation
            self._cot_masking_override = False
            # Print to stdout for visibility (logger might not appear in main process)
            print("=" * 80)
            print("🔓 CoT MASKING DISABLED for old_log_prob computation")
            print("=" * 80, flush=True)
            logger = logging.getLogger(__name__)
            logger.info("🔓 CoT masking DISABLED for this compute_log_prob call (old_log_prob computation)")

        try:
            select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
            non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
            if self.use_prefix_grouper:
                select_keys += [k for k in ["prompts", "response_mask"] if k in data.batch]
                if "uid" in data.non_tensor_batch:
                    non_tensor_select_keys.append("uid")

            data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

            if use_dynamic_bsz:
                max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
                micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
            else:
                micro_batches = data.split(micro_batch_size)

            log_probs_lst = []
            entropy_lst = []
            sum_pi_squared_lst = []
            for micro_batch in micro_batches:
                micro_batch = micro_batch.to(get_device_id())
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}

                # Apply attention gradient masking if enabled (for forward consistency with update_policy)
                use_attention_gradient_masking = self.config.get("use_attention_gradient_masking", False)
                if use_attention_gradient_masking:
                    # Extract input_ids and response_length
                    input_ids = model_inputs["input_ids"]
                    response_length = model_inputs["responses"].size(-1)

                    # Determine delimiter mode (special token or substring matching)
                    use_substring_matching = self.config.get('use_substring_delimiter_matching', False)

                    # Create attention gradient mask
                    from verl.models.transformers.attention_gradient_masking_patch import create_attention_gradient_mask

                    if use_substring_matching:
                        # Substring matching mode: tokenize delimiter string
                        from verl.utils.cot_masking import tokenize_delimiter_string
                        delimiter_str = self.config.get('end_think_delimiter_str', '</think>')
                        end_delimiter_ids = tokenize_delimiter_string(
                            tokenizer=self.tokenizer,
                            delimiter_str=delimiter_str,
                            device=input_ids.device
                        )
                        attention_gradient_mask = create_attention_gradient_mask(
                            input_ids=input_ids,
                            response_length=response_length,
                            end_think_delimiter_ids=end_delimiter_ids,
                            verbose=False,  # Don't log during rollout
                        )
                    else:
                        # Special token mode: use token ID
                        end_think_token_id = self.config.get('end_think_token_id', 151666)
                        attention_gradient_mask = create_attention_gradient_mask(
                            input_ids=input_ids,
                            response_length=response_length,
                            end_think_token_id=end_think_token_id,
                            verbose=False,  # Don't log during rollout
                        )

                    # Pass attention gradient mask to _forward_micro_batch
                    model_inputs["attention_gradient_mask"] = attention_gradient_mask

                with torch.no_grad():
                    outputs = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )
                log_probs_lst.append(outputs["log_probs"])
                if calculate_entropy:
                    entropy_lst.append(outputs["entropys"])
                if calculate_sum_pi_squared:
                    sum_pi_squared_lst.append(outputs["sum_pi_squared"])

            log_probs = torch.concat(log_probs_lst, dim=0)
            if calculate_entropy:
                entropys = torch.concat(entropy_lst, dim=0)
            if calculate_sum_pi_squared:
                sum_pi_squared = torch.concat(sum_pi_squared_lst, dim=0)

            if use_dynamic_bsz:
                log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
                if calculate_entropy:
                    entropys = restore_dynamic_batch(entropys, batch_idx_list)
                if calculate_sum_pi_squared:
                    sum_pi_squared = restore_dynamic_batch(sum_pi_squared, batch_idx_list)

            outputs = {"log_probs": log_probs}
            if calculate_entropy:
                outputs["entropys"] = entropys
            if calculate_sum_pi_squared:
                outputs["sum_pi_squared"] = sum_pi_squared

            # Add verification flag for CoT masking override
            if disable_cot_masking:
                outputs["_cot_masking_was_disabled"] = True

            return outputs
        finally:
            # Restore original override state
            if disable_cot_masking:
                if old_override is None:
                    # Remove the override attribute if it didn't exist before
                    if hasattr(self, '_cot_masking_override'):
                        delattr(self, '_cot_masking_override')
                else:
                    self._cot_masking_override = old_override

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        pad_token_id = data.meta_info.get("pad_token_id", 0)

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.use_prefix_grouper and "prompts" in data.batch.keys():
            select_keys.append("prompts")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = []
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
        if self.use_prefix_grouper and "uid" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {
            "actor/pg_loss": 0.0,
            "actor/kl_loss": 0.0,
        }
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # Apply CoT parameter gradient masking if enabled
                    use_cot_gradient_masking = self.config.get('use_cot_gradient_masking', False)
                    if use_cot_gradient_masking:
                        # Get config values
                        block_prompt = self.config.get('block_prompt_gradients', True)
                        block_answer = self.config.get('block_answer_gradients', True)
                        use_substring_matching = self.config.get('use_substring_delimiter_matching', False)

                        # Extract input_ids and compute response_length from model_inputs
                        input_ids = model_inputs["input_ids"]
                        response_length = model_inputs["responses"].size(-1)

                        # MODE SELECTION: Choose delimiter detection method
                        if use_substring_matching:
                            # Substring matching mode: tokenize delimiter string
                            from verl.utils.cot_masking import tokenize_delimiter_string
                            delimiter_str = self.config.get('end_think_delimiter_str', '</think>')
                            end_delimiter_ids = tokenize_delimiter_string(
                                tokenizer=self.tokenizer,
                                delimiter_str=delimiter_str,
                                device=input_ids.device
                            )
                            # Find positions using substring matching
                            prompt_mask, cot_mask, answer_mask = find_special_token_positions(
                                input_ids=input_ids,
                                response_length=response_length,
                                end_think_delimiter_ids=end_delimiter_ids,
                            )
                        else:
                            # Special token mode: use single token ID
                            end_think_token_id = self.config.get('end_think_token_id', 151666)
                            # Find positions using special token
                            prompt_mask, cot_mask, answer_mask = find_special_token_positions(
                                input_ids=input_ids,
                                response_length=response_length,
                                end_think_token_id=end_think_token_id,
                            )

                        # Create gradient mask (same for both modes)
                        gradient_mask = torch.ones_like(input_ids, dtype=torch.float32)
                        if block_prompt:
                            gradient_mask = gradient_mask.masked_fill(prompt_mask, 0.0)
                        if block_answer:
                            gradient_mask = gradient_mask.masked_fill(answer_mask, 0.0)

                        # Pass gradient mask to _forward_micro_batch for preprocessing
                        # It will be unpacked there if use_remove_padding is enabled
                        model_inputs["gradient_mask"] = gradient_mask

                    # Apply attention gradient masking if enabled (ONLY during update_policy, not rollout)
                    use_attention_gradient_masking = self.config.get("use_attention_gradient_masking", False)
                    if use_attention_gradient_masking:
                        # Determine delimiter mode (special token or substring matching)
                        use_substring_matching = self.config.get('use_substring_delimiter_matching', False)

                        # If we already computed input_ids and response_length above (from use_cot_gradient_masking),
                        # we can reuse them. Otherwise, extract them now.
                        if not use_cot_gradient_masking:
                            input_ids = model_inputs["input_ids"]
                            response_length = model_inputs["responses"].size(-1)

                        # Create attention gradient mask
                        from verl.models.transformers.attention_gradient_masking_patch import create_attention_gradient_mask

                        # Create mask with verbose logging (first time only)
                        verbose = not hasattr(self, '_attention_gradient_mask_logged')

                        if use_substring_matching:
                            # Substring matching mode: tokenize delimiter string
                            from verl.utils.cot_masking import tokenize_delimiter_string
                            delimiter_str = self.config.get('end_think_delimiter_str', '</think>')
                            end_delimiter_ids = tokenize_delimiter_string(
                                tokenizer=self.tokenizer,
                                delimiter_str=delimiter_str,
                                device=input_ids.device
                            )
                            attention_gradient_mask = create_attention_gradient_mask(
                                input_ids=input_ids,
                                response_length=response_length,
                                end_think_delimiter_ids=end_delimiter_ids,
                                verbose=verbose,
                            )
                        else:
                            # Special token mode: use token ID
                            end_think_token_id = self.config.get('end_think_token_id', 151666)
                            attention_gradient_mask = create_attention_gradient_mask(
                                input_ids=input_ids,
                                response_length=response_length,
                                end_think_token_id=end_think_token_id,
                                verbose=verbose,
                            )

                        if verbose:
                            self._attention_gradient_mask_logged = True

                        # Pass attention gradient mask to _forward_micro_batch
                        model_inputs["attention_gradient_mask"] = attention_gradient_mask

                    # Forward pass (gradient mask context set inside _forward_micro_batch)
                    outputs = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )
                    log_prob = outputs["log_probs"]
                    entropy = outputs["entropys"] if calculate_entropy else None
                    pep_aux_loss = outputs.get("pep_aux_loss", None)

                    # for fully_async_policy
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss (any function is expected to return 2 values)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    # Skip if using bypass_mode loss (metrics already computed in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    policy_loss = pg_loss
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if pep_aux_loss is not None and self.config.get("pep_dual_loss", False):
                        _pep_coef = self.config.get("pep_dual_loss_coef", 1.0)
                        policy_loss = policy_loss + _pep_coef * pep_aux_loss
                        micro_batch_metrics["actor/pep_aux_loss"] = pep_aux_loss.detach().item()

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor

                    # Backward pass. The gradient mask for CoT parameter gradient masking
                    # is stored in a ContextVar (captured by torch.utils.checkpoint via
                    # copy_context() during the original forward), so recomputation during
                    # backward automatically sees the same mask — no manual re-set needed.
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
