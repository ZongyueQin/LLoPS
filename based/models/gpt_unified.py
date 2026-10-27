import logging
import math
import re
from collections import OrderedDict, namedtuple
from collections.abc import Sequence
from functools import partial
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import GPT2Config
import hydra
import numpy as np

from .block import Block
from flash_attn.modules.embedding import GPT2Embeddings, ParallelGPT2Embeddings
from flash_attn.modules.mha import MHA, ParallelMHA
from flash_attn.modules.mlp import (
    FusedMLP,
    # GatedMlp,
    Mlp,
    ParallelFusedMLP,
    # ParallelGatedMlp,
    ParallelMLP,
)
from .mlp import GatedMlp, ParallelGatedMlp
from flash_attn.ops.activations import sqrelu_fwd
from flash_attn.utils.distributed import all_gather_raw, get_dim_for_local_rank, sync_shared_params
# from flash_attn.utils.generation import GenerationMixin
from based.generation import GenerationMixin, NaiveGenerationMixin
from flash_attn.utils.pretrained import state_dict_from_pretrained

try:
    from flash_attn.ops.fused_dense import ColumnParallelLinear
except ImportError:
    ColumnParallelLinear = None

try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
except ImportError:
    dropout_add_layer_norm = None

from based.ops.triton.layer_norm import RMSNorm

try:
    from flash_attn.ops.triton.mlp import FusedDenseSqreluDense
except ImportError:
    FusedDenseSqreluDense = None

logger = logging.getLogger(__name__)

# torch.backends.cuda.matmul.allow_tf32 = False   # FLAG

from based.utils.hf import load_config_hf, load_state_dict_hf

# Import sketch-related components from attention_based.py
from based.models.gpt import (
    GPT2MixerConfig,
    create_mixer_cls,
    create_mha_cls,
    create_mlp_cls,
    create_block,
    GPTPreTrainedModel,
    _init_weights,
    DecayClass,
    GPTModel,
    GPTLMHeadModel
)

# ==================== Sketch Components ====================
from based.models.mixers.linear_attention import LinearAttention
from based.generation import InferenceParams

class SketchAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        #feature_map: FeatureMap = TaylorExp,
        l_max: int = 2048,
        feature_dim: int = 256,
        head_dim: int = None,
        num_heads: int = 16,
        power: int = 1,
        eps: float = 1e-12,
        layer_idx: int = None,
        use_linear_term: bool = False,
        linear_term_dim: int = 16,
        parallel_implementation: str="quadratic",
        **kwargs
    ):
        super().__init__()

        self.layer_idx = layer_idx
        self.d_model = d_model
        self.l_max = l_max
        self.eps = eps
        self.parallel_implementation = parallel_implementation
        self.power = power

        # set dimension
        self.num_heads = num_heads
        self.head_dim = self.d_model // self.num_heads if head_dim is None else head_dim
        self.feature_dim = feature_dim
        self.use_linear_term = use_linear_term
        self.linear_term_dim = linear_term_dim
        self.r2  = math.sqrt(2)
        self.rd  = math.sqrt(math.sqrt(feature_dim))
        self.rrd = math.sqrt(self.rd)

        # initialize projections and feature map
        #self.feature_map = feature_map
        if use_linear_term:
            self.proj_q = nn.Linear(self.d_model, self.linear_term_dim * self.num_heads, bias=False)
            self.proj_k = nn.Linear(self.d_model, self.linear_term_dim * self.num_heads, bias=False)

        self.proj_q1 = nn.Linear(self.d_model, self.feature_dim * self.num_heads, bias=False)
        self.proj_k1 = nn.Linear(self.d_model, self.feature_dim * self.num_heads, bias=False)
        self.proj_q2 = nn.Linear(self.d_model, self.feature_dim * self.num_heads, bias=False)
        self.proj_k2 = nn.Linear(self.d_model, self.feature_dim * self.num_heads, bias=False)

        self.proj_v = nn.Linear(self.d_model, self.num_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.d_model, bias=False)

        if self.parallel_implementation == "tk":
            self.g_norm = RMSNorm(self.head_dim, eps=1e-5)

    @classmethod
    def from_based_linear_attention(cls, linear_attention_layer: LinearAttention, power: int = 1,
                                    use_linear_term: bool = True):
        """
        create sketch attention based on a based linear attention checkpoint,
        after converting, the computation result should be exactly the same,
        but has better expressiveness for continual training
        """
        # first get the dimensions and input arguments
        d_model = linear_attention_layer.d_model
        l_max = linear_attention_layer.l_max
        feature_dim = linear_attention_layer.feature_dim * linear_attention_layer.feature_dim
        head_dim = linear_attention_layer.head_dim
        num_heads = linear_attention_layer.num_heads
        linear_term_dim = linear_attention_layer.feature_dim
        eps = linear_attention_layer.eps
        layer_idx = linear_attention_layer.layer_idx
        pass

        # create the SketchAttention checkpoint
        layer = SketchAttention(
            d_model,
            l_max,
            feature_dim,
            head_dim,
            num_heads,
            power,
            eps,
            layer_idx,
            use_linear_term,
            linear_term_dim,
            parallel_implementation="quadratic",
        )

        # TODO convert and load parameters
        assert linear_attention_layer.proj_q.bias is None
        assert linear_attention_layer.proj_k.bias is None
        if use_linear_term:
            layer.proj_q.load_state_dict(linear_attention_layer.proj_q.state_dict())
            layer.proj_k.load_state_dict(linear_attention_layer.proj_k.state_dict())
        def convert_weight(W):
            W = W.reshape(num_heads, linear_term_dim, d_model)
            W1 = W.repeat_interleave(linear_term_dim, dim=1)
            W1 = W1.reshape(num_heads*feature_dim, d_model)
            W2 = W.repeat(1, linear_term_dim, 1)
            W2 = W2.reshape(num_heads*feature_dim, d_model)
            return W1, W2
        W = linear_attention_layer.proj_q.weight.data
        W1, W2 = convert_weight(W)
        layer.proj_q1.weight.data.copy_(W1)
        layer.proj_q2.weight.data.copy_(W2)

        W = linear_attention_layer.proj_k.weight.data
        W1, W2 = convert_weight(W)
        layer.proj_k1.weight.data.copy_(W1)
        layer.proj_k2.weight.data.copy_(W2)

        layer.proj_v.load_state_dict(linear_attention_layer.proj_v.state_dict())
        layer.out_proj.load_state_dict(linear_attention_layer.out_proj.state_dict())
        return layer


    def forward(self,
        hidden_states: torch.Tensor,
        inference_params: InferenceParams = None,
        *args: any,
        **kwargs: any
    ):
        """
        x (torch.Tensor): tensor of shape (b, d, l)
        y (torch.Tensor): tensor of shape (b, d, l)
        """
        b, l, _ = hidden_states.size()
        q1, k1, v = self.proj_q1(hidden_states), self.proj_k1(hidden_states), self.proj_v(hidden_states)
        q2, k2 = self.proj_q2(hidden_states), self.proj_k2(hidden_states)
        if self.power == 1:
            q = q1 * q2
            k = k1 * k2
            #q = q.abs()
            #k = k.abs()
        else:
            assert self.power == 2
            q = (q1 * q2)**2
            k = (k1 * k2)**2
            #TODO
            #q = (q1 * q2).abs()
            #k = (k1 * k2).abs()


        q = q.view(b, l, self.num_heads, self.feature_dim).transpose(1, 2)/self.r2/self.rd
        k = k.view(b, l, self.num_heads, self.feature_dim).transpose(1, 2)/self.r2/self.rd
        v = v.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        if self.use_linear_term:
            q_linear = self.proj_q(hidden_states)
            k_linear = self.proj_k(hidden_states)
            q_linear = q_linear.view(b, l, self.num_heads, self.linear_term_dim).transpose(1, 2)
            k_linear = k_linear.view(b, l, self.num_heads, self.linear_term_dim).transpose(1, 2)
            q = torch.cat([q[...,:1]**0, q_linear/self.rrd, q], dim=-1)
            k = torch.cat([k[...,:1]**0, k_linear/self.rrd, k], dim=-1)

        self.is_inference = inference_params is not None
        if inference_params is None:
            return self.parallel_forward(hidden_states, q, k, v)
        else:
            # check if we are doing prefill or generation
            if inference_params.seqlen_offset > 0:
                # recurrent
                kv_state, k_state = self._get_inference_cache(inference_params)
                return self.recurrent_forward(hidden_states, kv_state, k_state, q, k, v)
            else:
                # prefill
                y, kv_state, k_state = self.parallel_forward(hidden_states, q, k, v)
                if self.layer_idx in inference_params.key_value_memory_dict:
                    # # update the state in-place when graph caching is enabled
                    inference_params.key_value_memory_dict[self.layer_idx][0].copy_(kv_state)
                    inference_params.key_value_memory_dict[self.layer_idx][1].copy_(k_state)
                else:
                    inference_params.key_value_memory_dict[self.layer_idx] = (kv_state, k_state)
                return y

    def parallel_forward(self, x: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):

        if self.parallel_implementation == "tk":
            raise NotImplementedError

        elif self.parallel_implementation == "quadratic":

            A_qk = torch.einsum("bhnd,bhmd->bhnm", q, k)
            try:
                A_qk = torch.tril(A_qk)
            except:
                # tril is incompatible with certain data types
                b, h, l, l = A_qk.shape
                cumsum_matrix = torch.tril(torch.ones((l, l))).to(q.device, q.dtype)
                A_qk = A_qk * cumsum_matrix
            y = torch.einsum("bhnm,bhme->bhne", A_qk.to(x.dtype), v.to(x.dtype))
            z = 1 / (torch.einsum("bhld,bhld->bhl", q, k.cumsum(2)) + self.eps)
            y = y * z[..., None]
            y = rearrange(y, 'b h l d -> b l (h d)')

        elif self.parallel_implementation == "linear":

            v = causal_dot_product(q.contiguous().to(dtype=torch.float32), k.contiguous().to(dtype=torch.float32),v.contiguous().to(dtype=torch.float32),)
            z = 1 / (
                torch.einsum(
                    "bhld,bhld->bhl",
                    q.to(dtype=torch.float32),
                    k.to(dtype=torch.float32).cumsum(2)
                ) + self.eps
            )
            y = v * z[..., None]
            y = rearrange(y, 'b h l d -> b l (h d)')

        elif self.parallel_implementation == "fla_parallel":
            raise NotImplementedError

        elif self.parallel_implementation == "fla_chunk":
            raise NotImplementedError

        else:
            raise ValueError(f"Parallel implementation {self.parallel_implementation} not supported")

        if self.is_inference and self.parallel_implementation != "tk":
            kv_state = torch.einsum("bhnd,bhnf->bhfd", k, v)[:, :, None]
            k_state = k.sum(dim=2)[:, :, None, None]

        if self.is_inference:
            return self.out_proj(y.to(x.dtype)), kv_state, k_state
        else:
            return self.out_proj(y.to(x.dtype))

    def recurrent_forward(self, hidden_states: torch.Tensor, kv_state: torch.Tensor, k_state: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, decay: torch.Tensor=None):
        """
        Compute linear attention with recurrent view
        -> Assume q.shape is (b, h, 1, d); k and v.shape are (b, h, l, d)
        """
        b, h, l, d = q.shape
        assert l == 1, f'q.shape is {q.shape} but should be ({b}, {h}, 1, {d})'
        # Expand dims for broadcasting to compute linear attention
        q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1)

        kv_state += k[:, :, -1:] * v[:, :, -1:]
        k_state  += k[:, :, -1:]

        # Compute linear attention
        num = (q * kv_state).sum(dim=-1)
        if 'fla' in self.parallel_implementation:
            eps = 1e-6 # this code uses an alternate eps
        else:
            eps = 1e-12
        y = num / ((q * k_state).sum(dim=-1) + eps)

        y = rearrange(y, 'b h l d -> b l (h d)').to(q.dtype)
        return self.out_proj(y)


    def expanded_size(self):
        return self.feature_dim if not self.use_linear_term else self.feature_dim + self.linear_term_dim + 1

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype=None, **kwargs):
        """Creates a state tensor of shape ..."""

        kv_shape = (
            batch_size, self.num_heads, 1, self.head_dim, self.expanded_size()
        )
        k_shape = (
            batch_size, self.num_heads, 1, 1, self.expanded_size()
        )
        kv_state = torch.zeros(*kv_shape, dtype=dtype, device=self.out_proj.weight.device)
        k_state = torch.zeros(*k_shape, dtype=dtype, device=self.out_proj.weight.device)
        return (kv_state, k_state)

    def _get_inference_cache(self, inference_params: InferenceParams):
        return inference_params.key_value_memory_dict[self.layer_idx]


# ==================== Sketch Model Components ====================

class GPTSketchConfig(GPT2Config):
    def __init__(self, *args, **kwargs):
        # Sketch-specific configurations
        self.sketch_config = kwargs.pop("sketch_config", None)
        self.use_sketch_attention = kwargs.pop("use_sketch_attention", False)
        super().__init__(*args, **kwargs)


"""
def create_sketch_mixer_cls(config, layer_idx=None, process_group=None, device=None, dtype=None):
    #TODO probably needs to change this
    factory_kwargs = {"device": device, "dtype": dtype}

    # If sketch attention is enabled for this layer
    if getattr(config, "use_sketch_attention", False) and hasattr(config, "sketch_config"):
        sketch_config = config.sketch_config

        # Add layer-specific configurations
        sketch_config_with_layer = sketch_config.copy()
        sketch_config_with_layer["layer_idx"] = layer_idx

        # Create partial SketchAttention
        mixer_cls = partial(
            SketchAttention,
            config=sketch_config_with_layer,
            **factory_kwargs,
        )
        return mixer_cls
    else:
        # Fall back to standard attention
        return create_mha_cls(config, layer_idx, process_group=process_group, device=device, dtype=dtype)
"""

def create_sketch_block(config, layer_idx=None, process_group=None, device=None, dtype=None, **kwargs):
    """Create block with sketch attention capability"""
    factory_kwargs = {"device": device, "dtype": dtype}
    sequence_parallel = getattr(config, "sequence_parallel", True)

    # Use sketch mixer if configured
    mixer_cls = create_mixer_cls(config, layer_idx, process_group=process_group, **factory_kwargs)
    mlp_cls = create_mlp_cls(config, layer_idx, process_group=process_group, **factory_kwargs)

    use_rms_norm = getattr(config, "rms_norm", False)
    norm_cls = partial(
        nn.LayerNorm if not use_rms_norm else RMSNorm,
        eps=config.layer_norm_epsilon,
        **factory_kwargs,
    )

    # TD [2022-07-30]: Force residual in fp32, seems to make fp16 training more stable
    residual_in_fp32 = getattr(config, "residual_in_fp32", False)
    resid_dropout1 = config.resid_pdrop if layer_idx is None or layer_idx > 0 else config.embd_pdrop
    prenorm = getattr(config, "prenorm", True)
    parallel_block = getattr(config, "parallel_block", False)

    if not parallel_block:
        block = Block(
            config.hidden_size,
            mixer_cls,
            mlp_cls,
            norm_cls=norm_cls,
            prenorm=prenorm,
            resid_dropout1=resid_dropout1,
            resid_dropout2=config.resid_pdrop,
            fused_dropout_add_ln=False,
            residual_in_fp32=residual_in_fp32,
            sequence_parallel=sequence_parallel and process_group is not None,
            mark_shared_params=process_group is not None,
            layer_idx=layer_idx,
        )
    else:
        assert prenorm
        block = ParallelBlock(
            config.hidden_size,
            mixer_cls,
            mlp_cls,
            norm_cls=norm_cls,
            resid_dropout1=resid_dropout1,
            resid_dropout2=config.resid_pdrop,
            tied_norm=getattr(config, "parallel_block_tied_norm", False),
            fused_dropout_add_ln=False,
            residual_in_fp32=residual_in_fp32,
            sequence_parallel=sequence_parallel and process_group is not None,
            mark_shared_params=process_group is not None,
        )
    block.layer_idx = layer_idx
    return block


class GPTSketchModel(GPTModel):
    """GPT model with sketch attention capability"""

    def __init__(self, config: GPTSketchConfig, process_group=None, device=None, dtype=None):
        # Initialize as standard GPT model first
        super().__init__(config, process_group=process_group, device=device, dtype=dtype)

        # Replace layers with sketch-enabled blocks if configured
        if getattr(config, "use_sketch_attention", False):
            factory_kwargs = {"device": device, "dtype": dtype}
            self.layers = nn.ModuleList(
                [
                    create_sketch_block(config, layer_idx=i, process_group=process_group, **factory_kwargs)
                    for i in range(config.num_hidden_layers)
                ]
            )

    def get_sketch_auxiliary_loss(self):
        """Get auxiliary loss from sketch components"""
        total_loss = 0.0
        count = 0

        for layer in self.layers:
            # Check if this layer has sketch attention
            if hasattr(layer, 'mixer') and hasattr(layer.mixer, 'feature_map_q'):
                if hasattr(layer.mixer.feature_map_q, 'get_auxiliary_loss'):
                    total_loss += layer.mixer.feature_map_q.get_auxiliary_loss()
                    count += 1
                if hasattr(layer.mixer.feature_map_k, 'get_auxiliary_loss'):
                    total_loss += layer.mixer.feature_map_k.get_auxiliary_loss()
                    count += 1

        return total_loss / max(count, 1)  # Avoid division by zero


class GPTSketchLMHeadModel(GPTLMHeadModel):
    """GPT Language Model with sketch attention capability"""

    def __init__(self, config, power=1, use_linear_term=True, process_group=None, device=None, dtype=None):
        # Convert to GPTSketchConfig if needed
        if not isinstance(config, GPTSketchConfig):
            if isinstance(config, dict):
                config = GPTSketchConfig(**config)
            else:
                # Convert existing GPT2Config to GPTSketchConfig
                config_dict = config.__dict__.copy()
                config = GPTSketchConfig(**config_dict)

        # Note: We need to be careful about initialization order
        factory_kwargs = {"device": device, "dtype": dtype}
        self.power = power
        self.use_linear_term = use_linear_term

        # Initialize GPTPreTrainedModel directly
        GPTPreTrainedModel.__init__(self, config)
        self.process_group = process_group

        # Use GPTSketchModel instead of GPTModel if sketch attention is enabled
        if getattr(config, "use_sketch_attention", False):
            self.transformer = GPTSketchModel(config, process_group=process_group, **factory_kwargs)
        else:
            # Use standard GPTModel for compatibility
            self.transformer = GPTModel(config, process_group=process_group, **factory_kwargs)

        # Initialize language modeling head (same as GPTLMHeadModel)
        self.tie_word_embeddings = getattr(config, "tie_word_embeddings", True)
        lm_head_bias = getattr(config, "lm_head_bias", False)
        pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
        vocab_size = (
            math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple
        )

        # This option is for OPT-350m
        word_embed_proj_dim = getattr(config, "word_embed_proj_dim", None)
        embed_dim = config.n_embd if word_embed_proj_dim is None else word_embed_proj_dim
        if word_embed_proj_dim is not None:
            self.project_out = nn.Linear(config.n_embd, embed_dim, bias=False, **factory_kwargs)
        else:
            self.project_out = None

        if process_group is None:
            self.lm_head = nn.Linear(embed_dim, vocab_size, bias=lm_head_bias, **factory_kwargs)
        else:
            if ColumnParallelLinear is None:
                raise ImportError("fused_dense_lib is not installed")
            self.lm_head = ColumnParallelLinear(
                embed_dim,
                vocab_size,
                process_group,
                bias=lm_head_bias,
                sequence_parallel=getattr(config, "sequence_parallel", True),
                **factory_kwargs,
            )

        # Initialize weights and apply final processing
        self.apply(
            partial(
                _init_weights,
                n_layer=config.num_hidden_layers,
                initializer_range=config.initializer_range,
            )
        )
        self.tie_weights()

    def change_stage(self):
        return
#        for idx, layer in enumerate(self.transformer.layers):
#            if isinstance(layer.mixer, LinearAttention):
#                layer.mixer = SketchAttention.from_based_linear_attention(layer.mixer, self.power,
#                                self.use_linear_term)
#                print(f"change layer {idx} power: {self.power}, use_linear_term: {self.use_linear_term}")
#                break
#        print(f"change stage, no layers to change")
#        return


    @classmethod
    def from_pretrained_hf(cls, pretrained_model_name, device=None,
                           power=None, use_linear_term=None, **kwargs):
        """
        Load a pretrained model from HuggingFace Hub, compatible with based checkpoints
        """
        config_data = load_config_hf(pretrained_model_name)
        saved_power = config_data.get("llops_power")
        saved_use_linear_term = config_data.get("llops_use_linear_term")
        if power is None:
            power = saved_power
        if use_linear_term is None:
            use_linear_term = saved_use_linear_term

        # Legacy checkpoints predate the LLoPS metadata fields.  Keep the
        # paper's 360M architecture as the backwards-compatible default, while
        # allowing callers to specify the 1B architecture explicitly.
        if power is None:
            power = 2
        if use_linear_term is None:
            use_linear_term = False
        config = GPTSketchConfig(**config_data)

        # By default, don't enable sketch attention when loading pretrained models
        # Users can enable it later or specify it in kwargs
        config.use_sketch_attention = kwargs.pop('use_sketch_attention', False)
        if config.use_sketch_attention:
            config.sketch_config = kwargs.pop('sketch_config', get_example_sketch_config())

        model = cls(config, power=power, use_linear_term=use_linear_term, device=device, **kwargs)
        state_dict = load_state_dict_hf(pretrained_model_name, device=device)

        # remove the 'model.' prefix from the keys
        state_dict = {re.sub("^model\.", "", k): v for k, v in state_dict.items()}
        # remove Unexpected key(s) in state_dict: "train_metrics.num-tokens.count", "val_metrics.num-tokens.count", "test_metrics.num-tokens.count". from the state_dict
        state_dict = {k: v for k, v in state_dict.items() if "metrics" not in k}


        is_llops_checkpoint = any(
            key.endswith(".mixer.proj_q1.weight") for key in state_dict
        )

        if not is_llops_checkpoint:
            # A BASED checkpoint must be loaded before replacing its linear
            # attention layers, so the original projection weights can seed
            # the LLoPS feature maps.
            model.load_state_dict(state_dict)
            for idx, layer in enumerate(model.transformer.layers):
                if isinstance(layer.mixer, LinearAttention):
                    layer.mixer = SketchAttention.from_based_linear_attention(layer.mixer, power,
                                    use_linear_term)
        else:
            # An LLoPS checkpoint already contains the learned sketch
            # projections, so instantiate the matching modules before loading.
            for idx, layer in enumerate(model.transformer.layers):
                if isinstance(layer.mixer, LinearAttention):
                    layer.mixer = SketchAttention.from_based_linear_attention(layer.mixer, power,
                                    use_linear_term)
            model.load_state_dict(state_dict)


        # running debugging check for my loading function
        """
        input_x = torch.randint(100, (1,200)).cuda()
        print("input")
        print(input_x)
        model.to("cuda")
        out = model(input_x)
        h1 = model.transformer(input_x)
        print(out)
        model.to("cpu")

        for layer in model.transformer.layers:
            if isinstance(layer.mixer, LinearAttention):
                layer.mixer = SketchAttention.from_based_linear_attention(layer.mixer, power,
                                    use_linear_term)
        model.to("cuda")
        out2 = model(input_x)
        print(out2)
        error = (out.logits-out2.logits).abs().max()
        print("error", error.item())
        h2 = model.transformer(input_x)
        print(h1)
        print(h2)
        error = (h1-h2).abs()
        print("hidden states error", error.max().item(), error.mean().item(), error.median().item())

        import os
        os._exit(0)
        """


        return model.to(device=device)

    @classmethod
    def from_pretrained(
        cls,
        model_name,
        config,
        *args,
        strict=True,
        device=None,
        dtype=None,
        world_size=1,
        rank=0,
        **kwargs,
    ):
        """
        Load a pretrained model from checkpoint, compatible with based checkpoints
        """
        # Convert config to GPTSketchConfig if needed
        if not isinstance(config, GPTSketchConfig):
            if isinstance(config, dict):
                config = GPTSketchConfig(**config)
            else:
                config_dict = config.__dict__.copy()
                config = GPTSketchConfig(**config_dict)

        # Instantiate model
        model = cls(config, *args, device=device, dtype=dtype, **kwargs)

        # Load state_dict in cpu because we already initialized the model in GPU
        state_dict = state_dict_from_pretrained(model_name, device="cpu", dtype=dtype)
        if model_name.startswith("gpt2"):
            from .....attention_based import remap_state_dict_hf_gpt2
            state_dict = remap_state_dict_hf_gpt2(state_dict, config)
        else:
            raise NotImplementedError(f"Model {model_name} not supported")
        if world_size > 1:
            from .....attention_based import shard_state_dict_tp
            state_dict = shard_state_dict_tp(state_dict, config, world_size, rank)
        load_return = model.load_state_dict(state_dict, strict=strict)
        logger.info(load_return)
        return model

    def get_sketch_auxiliary_loss(self):
        """Get auxiliary loss from sketch components"""
        if hasattr(self.transformer, 'get_sketch_auxiliary_loss'):
            return self.transformer.get_sketch_auxiliary_loss()
        else:
            return torch.tensor(0.0, device=next(self.parameters()).device)

    def enable_sketch_attention(self, sketch_config=None):
        """
        Enable sketch attention on an existing model

        Args:
            sketch_config: Dictionary with sketch configuration parameters
        """
        if sketch_config is None:
            sketch_config = get_example_sketch_config()

        # Update config
        self.config.use_sketch_attention = True
        self.config.sketch_config = sketch_config

        # Replace transformer with sketch-enabled version
        factory_kwargs = {"device": next(self.parameters()).device, "dtype": next(self.parameters()).dtype}
        new_transformer = GPTSketchModel(self.config, process_group=self.process_group, **factory_kwargs)

        # Copy existing weights where possible
        self._copy_transformer_weights(self.transformer, new_transformer)

        # Replace transformer
        self.transformer = new_transformer

        logger.info("Sketch attention enabled successfully")

    def _copy_transformer_weights(self, source_transformer, target_transformer):
        """Helper method to copy weights from source to target transformer"""
        try:
            # Copy embeddings
            target_transformer.embeddings.load_state_dict(source_transformer.embeddings.state_dict())

            # Copy non-sketch layers
            for i, (source_layer, target_layer) in enumerate(zip(source_transformer.layers, target_transformer.layers)):
                # Only copy if the target layer doesn't use sketch attention
                if not (hasattr(target_layer, 'mixer') and hasattr(target_layer.mixer, 'feature_map_q')):
                    target_layer.load_state_dict(source_layer.state_dict())

            # Copy final layer norm if it exists
            if hasattr(source_transformer, 'ln_f') and hasattr(target_transformer, 'ln_f'):
                target_transformer.ln_f.load_state_dict(source_transformer.ln_f.state_dict())

        except Exception as e:
            logger.warning(f"Could not copy all weights during sketch attention enablement: {e}")

    def disable_sketch_attention(self):
        """
        Disable sketch attention and revert to standard attention
        """
        # Update config
        self.config.use_sketch_attention = False

        # Replace transformer with standard version
        factory_kwargs = {"device": next(self.parameters()).device, "dtype": next(self.parameters()).dtype}
        new_transformer = GPTModel(self.config, process_group=self.process_group, **factory_kwargs)

        # Copy existing weights where possible
        self._copy_transformer_weights(self.transformer, new_transformer)

        # Replace transformer
        self.transformer = new_transformer

        logger.info("Sketch attention disabled successfully")

    def forward(self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0, stream=None, return_aux_loss=False, **kwargs):
        """
        Forward pass with optional auxiliary loss computation

        Args:
            return_aux_loss: If True, return auxiliary loss along with logits
        """
        # Standard forward pass
        result = super().forward(
            input_ids=input_ids,
            position_ids=position_ids,
            inference_params=inference_params,
            num_last_tokens=num_last_tokens,
            stream=stream,
            **kwargs
        )

        if return_aux_loss and False:
            aux_loss = self.get_sketch_auxiliary_loss()
            # Return both logits and auxiliary loss
            SketchCausalLMOutput = namedtuple("SketchCausalLMOutput", ["logits", "aux_loss"])
            return SketchCausalLMOutput(logits=result.logits, aux_loss=aux_loss)

        return result

    def set_sketch_mode(self, mode):
        """Set sketch mode for all sketch attention layers"""
        for layer in self.transformer.layers:
            if hasattr(layer, 'mixer') and hasattr(layer.mixer, 'feature_map_q'):
                if hasattr(layer.mixer.feature_map_q, 'set_mode'):
                    layer.mixer.feature_map_q.set_mode(mode)
                if hasattr(layer.mixer.feature_map_k, 'set_mode'):
                    layer.mixer.feature_map_k.set_mode(mode)

    def save_pretrained(self, save_directory):
        """
        Save the model and its configuration file to a directory in HuggingFace format.
        Extends the base GPTLMHeadModel save_pretrained to handle sketch-specific config.
        """
        import os
        import json

        # Ensure save_directory exists
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, 'config.json')
        config_dict = self.config.__dict__.copy()
        config_dict["architectures"] = [self.__class__.__name__]
        config_dict["llops_architecture"] = "learnable_low_rank_polynomial_sketch"
        config_dict["llops_power"] = self.power
        config_dict["llops_polynomial_degree"] = 2 * self.power
        config_dict["llops_use_linear_term"] = self.use_linear_term

        # Handle sketch-specific configuration
        if hasattr(self.config, 'use_sketch_attention') and self.config.use_sketch_attention:
            # Ensure sketch config is serializable
            if 'sketch_config' in config_dict and config_dict['sketch_config'] is not None:
                # If sketch_config is an OmegaConf object, convert it to dict
                try:
                    from omegaconf import OmegaConf
                    if OmegaConf.is_config(config_dict['sketch_config']):
                        config_dict['sketch_config'] = OmegaConf.to_container(config_dict['sketch_config'], resolve=True)
                except:
                    pass

        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        logger.info(f"Model saved to {save_directory}")


# ==================== Configuration utilities ====================

def create_sketch_config(
    base_config_dict,
    sketch_config_dict,
    sketch_layers=None
):
    """
    Create a GPTSketchConfig from base config and sketch config

    Args:
        base_config_dict: Base GPT configuration
        sketch_config_dict: Sketch attention configuration
        sketch_layers: List of layer indices to use sketch attention (None = all layers)
    """
    config = GPTSketchConfig(**base_config_dict)
    config.use_sketch_attention = True
    config.sketch_config = sketch_config_dict
    config.sketch_layers = sketch_layers

    return config


# Example usage configuration
def get_example_sketch_config():
    """Example configuration for sketch attention"""
    return {
        "transformer_dim": 1024,  # Should match model hidden size
        "head_dim": 64,           # Should match model head dimension
        "num_head": 16,           # Should match number of attention heads
        "sketch_dim": 256,        # Sketch dimension
        "sketch_mode": 1,         # 0: Based mode, 1: Flattenformer mode, 2: SRHT mode, 3: Free mode
        "sketch_dropout_prob": 0.1,
        "weight": 0.001,          # Weight for auxiliary loss
        "attn_type": "sketch",
        "multi_head": False,
        "slide": False
    }
