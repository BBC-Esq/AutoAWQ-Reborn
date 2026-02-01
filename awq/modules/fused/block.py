import os
import torch.nn as nn
from awq.modules.fused.attn import QuantAttentionFused


class MixtralBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        n_heads,
        n_kv_heads,
        qkv_layer,
        o_proj,
        moe,
        norm_1,
        norm_2,
        dev,
        max_seq_len,
        rope_theta,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.hidden_size = hidden_size
        self.norm_1 = norm_1.to(dev)
        self.attn = QuantAttentionFused(
            self.hidden_size,
            self.n_heads,
            self.n_kv_heads,
            qkv_layer,
            o_proj,
            dev=dev,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
        ).to(dev)
        self.norm_2 = norm_2.to(dev)
        self.moe = moe
        self.device = dev

    def forward(
        self,
        hidden_states,
    ):
        norm_out = self.norm_1(hidden_states)
        attn_output, _, _ = self.attn.forward(
            hidden_states=norm_out,
        )

        h = hidden_states.to(attn_output.device) + attn_output
        out = self.moe.forward(self.norm_2(h))
        out = h + out

        return out


class LlamaLikeBlock(nn.Module):
    """
    LlamaLikeBlock is intended to be reused across blocks that have
    an architecture that closely resembles Llama, e.g. Mistral.
    """

    def __init__(
        self,
        hidden_size,
        n_heads,
        n_kv_heads,
        qkv_layer,
        o_proj,
        mlp,
        norm_1,
        norm_2,
        dev,
        max_seq_len,
        rope_theta=10000,
        partial_rotary_factor=1.0,
        head_dim=None,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = hidden_size // n_heads

        if head_dim:
            self.head_dim = head_dim

        self.hidden_size = hidden_size
        self.norm_1 = norm_1.to(dev)
        self.attn = QuantAttentionFused(
            self.hidden_size,
            self.n_heads,
            self.n_kv_heads,
            qkv_layer,
            o_proj,
            dev=dev,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            partial_rotary_factor=partial_rotary_factor,
            head_dim=head_dim,
        ).to(dev)
        self.norm_2 = norm_2.to(dev)
        self.mlp = mlp.to(dev)
        self.device = dev

    def forward(
        self,
        hidden_states,
    ):
        norm_out = self.norm_1(hidden_states)
        attn_output, _, _ = self.attn.forward(
            hidden_states=norm_out,
        )

        h = hidden_states.to(attn_output.device) + attn_output
        out = h + self.mlp.forward(self.norm_2(h))

        return out


class QwenBlock(nn.Module):
    """
    QwenBlock is intended to be reused across blocks that have
    an architecture that closely resembles Qwen2/Qwen3, e.g. use q_norm and k_norm.
    """

    def __init__(
        self,
        hidden_size,
        n_heads,
        n_kv_heads,
        qkv_layer,
        o_proj,
        mlp,
        norm_1,
        norm_2,
        dev,
        max_seq_len,
        rope_theta=10000,
        partial_rotary_factor=1.0,
        head_dim=None,
        q_norm=None,
        k_norm=None,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = hidden_size // n_heads

        if head_dim:
            self.head_dim = head_dim

        self.hidden_size = hidden_size
        self.norm_1 = norm_1.to(dev)
        self.attn = QuantAttentionFused(
            self.hidden_size,
            self.n_heads,
            self.n_kv_heads,
            qkv_layer,
            o_proj,
            dev=dev,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            partial_rotary_factor=partial_rotary_factor,
            head_dim=head_dim,
            q_norm=q_norm,
            k_norm=k_norm,
        ).to(dev)
        self.norm_2 = norm_2.to(dev)
        self.mlp = mlp.to(dev)
        self.device = dev

    def forward(
        self,
        hidden_states,
    ):
        norm_out = self.norm_1(hidden_states)
        attn_output, _, _ = self.attn.forward(
            hidden_states=norm_out,
        )

        h = hidden_states.to(attn_output.device) + attn_output
        out = h + self.mlp.forward(self.norm_2(h))

        return out


class Gemma2LikeBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        n_heads,
        n_kv_heads,
        qkv_layer,
        o_proj,
        mlp,
        norm_1,
        norm_2,
        norm_3,
        norm_4,
        dev,
        max_seq_len,
        rope_theta=10000,
        partial_rotary_factor=1.0,
        head_dim=None,
        attn_logit_softcapping=None,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = hidden_size // n_heads

        if head_dim:
            self.head_dim = head_dim

        self.hidden_size = hidden_size
        self.norm_1 = norm_1.to(dev)
        self.attn = QuantAttentionFused(
            self.hidden_size,
            self.n_heads,
            self.n_kv_heads,
            qkv_layer,
            o_proj,
            dev=dev,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            partial_rotary_factor=partial_rotary_factor,
            head_dim=head_dim,
            attn_logit_softcapping=attn_logit_softcapping,
        ).to(dev)

        self.norm_2 = norm_2.to(dev)
        self.norm_3 = norm_3.to(dev)
        self.mlp = mlp.to(dev)
        self.norm_4 = norm_4.to(dev)
        self.device = dev

    def forward(
        self,
        hidden_states,
    ):
        residual = hidden_states
        hidden_states = self.norm_1(hidden_states)

        hidden_states, _, _ = self.attn.forward(
            hidden_states=hidden_states,
        )

        hidden_states = self.norm_2(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm_3(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.norm_4(hidden_states)
        out = residual + hidden_states

        return out


class Phi3Block(nn.Module):
    """
    Phi3Block is intended to be reused across blocks that have
    an architecture that closely resembles Phi-3.
    """

    def __init__(
        self,
        hidden_size,
        n_heads,
        n_kv_heads,
        qkv_layer,
        o_proj,
        mlp,
        norm_1,
        norm_2,
        dev,
        max_seq_len,
        rope_theta=10000,
        head_dim=None,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = hidden_size // n_heads

        if head_dim:
            self.head_dim = head_dim

        self.hidden_size = hidden_size
        self.norm_1 = norm_1.to(dev)
        self.attn = QuantAttentionFused(
            self.hidden_size,
            self.n_heads,
            self.n_kv_heads,
            qkv_layer,
            o_proj,
            dev=dev,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            head_dim=head_dim,
        ).to(dev)
        self.norm_2 = norm_2.to(dev)
        self.mlp = mlp.to(dev)
        self.device = dev

    def forward(
        self,
        hidden_states,
    ):
        norm_out = self.norm_1(hidden_states)
        attn_output, _, _ = self.attn.forward(
            hidden_states=norm_out,
        )

        h = hidden_states.to(attn_output.device) + attn_output
        out = h + self.mlp.forward(self.norm_2(h))

        return out