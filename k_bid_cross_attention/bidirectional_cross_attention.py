import keras
from keras import ops
from keras import layers
from einops import rearrange


class BidirectionalCrossAttention(layers.Layer):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        context_dim=None,
        dropout=0.0,
        talking_heads=False,
        prenorm=False,
    ):
        super(BidirectionalCrossAttention, self).__init__()
        context_dim = context_dim or dim
        self.dim_head = dim_head
        self.heads = heads
        self.norm = layers.LayerNormalization() if prenorm else layers.Identity()
        self.context_norm = (
            layers.LayerNormalization() if prenorm else layers.Identity()
        )
        self.scale = keras.ops.rsqrt(ops.cast(dim_head, dtype="float32"))
        inner_dim = dim_head * heads
        self.dropout = layers.Dropout(dropout)
        self.context_dropout = layers.Dropout(dropout)
        self.to_qk = layers.Dense(
            inner_dim, use_bias=False
        )
        self.context_to_qk = layers.Dense(
            inner_dim, use_bias=False
        )
        self.to_v = layers.Dense(
            inner_dim, use_bias=False
        )
        self.context_to_v = layers.Dense(
            inner_dim, use_bias=False
        )
        self.to_out = layers.Dense(dim)
        self.context_to_out = layers.Dense(context_dim)
        self.talking_heads = (
            layers.Conv2D(heads, 1, use_bias=False)
            if talking_heads
            else layers.Identity()
        )
        self.context_talking_heads = (
            layers.Conv2D(heads, 1, use_bias=False)
            if talking_heads
            else layers.Identity()
        )

    def call(self, x, context, return_attn=False):
        x = self.norm(x)
        context = self.context_norm(context)
        # get shared query/keys and values for sequence and context
        qk, v = self.to_qk(x), self.to_v(x)
        context_qk, context_v = self.context_to_qk(context), self.context_to_v(context)
        # split out head
        qk, context_qk, v, context_v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads),
            (qk, context_qk, v, context_v),
        )
        # get similarities
        sim = ops.einsum("bhid,bhjd->bhij", qk, context_qk) * self.scale
        # get attention along both sequence length and context length dimensions
        # shared similarity matrix
        attn = ops.nn.softmax(sim, axis=-1)
        context_attn = ops.nn.softmax(sim, axis=-2)
        # dropouts
        attn = self.dropout(attn)
        context_attn = self.context_dropout(context_attn)
        # talking heads
        attn = self.talking_heads(attn)
        context_attn = self.context_talking_heads(context_attn)
        # src sequence aggregates values from context, context aggregates values from src sequence
        out = ops.einsum("bhij,bhjd->bhid", attn, context_v)
        context_out = ops.einsum("bhji,bhjd->bhid", context_attn, v)
        # merge heads and combine out
        out, context_out = map(
            lambda t: rearrange(t, "b h n d -> b n (h d)"), (out, context_out)
        )

        out = self.to_out(out)
        context_out = self.context_to_out(context_out)
        if return_attn:
            return out, context_out, attn, context_attn
        return out, context_out
