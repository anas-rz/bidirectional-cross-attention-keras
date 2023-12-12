## Bidirectional Cross Attention in Keras 3 


Ported from Phil Wang's implementation [here](https://github.com/lucidrains/bidirectional-cross-attention)

A simple cross attention that updates both the source and target in one step. The key insight is that one can do <a href="https://arxiv.org/abs/2001.04451">shared query / key attention</a> and use the attention matrix twice to update both ways. 


## Usage

```python
import keras
from k_bid_cross_attention import BidirectionalCrossAttention

video = keras.random.normal((1, 4096, 512))
audio = keras.random.normal((1, 8192, 386))

joint_cross_attn = BidirectionalCrossAttention(
    dim = 512,
    heads = 8,
    dim_head = 64,
    context_dim = 386
)

video_out, audio_out = joint_cross_attn(
    video,
    audio,
)


# attended output should have the same shape as input

assert video_out.shape == video.shape
assert audio_out.shape == audio.shape
```

## Todo

- [X] Test initial implementation with all backends
- [ ] Add masking
- [ ] PyPI release
