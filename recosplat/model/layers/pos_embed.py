"""Two-dimensional rotary position embeddings."""

import torch


class RoPE2D(torch.nn.Module):
    def __init__(self, freq=100.0, F0=1.0):
        super().__init__()
        self.base = freq
        self.F0 = F0
        self.cache = {}

    def get_cos_sin(self, D, seq_len, device, dtype):
        if (D, seq_len, device, dtype) not in self.cache:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2).float().to(device) / D))
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
            freqs = torch.cat((freqs, freqs), dim=-1)
            cos = freqs.cos()
            sin = freqs.sin()
            self.cache[D, seq_len, device, dtype] = (cos, sin)
        return self.cache[D, seq_len, device, dtype]

    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope1d(self, tokens, pos1d, cos, sin):
        assert pos1d.ndim == 2
        if pos1d.is_floating_point():
            # RoPE position interpolation: linearly interpolate the cos/sin
            # tables between floor() and floor()+1 indices for fractional positions.
            pos_lo = pos1d.floor().long()
            max_idx = cos.shape[0] - 1
            pos_hi = (pos_lo + 1).clamp(max=max_idx)
            frac = (pos1d - pos1d.floor()).to(cos.dtype).unsqueeze(-1)
            cos_lo = torch.nn.functional.embedding(pos_lo, cos)
            cos_hi = torch.nn.functional.embedding(pos_hi, cos)
            sin_lo = torch.nn.functional.embedding(pos_lo, sin)
            sin_hi = torch.nn.functional.embedding(pos_hi, sin)
            cos_emb = (cos_lo + (cos_hi - cos_lo) * frac)[:, None, :, :]
            sin_emb = (sin_lo + (sin_hi - sin_lo) * frac)[:, None, :, :]
            return (tokens * cos_emb) + (self.rotate_half(tokens) * sin_emb)
        cos = torch.nn.functional.embedding(pos1d, cos)[:, None, :, :]
        sin = torch.nn.functional.embedding(pos1d, sin)[:, None, :, :]
        return (tokens * cos) + (self.rotate_half(tokens) * sin)

    def forward(self, tokens, positions):
        """
        input:
            * tokens: batch_size x nheads x ntokens x dim
            * positions: batch_size x ntokens x 2 (y and x position of each token)
        output:
            * tokens after appplying RoPE2D (batch_size x nheads x ntokens x dim)
        """
        assert tokens.size(3) % 2 == 0, "number of dimensions should be a multiple of two"
        D = tokens.size(3) // 2
        assert positions.ndim == 3 and positions.shape[-1] == 2
        cos, sin = self.get_cos_sin(D, int(positions.max()) + 1, tokens.device, tokens.dtype)
        y, x = tokens.chunk(2, dim=-1)
        y = self.apply_rope1d(y, positions[:, :, 0], cos, sin)
        x = self.apply_rope1d(x, positions[:, :, 1], cos, sin)
        tokens = torch.cat((y, x), dim=-1)
        return tokens


class PositionGetter(object):
    """return positions of patches"""

    def __init__(self):
        self.cache_positions = {}

    def __call__(self, b, h, w, device, anchor_hw=None):
        if anchor_hw is None or (int(anchor_hw[0]) == h and int(anchor_hw[1]) == w):
            if (h, w) not in self.cache_positions:
                x = torch.arange(w, device=device)
                y = torch.arange(h, device=device)
                self.cache_positions[h, w] = torch.cartesian_prod(y, x)
            pos = self.cache_positions[h, w].view(1, h * w, 2).expand(b, -1, 2).clone()
            return pos
        # RoPE position interpolation path: float positions linearly mapped onto
        # the trained grid [0, anchor_h-1] x [0, anchor_w-1].
        anchor_h, anchor_w = int(anchor_hw[0]), int(anchor_hw[1])
        cache_key = (h, w, anchor_h, anchor_w)
        if cache_key not in self.cache_positions:
            y = (
                torch.linspace(0, anchor_h - 1, steps=h, device=device, dtype=torch.float32)
                if h > 1
                else torch.zeros(1, device=device, dtype=torch.float32)
            )
            x = (
                torch.linspace(0, anchor_w - 1, steps=w, device=device, dtype=torch.float32)
                if w > 1
                else torch.zeros(1, device=device, dtype=torch.float32)
            )
            self.cache_positions[cache_key] = torch.cartesian_prod(y, x)
        pos = self.cache_positions[cache_key].view(1, h * w, 2).expand(b, -1, 2).clone()
        return pos
