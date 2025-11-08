import torch
import torch.nn as nn
from einops import rearrange
import math

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(nn.functional.softplus(x))

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        # PyTorch Conv1d expects (batch, channels, length)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        return x.permute(0, 2, 1)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        return x.permute(0, 2, 1)

class Conv1dBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.conv = nn.Conv1d(inp_channels, out_channels, kernel_size, padding='same')
        self.norm = nn.GroupNorm(n_groups, out_channels)
        self.mish = Mish()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.norm(x)
        x = self.mish(x)
        return x.permute(0, 2, 1)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1) # [B, C, L]
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) n -> b h n d', h=self.heads), qkv)

        q = q * self.scale
        k = torch.softmax(k, dim=-2)
        
        context = torch.einsum('b h n d, b h n e -> b h d e', k, v)
        out = torch.einsum('b h d e, b h n d -> b h n e', context, q)
        
        out = rearrange(out, 'b h n d -> b (h d) n')
        out = self.to_out(out)
        return out.permute(0, 2, 1)

class ResidualTemporalBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, embed_dim, kernel_size=5):
        super().__init__()
        self.conv_block1 = Conv1dBlock(inp_channels, out_channels, kernel_size)
        self.conv_block2 = Conv1dBlock(out_channels, out_channels, kernel_size)
        
        self.time_mlp = nn.Sequential(Mish(), nn.Linear(embed_dim, out_channels))
        
        if inp_channels != out_channels:
            self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, t):
        t_emb = self.time_mlp(t).unsqueeze(1) # [B, 1, C]
        
        h = self.conv_block1(x)
        h = h + t_emb
        h = self.conv_block2(h)
        
        res = self.residual_conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        return h + res

class TemporalUnet(nn.Module):
    def __init__(
        self,
        horizon: int,
        transition_dim: int,
        cond_dim: int,
        dim: int = 32,
        dim_mults: tuple = (1, 2, 4, 8),
        attention: bool = False
    ):
        super().__init__()
        self.transition_dim = transition_dim
        
        # --- Time and Condition Embedding ---
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim),
        )
        self.cond_mlp = nn.Linear(cond_dim, dim)

        # --- Channel Dimensions for U-Net ---
        dims = [transition_dim] + [dim * m for m in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)

        # --- Encoder Path ---
        self.downs = nn.ModuleList([])
        for i, (dim_in, dim_out) in enumerate(in_out):
            is_last = (i >= (num_resolutions - 1))
            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=dim),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=dim),
                PreNorm(dim_out, LinearAttention(dim_out)) if attention else nn.Identity(),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))
        
        # --- Bottleneck ---
        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=dim)
        self.mid_attn = PreNorm(mid_dim, LinearAttention(mid_dim)) if attention else nn.Identity()
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=dim)
        
        # --- Decoder Path ---
        self.ups = nn.ModuleList([])
        for i, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=dim),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=dim),
                PreNorm(dim_in, LinearAttention(dim_in)) if attention else nn.Identity(),
                Upsample1d(dim_in)
            ]))

        # --- Final Layers ---
        self.final_res_block = Conv1dBlock(dim * 2, dim, kernel_size=5)
        self.final_conv = nn.Conv1d(dim, transition_dim, 1)

    def forward(self, x, cond, time):
        # Time and condition embeddings
        t = self.time_mlp(time)
        c = self.cond_mlp(cond)
        emb = t + c

        skips = []
        # Encoder
        for res1, res2, attn, downsample in self.downs:
            x = res1(x, emb)
            x = res2(x, emb)
            x = attn(x)
            skips.append(x)
            x = downsample(x)

        # Bottleneck
        x = self.mid_block1(x, emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, emb)
        
        # Decoder
        for res1, res2, attn, upsample in self.ups:
            x = torch.cat((x, skips.pop()), dim=-1)
            x = res1(x, emb)
            x = res2(x, emb)
            x = attn(x)
            x = upsample(x)

        x = torch.cat((x, skips.pop()), dim=-1)

        # Final projection
        x = self.final_res_block(x)
        x = x.permute(0, 2, 1)
        x = self.final_conv(x)
        return x.permute(0, 2, 1)