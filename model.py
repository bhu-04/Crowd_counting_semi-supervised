import torch
import torch.nn as nn
import torchvision.models as models

class MultiScaleDilatedAttention(nn.Module):
    def __init__(self, in_dim, heads=3, window_size=3, dilation_rates=[1,2,3]):
        super().__init__()
        # NOTE: Dilated logic is not explicitly implemented with window_size or dilation_rates
        # in this attention block, but the multi-head structure is fixed.
        self.heads = heads
        self.window_size = window_size
        self.dilation_rates = dilation_rates
        self.head_dim = in_dim // heads
        self.scale = self.head_dim ** 0.5
        
        # QKV projection: in_dim -> 3 * in_dim
        self.qkv_proj = nn.Linear(in_dim, in_dim * 3, bias=False)
        self.out_proj = nn.Linear(in_dim, in_dim)
    
    def forward(self, x):
        B, N, C = x.shape
        
        # Reshape to (B, N, 3, heads, head_dim) and split into q, k, v (B, N, heads, head_dim)
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.heads, self.head_dim)
        q, k, v = [qkv[:,:,i] for i in range(3)] 
        
        # q, k, v are shape (B, N, heads, head_dim)
        # Permute to (B, heads, N, head_dim) for batch matrix multiplication
        q = q.transpose(1, 2) # (B, heads, N, head_dim)
        k = k.transpose(1, 2) # (B, heads, N, head_dim)
        v = v.transpose(1, 2) # (B, heads, N, head_dim)
        
        # Attention weights (B, heads, N, N)
        attn_weights = (q @ k.transpose(-2, -1)) / self.scale
        attn_scores = attn_weights.softmax(dim=-1)
        
        # Output (B, heads, N, head_dim)
        out = attn_scores @ v
        
        # Concatenate heads and restore original feature dimension
        out = out.transpose(1, 2).reshape(B, N, C) # (B, N, C)
        
        out = self.out_proj(out)
        return out

class LocationEnhancedAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.qkv_proj = nn.Linear(in_dim, in_dim*3)
        self.norm1 = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim*4),
            nn.ReLU(),
            nn.Linear(in_dim*4, in_dim)
        )
        self.dwconv = nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x, img_shape):
        B, N, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        
        # Standard Self-Attention
        attn_scores = (q @ k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        attn = attn_scores.softmax(-1)
        out = attn @ v
        
        # Positional/Location Enhancement via DWConv
        # Permute to (B, C, N) and reshape to (B, C, Hc, Wc)
        x_img = x.permute(0,2,1).reshape(B, C, img_shape[0], img_shape[1])
        x_pe = self.dwconv(x_img)
        # Reshape back to (B, N, C)
        x_pe = x_pe.flatten(2).transpose(1,2) 
        
        enriched = out + x_pe
        # Feed-Forward Network
        enriched = enriched + self.dropout(self.mlp(self.norm2(enriched))) # Add residual connection
        return enriched

class CrowdCCT(nn.Module):
    def __init__(self):
        super().__init__()
        # Use VGG-16 for simplicity/compatibility if Densenet weights fail, but sticking to original.
        dnet = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(*list(dnet.features.children())[:-1])
        self.reduce = nn.Conv2d(1024, 768, 1)
        self.flatten = nn.Flatten(2)
        self.msda = MultiScaleDilatedAttention(768)
        self.lea = LocationEnhancedAttention(768)
        self.regressor = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        f = self.cnn(x)
        f = self.reduce(f)
        B, C, Hc, Wc = f.shape
        x_flat = f.flatten(2).transpose(1,2)
        msda_out = self.msda(x_flat)
        lea_out = self.lea(x_flat, (Hc, Wc))
        fused = msda_out * lea_out
        fused = fused.mean(dim=1)
        count = self.regressor(fused)
        return count.squeeze(-1)