import copy
import torch
import torch.nn as nn
from typing import Any

# Adapted from ssl neuron
class GraphAttention(nn.Module):
    """ Implements GraphAttention.

    Graph Attention interpolates global transformer attention
    (all nodes attend to all other nodes based on their
    dot product similarity) and message passing (nodes attend
    to their 1-order neighbour based on dot-product
    attention).

    Attributes:
        dim: Dimensionality of key, query and value vectors.
        num_heads: Number of parallel attention heads.
        bias: If set to `True`, use bias in input projection layers.
          Default is `False`.
        use_exp: If set to `True`, use the exponential of the predicted
          weights to trade-off global and local attention.
    """
    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 bias: bool = False,
                 use_exp: bool = True) -> nn.Module:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.scale = dim ** -0.5
        self.use_exp = use_exp

        self.qkv_projection = nn.Linear(dim, dim * num_heads * 3, bias=bias)
        self.proj = nn.Linear(dim * num_heads, dim)
        
        # Weigth to trade of local vs. global attention.
        self.predict_gamma = nn.Linear(dim, 2)
        # Initialize projection such that gamma is close to 1
        # in the beginning of training.
        self.predict_gamma.weight.data.uniform_(0.0, 0.01)

        
    @torch.jit.script
    def fused_mul_add(a, b, c, d):
        return (a * b) + (c * d)

    def forward(self, x, adj):
        B, N, C = x.shape # (batch x num_nodes x feat_dim)
        qkv = self.qkv_projection(x).view(B, N, 3, self.num_heads, self.dim).permute(0, 3, 1, 2, 4)
        query, key, value = qkv.unbind(dim=3) # (batch x num_heads x num_nodes x dim)

        attn = (query @ key.transpose(-2, -1)) * self.scale # (batch x num_heads x num_nodes x num_nodes)

        # Predict trade-off weight per node
        gamma = self.predict_gamma(x)[:, None].repeat(1, self.num_heads, 1, 1)
        if self.use_exp:
            # Parameterize gamma to always be positive
            gamma = torch.exp(gamma)

        adj = adj[:, None].repeat(1, self.num_heads, 1, 1)

        # Compute trade-off between local and global attention.
        attn = self.fused_mul_add(gamma[:, :, :, 0:1], attn, gamma[:, :, :, 1:2], adj)
        
        attn = attn.softmax(dim=-1)

        x = (attn @ value).transpose(1, 2).reshape(B, N, -1) # (batch_size x num_nodes x (num_heads * dim))
        return self.proj(x)

class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> nn.Module:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)


class AttentionBlock(nn.Module):
    """ Implements an attention block."""
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_ratio: int = 4,
                 bias: bool = False,
                 use_exp: bool = True,
                 norm_layer: Any = nn.LayerNorm) -> nn.Module:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GraphAttention(dim, num_heads=num_heads, bias=bias, use_exp=use_exp)
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim=dim, hidden_dim=dim * mlp_ratio)

    def forward(self, x, a):
        x = self.norm1(x)
        x = self.attn(x, a) + x
        x = self.norm2(x)
        x = self.mlp(x) + x
        return x
    
    
class GraphTransformer(nn.Module):
    def __init__(self,
                 dim: int = 32,
                 depth: int = 5,
                 num_heads: int = 8,
                 mlp_ratio: int = 2,
                 feat_dim: int = 36,
                 num_classes: int = 1,
                 proj_dim: int = 128,
                 use_exp: bool = True) -> nn.Module:
        super().__init__()

        self.blocks = nn.Sequential(*[
            AttentionBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, use_exp=use_exp)
            for i in range(depth)])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )

        self.projector = nn.Sequential(
            nn.Linear(dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
            nn.GELU(),
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, num_classes) # num_classes
        )

        self.to_node_embedding = nn.Sequential(
            nn.Linear(feat_dim, dim * 2),
            nn.ReLU(True),
            nn.Linear(dim * 2, dim)
        )

    # Currently concatenating pos enc to node features
    def forward(self, input, adj):
        B, N, _ = input.shape

        # Compute initial node embedding.
        x = self.to_node_embedding(input)

        for block in self.blocks:
            x = block(x, adj)
        x = self.mlp_head(x)

        x = self.projector(x)

        return x

    def compute_loss(self, output, labels, confidences, dist_to_error, max_dist, class_weight, conf_weight, tolerance_weight, rank, box_cutoff, box_weight):
        mask = labels != -1 # (b, fov, 1)
        mask = mask.squeeze(-1) # (b, fov)

        output = output[mask].squeeze(-1) # (b * fov - buffer,)
        labels = labels[mask].squeeze(-1) # (b * fov - buffer,)
        confidences = confidences[mask].squeeze(-1) # (b * fov - buffer,)
        dist_to_error = dist_to_error[mask].squeeze(-1) # (b * fov - buffer,)
        rank = rank[mask].squeeze(-1) # (b * fov - buffer,)
        
        loss_function = nn.BCEWithLogitsLoss(reduction='none', pos_weight=class_weight)
        losses = loss_function(output, labels)

        rank_mask = rank >= box_cutoff
        losses[rank_mask] *= box_weight

        # Create a tolerance around labeled errors
        # There might be situations where we ignore spots where there are no errors nearby due to fov cutoff
        dist_mask = torch.logical_and(dist_to_error >= 0, dist_to_error <= max_dist)
        losses[dist_mask] *= tolerance_weight

        conf_mask = confidences == 0 # (b * fov - buffer - tolerance)
        losses[conf_mask] *= conf_weight

        return losses.mean()


def create_model(config):
    num_classes = config['model']['num_classes']

    # Create model
    model = GraphTransformer(
                 dim=config['model']['dim'], 
                 depth=config['model']['depth'], 
                 num_heads=config['model']['n_head'],
                 feat_dim=config['loader']['feat_dim'],
                 num_classes=num_classes)
    
    return model