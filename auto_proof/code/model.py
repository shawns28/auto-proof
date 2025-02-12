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
    """ Implements an attention block.
    """
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

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_pos_embedding = nn.Parameter(torch.randn(1, 1, dim))

        self.blocks = nn.Sequential(*[
            AttentionBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, use_exp=use_exp)
            for i in range(depth)])

        # self.to_pos_embedding = nn.Linear(pos_dim, dim)

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

        # cls_tokens = self.cls_token.repeat(B, 1, 1)
        # x = torch.cat((cls_tokens, x), dim=1)

        # Add classification token entry to adjanceny matrix. 
        # adj_cls = torch.zeros(B, N + 1, N + 1, device=input.device)
        # adj_cls[:, 0, 0] = 1.
        # adj_cls[:, 1:, 1:] = adj
        adj_cls = adj

        for block in self.blocks:
            x = block(x, adj_cls)
        x = self.mlp_head(x)

        x = self.projector(x)

        # Softmax, cross entropy already does that
        # sigmoid = nn.Sigmoid()
        # x = sigmoid(x)

        return x

    def compute_loss(self, output, labels, confidences, class_weights):
        mask = labels != -1
        mask = mask.squeeze(-1)
        output = output[mask]
        labels = labels[mask]
        confidences = confidences[mask]
        # torch.set_printoptions(profile="full")

        # print("output", output)
        output = output
        labels = labels.squeeze(-1).long()
        confidences = confidences.squeeze(-1)
        # print("labels", labels)
        # Think about doing cross entropy instead with 2 classes and weights
        
        loss_function = nn.CrossEntropyLoss(reduction='none', weight=class_weights)
        losses = loss_function(output, labels)

        # print("losses", losses)

        # This is wrong but I need to fix it for focal loss
        # probs = output[labels]
        # print("probs size", probs.shape)
        # print("probs", probs)

        # loss_function = nn.BCELoss(reduction='none')
        # losses = loss_function(output, labels)

        # Should add to config
        # non_merge_weight = 0.1
        # # non_merge_mask = (output >= 0.5) & (labels == 1)
        # non_merge_mask = labels == 1
        # losses[non_merge_mask] *= non_merge_weight

        # This is currently going to stack but its for confidence weighting
        
        # This also seems to be wrong right now so lets just ignore it for now
        conf_weight = 0.5
        conf_mask = confidences == 0
        # print("labels", labels)
        # print("confis", confidences.long())
        # print("conf mask", conf_mask)
        losses[conf_mask] *= conf_weight

        # print("losses post confidence", losses)
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