import torch
import torch.nn as nn

class MultiLayerPerceptronBlock(nn.Module):
  def __init__(self, embedding_dims, mlp_size, mlp_dropout,forward_mul,activation_function=nn.GELU()):
    super().__init__()
    self.embedding_dims = embedding_dims
    self.mlp_size = mlp_size # can be same as embedding_dims
    self.dropout = mlp_dropout

    self.layernorm = nn.LayerNorm(normalized_shape = embedding_dims)
    self.mlp = nn.Sequential(
        nn.Linear(in_features = embedding_dims, out_features = mlp_size*forward_mul),
        activation_function,
        nn.Dropout(p = mlp_dropout),
        nn.Linear(in_features = mlp_size*forward_mul, out_features = embedding_dims),
        nn.Dropout(p = mlp_dropout)
    )

  def forward(self, x):
    return self.mlp(self.layernorm(x))