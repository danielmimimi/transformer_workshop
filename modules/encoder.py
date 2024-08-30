import torch.nn as nn

from modules.mlp_block import MultiLayerPerceptronBlock
from modules.multi_head_self_attention_simple_advanced import MultiHeadSelfAttentionAdvanced
from modules.multi_head_self_attention_simple_fast import MultiHeadSelfAttentionFast
from modules.multi_head_self_attention_simple_slow import MultiHeadSelfAttentionSimpleSlow

class Encoder(nn.Module):
    def __init__(self, embedding_dimension = 128,
                    mlp_dropout=0.1,
                    mlp_size = 128,
                    num_heads = 8,
                    forward_mul = 2,
                    multi_head_attention_version = "",
                    mlp_activation=nn.GELU()
                    ):
        super().__init__()
        
        self.layer_norm_1 = nn.LayerNorm(embedding_dimension)
        self.layer_norm_2 = nn.LayerNorm(embedding_dimension)
        
        if multi_head_attention_version == "slow_simple":
            self.multi_head_self_attention = MultiHeadSelfAttentionSimpleSlow(input_dimension = embedding_dimension,
                                                    amount_of_heads = num_heads)
        elif multi_head_attention_version == "fast":
            self.multi_head_self_attention = MultiHeadSelfAttentionFast(input_dimension = embedding_dimension,
                                                    amount_of_heads = num_heads)
        else:
            self.multi_head_self_attention = MultiHeadSelfAttentionAdvanced(input_dimension = embedding_dimension,
                                                    amount_of_heads = num_heads)

        self.final_mlp_with_activation = MultiLayerPerceptronBlock(embedding_dims = embedding_dimension,
                                                        mlp_size = mlp_size,
                                                        mlp_dropout = mlp_dropout,
                                                        forward_mul = forward_mul,
                                                        activation_function=mlp_activation
                                                        )
        
    def forward(self,input_sequence):
        normalized_attention_maps, maps_to_visualize = self.multi_head_self_attention(self.layer_norm_1(input_sequence))
        
        residual_connection_1 = input_sequence+normalized_attention_maps
        
        linear_transformed_results = self.final_mlp_with_activation(self.layer_norm_2(residual_connection_1))
        
        residual_connection_2 = residual_connection_1+linear_transformed_results

        return residual_connection_2,maps_to_visualize