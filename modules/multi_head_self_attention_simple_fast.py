import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttentionFast(nn.Module):
    def __init__(self, input_dimension, amount_of_heads):
        super(MultiHeadSelfAttentionFast, self).__init__()


        self.input_dimension = input_dimension
        self.amount_of_heads = amount_of_heads
        assert self.input_dimension % self.amount_of_heads == 0, "the input dimension must be devisible from the amount of heads"


        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.input_dimension,
                                                    num_heads=self.amount_of_heads,
                                                    kdim=self.input_dimension,
                                                    vdim=self.input_dimension,batch_first=True)
        
        self.query_projection = nn.Linear(self.input_dimension, self.input_dimension)
        self.key_projection = nn.Linear(self.input_dimension, self.input_dimension)
        self.value_projection = nn.Linear(self.input_dimension, self.input_dimension)
        
        # ONLY REQUIRED IF DIM IS REDUCED OR INCREASED
        # self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        num_patches = x.size(1)
        sequence_length = x.size(2)
        
        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x)
        
        # # Splitting heads
        # query = query.view(batch_size, -1, self.amount_of_heads, self.head_dimension).permute(0, 2, 1, 3)
        # key = key.view(batch_size, -1, self.amount_of_heads, self.head_dimension).permute(0, 2, 1, 3)
        # value = value.view(batch_size, -1, self.amount_of_heads,self.head_dimension).permute(0, 2, 1, 3)


        
        attn_output, attn_output_weights = self.multihead_attn(query, key, value)
        

        return attn_output,attn_output_weights # For visibility
    