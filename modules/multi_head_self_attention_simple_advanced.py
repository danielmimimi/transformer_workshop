import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttentionAdvanced(nn.Module):
    def __init__(self, input_dimension, amount_of_heads):
        super(MultiHeadSelfAttentionAdvanced, self).__init__()


        self.input_dimension = input_dimension
        self.amount_of_heads = amount_of_heads
        assert self.input_dimension % self.amount_of_heads == 0, "the input dimension must be devisible from the amount of heads"



        self.head_dimension = int(self.input_dimension/self.amount_of_heads)
        
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
        
        # Splitting heads
        query = query.view(batch_size, -1, self.amount_of_heads, self.head_dimension).permute(0, 2, 1, 3)
        key = key.view(batch_size, -1, self.amount_of_heads, self.head_dimension).permute(0, 2, 1, 3)
        value = value.view(batch_size, -1, self.amount_of_heads,self.head_dimension).permute(0, 2, 1, 3)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dimension ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention weights to value
        attention_output = torch.matmul(attention_weights, value)
        
        # Merge heads
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.input_dimension)
        
        # Project back to the original dimension
        # output = self.output_projection(attention_output)
        
        return attention_output,attention_weights # For visibility
    