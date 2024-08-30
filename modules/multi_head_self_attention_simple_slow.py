import torch
import torch.nn as nn

class MultiHeadSelfAttentionSimpleSlow(nn.Module):
    """Really slow version of multi head attention - if confident, use torch.nn.MultiheadAttention"""
    def __init__(self, input_dimension,amount_of_heads):
        super(MultiHeadSelfAttentionSimpleSlow,self).__init__()

        self.input_dimension = input_dimension
        self.amount_of_heads = amount_of_heads

        assert self.input_dimension % self.amount_of_heads == 0, "the input dimension must be devisible from the amount of heads"

        self.head_dimension = int(self.input_dimension/self.amount_of_heads)

        # Every head has its own linear representation of k q v
        self.query_mappings = self.create_module_mappings()
        self.key_mappings = self.create_module_mappings()
        self.value_mappings =  self.create_module_mappings()

        self.softmax = nn.Softmax(dim=-1)


    def create_module_mappings(self):
        return nn.ModuleList(
            [nn.Linear(self.head_dimension,self.head_dimension) for _ in range(self.amount_of_heads)]
        )
    
    def forward(self, input_sequence):
        """Input sequence with [Batch Size, amount_of_patches ,sequence_length],
        Returns the normalized and the basic attention maps"""

        multi_head_attention_result = []
        multi_head_normalized_attention_result = []

        for single_sequence in input_sequence:
            normalized_attention_result = []
            attention_result = []

            for head in range(self.amount_of_heads):
                query_mapping = self.query_mappings[head]
                key_mapping = self.key_mappings[head]
                value_mapping = self.value_mappings[head]

                part_sequence_for_single_head = single_sequence[:,head*self.head_dimension : (head+1)*self.head_dimension]
                query = query_mapping(part_sequence_for_single_head)
                key = key_mapping(part_sequence_for_single_head)
                value = value_mapping(part_sequence_for_single_head)

                # @ Matrix Multiplication
                basic_attention = (query @ key.T)/self.head_dimension**0.5  
                normalized_attention = self.softmax(basic_attention)

                attention_result.append(torch.unsqueeze(normalized_attention,dim=0))
                normalized_attention_result.append(normalized_attention @ value)

            multi_head_attention_result.append(torch.concat(attention_result,dim=0))
            multi_head_normalized_attention_result.append(torch.hstack(normalized_attention_result))

        return torch.cat([torch.unsqueeze(single_result,dim=0) for single_result in multi_head_normalized_attention_result]),torch.cat([torch.unsqueeze(single_result,dim=0) for single_result in multi_head_attention_result])