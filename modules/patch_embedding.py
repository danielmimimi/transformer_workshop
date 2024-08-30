
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, number_of_channels, embedding_dimension, image_size, patch_size, dropout=0.0):
        super().__init__()
        self.conv1         = nn.Conv2d(number_of_channels, embedding_dimension, kernel_size=patch_size, stride=patch_size)                       # Patch Encoding
        self.pos_embedding = nn.Parameter(torch.zeros(1, (image_size // patch_size) ** 2, embedding_dimension), requires_grad=True)      # Learnable Positional Embedding
        self.cls_token     = nn.Parameter(torch.zeros(1, 1, embedding_dimension), requires_grad=True)                                    # Classification Token
        self.dropout       = nn.Dropout(dropout)

    def forward(self, x):
        B = x.shape[0]
        x = self.conv1(x)  # split via conv                                                       
        x = x.reshape([B, x.shape[1], -1])   # flatten 
        x = x.permute(0, 2, 1)  
        x = x + self.pos_embedding  # add positional embedding
        x = torch.cat((torch.repeat_interleave(self.cls_token, B, 0), x), dim=1) # add classification token at the start of every sequence
        x = self.dropout(x)
        return x