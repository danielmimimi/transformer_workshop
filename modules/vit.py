
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from modules.encoder import Encoder
from modules.patch_embedding import PatchEmbedding




class VisionTransformer(nn.Module):
    def __init__(self,image_shape,
                 amount_of_patches ,
                 amount_of_encoders,
                 embedding_dimension,
                 amount_of_heads,
                 multi_head_attention_version = "",
                 encoder_mlp_ratio=2,
                 number_of_classes=10,
                 general_dropout_rate=0.1,
                 mlp_activation=nn.GELU()):
        super(VisionTransformer,self).__init__()
        
        self.image_shape = image_shape # (C,H,W) of image
        self.amount_of_patches = amount_of_patches
        self.amount_of_encoders = amount_of_encoders
        self.embedding_dimension = embedding_dimension
        self.amount_of_heads = amount_of_heads
        self.encoder_mlp_ratio = encoder_mlp_ratio
        self.number_of_classes = number_of_classes
        self.multi_head_attention_version = multi_head_attention_version

        assert self.image_shape[1] % self.amount_of_patches == 0, "Hight not patchable"
        assert self.image_shape[2] % self.amount_of_patches == 0, "Width not patchable"
   
        self.patch_size = (self.image_shape[1] / self.amount_of_patches,self.image_shape[2] / self.amount_of_patches)   
        
        # self.image_shape[1] must be squared
        self.patch_embedding = PatchEmbedding(number_of_channels=self.image_shape[0],embedding_dimension=self.embedding_dimension,image_size=self.image_shape[1],patch_size=int(self.patch_size[0]))
           
        self.transformer_encoder_stack = nn.ModuleList([
            Encoder(embedding_dimension = self.embedding_dimension,
                    mlp_dropout = general_dropout_rate,
                    mlp_size = self.embedding_dimension,
                    forward_mul = self.encoder_mlp_ratio,
                    num_heads = amount_of_heads,
                    multi_head_attention_version = self.multi_head_attention_version,
                    mlp_activation=mlp_activation) 
            for _ in range(amount_of_encoders)])
        
    
        self.classifier = nn.Sequential(nn.LayerNorm(normalized_shape = self.embedding_dimension),
                                nn.Linear(in_features = self.embedding_dimension,
                                         out_features = self.number_of_classes))
    
    def forward(self, x):
        batch_size, channel, height, width = x.shape

        tokens_with_positional_encoder_ready_for_encoder = self.patch_embedding(x)

        visualized_multi_head_self_attention_of_each_encoder_layer = []
        for encoder in self.transformer_encoder_stack:
            tokens_with_positional_encoder_ready_for_encoder,attention_maps = encoder(tokens_with_positional_encoder_ready_for_encoder)
            visualized_multi_head_self_attention_of_each_encoder_layer.append(attention_maps)
           
        return F.log_softmax(self.classifier(tokens_with_positional_encoder_ready_for_encoder[:,0])),visualized_multi_head_self_attention_of_each_encoder_layer
