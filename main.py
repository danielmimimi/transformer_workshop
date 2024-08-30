from typing import Union, List
from matplotlib import pyplot as plt
import torch
import argparse

import torch.nn as nn
from torchvision.transforms import transforms
from torchsummary import summary
from modules.vit import VisionTransformer
from prepare_dataset import handle_dataset
from utils.vit_trainer import VitTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
parser.add_argument('-d', '--dataset',default="MNIST",help="[MNIST,CIFAR,FASHTIONMNIST]")
parser.add_argument('-a', '--attention_version',default='advanced',help="[slow_simple,fast,advanced]")
parser.add_argument('-g', '--mpl_activation_function',default='GELU',help="[GELU,RELU]")
parser.add_argument('-b', '--batch_size',default=128)
parser.add_argument('-e', '--epochs',default=100)
parser.add_argument('-lr', '--learning_rate',default=5e-5)

parser.add_argument('--amount_of_patches',default=7)
parser.add_argument( '--amount_of_encoders',default=6)
parser.add_argument( '--embedding_dimension',default=128)
parser.add_argument( '--amount_of_heads',default=8)
parser.add_argument( '--mlp_ratio',default=2)
parser.add_argument( '--enable_logging_image',default=True)

def main():
    args = parser.parse_args()
    train_data,validation_data,input_dimension = handle_dataset(args.dataset)
    
    # Create vit 
    vision_transformer = VisionTransformer(input_dimension,
                      args.amount_of_patches,
                      args.amount_of_encoders,
                      args.embedding_dimension,
                      args.amount_of_heads,args.attention_version,
                      encoder_mlp_ratio=args.mlp_ratio,
                      number_of_classes=10, 
                      mlp_activation=nn.GELU() if args.mpl_activation_function == 'GELU' else nn.ReLU())

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=args.batch_size, shuffle=True)
    
    
    
    vision_transformer.to(device)
    # summary(vision_transformer,input_dimension)
    
    trainer = VitTrainer(args,vision_transformer,train_loader,validation_loader)
    trainer.train()


if __name__ == "__main__":
   main()