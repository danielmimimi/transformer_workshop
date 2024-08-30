from torchvision.transforms import transforms
from torchvision.datasets.mnist import MNIST,FashionMNIST
from torchvision.datasets.cifar import CIFAR10

def handle_dataset(name:str):
    input_dimension = 0
    if name == "MNIST" or name == "FASHTIONMNIST":
        transform= transforms.Compose( [transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,),),transforms.RandomResizedCrop(28, scale=(0.675, 1.0))])
        transform_val= transforms.Compose( [transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
        input_dimension = (1,28,28)
    else:
        transform= transforms.Compose( [transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,),),transforms.RandomResizedCrop(32, scale=(0.675, 1.0))])
        transform_val= transforms.Compose( [transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
        input_dimension = (3,32,32)
        
    if name == "MNIST":
        train_data = MNIST(root="/workspaces/phd_framework/data/MNIST",train=True,download=True,transform=transform)
        validation_data = MNIST(root="/workspaces/phd_framework/data/MNIST",train=False,download=True,transform=transform_val)
    elif name == "FASHTIONMNIST":
        train_data = FashionMNIST(root="/workspaces/phd_framework/data/FashionMNIST",train=True,download=True,transform=transform)
        validation_data = FashionMNIST(root="/workspaces/phd_framework/data/FashionMNIST",train=False,download=True,transform=transform_val)
    elif name == "CIFAR":
        train_data = CIFAR10(root="/workspaces/phd_framework/data/FashionMNIST",train=True,download=True,transform=transform)
        validation_data = CIFAR10(root="/workspaces/phd_framework/data/FashionMNIST",train=False,download=True,transform=transform_val)
    
    return train_data,validation_data,input_dimension