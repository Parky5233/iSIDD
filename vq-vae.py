import os
import cv2
import sys
import math
import tqdm
import glob
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from accelerate import Accelerator

import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as T

from colorama import Fore, Back, Style
r_ = Fore.RED
b_ = Fore.BLUE
c_ = Fore.CYAN
g_ = Fore.GREEN
y_ = Fore.YELLOW
m_ = Fore.MAGENTA
sr_ = Style.RESET_ALL

#https://stackoverflow.com/questions/60607824/pytorch-imagenet-dataset
#https://www.kaggle.com/code/maunish/training-vq-vae-on-imagenet-pytorch

in_train = "datasets\ILSVRC2012_img_train.tar"
in_valid = "datasets\ILSVRC2012_img_val.tar"

config = {'lr':1e-3,'bs':16,'img_size':256,'seed':420}

def seed_it(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_it(seed=config['seed'])

train_transforms = T.Compose([
    T.Resize((config['img_size'],config['img_size'])),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    T.RandomRotation(degrees=(30, 70)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

class ImageNetDS(Dataset):
    def __init__(self,paths,augmentations):
        self.paths = paths
        self.augmentations = augmentations

    def __getitem__(self,idx):
        path = self.paths[idx]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']

        return image

    def __len__(self):
        return len(self.paths)


test_ds = ImageNetDS(in_valid,augmentations=train_transforms)
test_dl = DataLoader(test_ds, batch_size= 16, shuffle=False, num_workers=4)

dataiter = iter(test_dl)
sample = dataiter.next()

img = torchvision.utils.make_grid(sample).permute(1,2,0).numpy()
plt.figure(figsize=(15,15))
plt.imshow(img)

class VQ(nn.Module):

    def __init__(self,num_embeddings=512,embedding_dim=64,commitment_cost=0.25):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(self.num_embeddings,self.embedding_dim)
        self.embeddings.weight.data.uniform(-1/self.num_embeddings,1/self.num_embeddings)

    def forward(self,inputs):
        inputs = inputs.permute(0,2,3,1).contiguous()
        input_shape = inputs.shape

        flat_inputs = inputs.view(-1,self.embedding_dim)

        distances = torch.cdist(flat_inputs,self.embeddings.weight)

        encoding_index = torch.argmin(distances,dim=1)

        quantized = torch.index_select(self.embeddings.weight,0,encoding_index).view(input_shape)

        e_latent_loss = F.mse_loss(quantized.detach(),inputs)
        q_latent_loss = F.mse_loss(quantized,inputs.detach())
        c_loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()

        quantized = quantized.permute(0,3,1,2).contiguous()
        return c_loss, quantized

class ResBlock(nn.Module):
    def __init__(self,in_channels,out_channels,hidden_channels):
        super(ResBlock,self).__init__()
        self.resblock = nn.Sequential(nn.ReLU(inplace=True),nn.Conv2d(in_channels,hidden_channels,kernel_size=3,stride=1,padding=1,bias=False)
                                      ,nn.ReLU(inplace=True),nn.Conv2d(hidden_channels,out_channels,kernel_size=1,stride=1,bias=False))
    def forward(self,x):
        return x + self.resblock(x)

class ResStack(nn.Module):
    def __init__(self,in_channels,out_channels,hidden_channels,num_res_layers):
        super(ResStack,self).__init__()
        self.num_res_layers = num_res_layers
        self.layers = nn.ModuleList([ResBlock(in_channels,out_channels,hidden_channels) for _ in range(num_res_layers)])

    def forward(self,x):
        for i in range(self.num_res_layers):
            x = self.layers[i](x)
        return F.relu(x)

class Model(nn.Module):
    def __init__(self,num_embeddings=512,embedding_dim=64,commitment_cost=0.25):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        #encoding
        self.conv1 = nn.Conv2d(3,64,kernel_size=4,stride=2,padding=1)
        self.conv2 = nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1)
        self.conv3 = nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1)
        self.resblock1 = ResStack(128,128,64,3)

        #vq
        self.vq_conv = nn.Conv2d(128,self.embedding_dim,kernel_size=1,stride=1)
        self.vq = VQ(self.num_embeddings,self.embedding_dim,self.commitment_cost)

        #decode
        self.conv4 = nn.Conv2d(self.embedding_dim,64,kernel_size=3,stride=1,padding=1)
        self.resblock2 = ResStack(64,64,32,3)
        self.conv5 = nn.ConvTranspose2d(64,32,kernel_size=4,stride=2,padding=1)
        self.conv6 = nn.ConvTranspose2d(32,3,kernel_size=4,stride=2,padding=1)

    def encode(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.resblock1(x)
        return x

    def decode(self,quantized):
        x = self.conv4(quantized)
        x = self.resblock2(x)
        x = F.relu(self.conv5(x))
        x = self.conv6(x)
        return x

    def forward(self,inputs):
        x = self.encode(inputs)
        c_loss,quantized = self.vq(self.vq_conv(x))
        outputs = self.decode(quantized)
        rec_loss = F.mse_loss(outputs,inputs)
        loss = rec_loss + c_loss
        return loss,outputs,rec_loss

def run():
    def eval(model,valid_loader):
        model.eval()
        valid_loss = 0
        rec_loss = 0
        with torch.no_grad():
            for i, inputs in enumerate(valid_loader):
                loss,_,loss2 = model(inputs)
                valid_loss += loss.item()
                rec_loss += loss2.item()

        valid_loss /= len(valid_loader)
        rec_loss /= len(valid_loader)
        return valid_loss,rec_loss

    def train_and_eval(train_loader,valid_loader,model,optimizer,epoch,best_loss,lr_scheduler=None):
        train_loss = 0
        for i, inputs in enumerate(train_loader):
            optimizer.zero_grad()
            model.train()
            loss,_,_ = model(inputs)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if lr_scheduler:
                lr_scheduler.step()

        train_loss /= len(train_loader)
        valid_loss, rec_loss = eval(model,valid_loader)

        print(f"Epoch:{epoch} |Train Loss:{train_loss}|Valid Loss:{valid_loss}|Rec Loss:{rec_loss}")

        if rec_loss <= best_loss:
            print(f"{g_}Loss Decreased from {best_loss} to {rec_loss}{sr_}")
            best_loss = rec_loss
            torch.save(model.state_dict(),'./imagenet_vq_vae_model.bin')

        return best_loss

    accelerator = Accelerator()
    print(f"{accelerator.device} is used")

    model = Model()

    #train
    train_dataset = ImageNetDS(in_train,train_transforms)
    train_dl = DataLoader(train_dataset,batch_size=config['bs'],shuffle=True,num_workers=4)

    #valid
    valid_dataset = ImageNetDS(in_valid,train_transforms)
    valid_dl = DataLoader(valid_dataset,batch_size=config['bs'],shuffle=False,num_workers=4)

    optimizer = optim.Adam(model.parameters(),lr=config['lr'],amsgrad=False)
    lr_scheduler = None

    model,train,valid_dl,optimizer,lr_scheduler = accelerator.prepare(model,train_dl,valid_dl,optimizer,lr_scheduler)

    best_loss = math.inf

    start_time = time.time()
    for epoch in range(config["epochs"]):
        print(f"Epoch Started:{epoch}")
        best_loss = train_and_eval(train_dl,valid_dl,model,optimizer,epoch,best_loss,lr_scheduler)

        end_time = time.time()
        print(f"{m_}Time taken by epoch {epoch} is {end_time-start_time:.2f}s{sr_}")
        start_time = end_time

    return best_loss

run()

