import copy
import os
import time
import tkinter.filedialog

import sklearn
import torch
import torch.nn as nn
import torch.utils.data.dataloader as DL
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import numpy as np
from PIL import Image
import random as r
from sklearn.metrics import confusion_matrix
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from torch.autograd import Variable
import shutil

#selecting dataset folder
print("Select your dataset folder (should contain two subfolders Train/Val)")
dataset_folder = tkinter.filedialog.askdirectory()
class_num = len(os.listdir(dataset_folder+"/Train"))
path = dataset_folder

#class imbalance weighting functions
def calc_cwb(beta=0.9):
    #calculates balance factor on effective number of samples
    #count number of images/class
    counts = [0]*class_num
    i = 0
    tot = 0
    for folder in os.listdir("/"+path+"/Train"):
        counts[i] = len(os.listdir("/"+path+"/Train/"+folder))
        tot += counts[i]
        i += 1
    #calculate class-wise balance factor from ratios of classes
    cwb = [0]*len(counts)
    for index in range(len(counts)):
        cwb[index] = (1 - beta) / (beta ** counts[index])
    #returning weights as a tensor
    return torch.cuda.FloatTensor(cwb)

def calc_alph(beta=0.9):
    counts = [0] * class_num
    i = 0
    tot = 0
    for folder in os.listdir(path+"/Train"):
        counts[i] = len(os.listdir(path+"/Train/" + folder))
        tot += counts[i]
        i += 1
    weights = []
    for count in counts:
        b = (count/tot)
        weights.append( (1 - b) / (len(counts)-1))
    return torch.cuda.FloatTensor(weights)

def set_seed(x=42):
    '''
    This function sets the seed for reproducibility
    '''
    r.seed(x)
    np.random.seed(x)
    torch.manual_seed(x)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(x)

#setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
torch.cuda.empty_cache()
set_seed()

#hyperparameters
batch_size = 16
resolution = 224 #1000: 58.49, 512: 56.6, 224: 31.13
print("Your image resolution is set to "+str(resolution))
inp = input('Please provide the model you wish to use \'ViT\'(1) or \'resnet\'(2): ')
model_name = inp.strip().lower() #ViT or resnet
shuffle = False
epochs = 100
lr = 1e-3
inp = input('Are you using weighted ce? (Y) (N): ')
if inp.lower() == 'y':
    weighted_ce = True
else:
    weighted_ce = False

#loading dataset and transforming for input to model
train_transforms = T.Compose([
    T.Resize((resolution,resolution)),
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

val_transforms = T.Compose([
    T.Resize((resolution,resolution)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

#building the filename for later when we save the model weights
save_name = model_name+"_wce_"+str(weighted_ce)+"_"+path.split("/")[1]+"_e"+str(epochs)+"_"

#preparing datasets using dataloaders
train_dataset = torchvision.datasets.ImageFolder(root=path+"/Train",transform=train_transforms)
val_dataset = torchvision.datasets.ImageFolder(root=path+"/Val",transform=val_transforms)

train_loader = DL.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
val_loader = DL.DataLoader(val_dataset,batch_size=len(val_dataset),shuffle=False)

datasets = {"train":train_dataset, "val":val_dataset}
dataloaders = {"train":train_loader,"val":val_loader}

#setting up model
if model_name == "1":
    model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT,progress=True).to(device)
    model.heads = nn.Sequential(nn.Linear(768,class_num,bias=True)).to(device)
else:#setting up ResNet101 model
    model = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.DEFAULT,progress=True).to(device)
    model.fc = nn.Linear(in_features=2048,out_features=class_num,bias=True).to(device)

#setting up optimizer with correct weighting function
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
if weighted_ce:
    criterion = nn.CrossEntropyLoss(weight=calc_alph()).to(device)
else:
    criterion = nn.CrossEntropyLoss().to(device)

#model training
def train_model(model,criterion,optimizer,scheduler,num_epochs,dataloader,dataset,plot):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    acc_vec = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        print("-"*10)

        for phase in ['train','val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs,labels in dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _,preds = torch.max(outputs,1)
                    loss = criterion(outputs,labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

                for pred in range(0,len(preds)):
                    if preds[pred] == labels.data[pred]:
                        running_corrects += 1

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataset[phase])
            epoch_acc = running_corrects / len(dataset[phase])

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if plot and phase == 'val':
                acc_vec.append(epoch_acc)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since

    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val acc: {best_acc:4f}')

    if plot:
        acc_vec = list(torch.tensor(acc_vec,device="cpu"))
        plt.plot(list(range(0,len(acc_vec))),acc_vec)
        if model_name.__eq__("1"):
            plt.title("ViT "+str(resolution))
        else:
            plt.title("ResNet "+str(resolution))
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.savefig(model_name+"_e"+str(epochs)+"acc_plot_"+str(resolution)+".png")

    model.load_state_dict(best_model_wts)
    return model, best_acc

#confusion matrix function
def eval_model(model,filename):
    print("moving model to cpu")
    model.to('cpu')
    model.eval()
    y_preds = []
    y_true = []
    print("beginning evaluation")
    with torch.no_grad():
        for inputs, labels in dataloaders["val"]:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            print(len(preds))
            # y_preds.extend(preds)
            # y_true.extend(labels.data)
    print("generating/saving performance metrics")
    classes = train_dataset.classes
    y_true = labels.data
    y_preds = preds
    cm = confusion_matrix(y_true,y_preds)
    accuracy = round(sklearn.metrics.accuracy_score(y_true, y_preds),4)
    df_cm = pd.DataFrame(cm, index=[i for i in classes], columns=[i for i in classes])
    bal_acc = round(sklearn.metrics.balanced_accuracy_score(y_true, y_preds),4)
    print("Accuracy: "+str(accuracy)+", Balanced Accuracy: "+str(bal_acc))
    plt.figure(figsize=(50, 30))
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    sb.heatmap(df_cm, annot=True)
    print("Please specify a save folder")
    save_folder = tkinter.filedialog.askdirectory()
    plt.savefig(save_folder+"/"+filename.split("/")[6].replace(".pkl","")+'_output_on_'+path.split("/")[1]+'_res_'+str(resolution)+'_balacc_'+str(bal_acc)+'.png')
    print("Confusion matrix saved as "+filename.split("/")[6].replace(".pkl","")+'_output_on_'+path.split("/")[1]+'_res_'+str(resolution)+'_balacc_'+str(bal_acc)+'.png')

def sort_data():
    '''
    This method takes an unsorted dataset and splits it into Train/Val based on the specified train_split
    '''
    new_folder = 'datasets/mass_1k_final'
    train_split = 0.85
    for folder in os.listdir(dataset_folder):
        #making folders for sorted dataset
        if not os.path.isdir(new_folder+"/Train/"+folder):
            os.mkdir(new_folder+"/Train/"+folder)
        if not os.path.isdir(new_folder+"/Val/"+folder):
            os.mkdir(new_folder+"/Val/"+folder)

        count = 0
        size = len(os.listdir(dataset_folder+"/"+folder))
        if size == 2:
            lim = 1
        elif size == 3:
            lim = 2
        elif size == 4:
            lim = 3
        else:
            lim = round(len(os.listdir(dataset_folder+"/"+folder))*train_split)
        for image in os.listdir(dataset_folder+"/"+folder):
            if count < lim:
                shutil.copy(dataset_folder+"/"+folder+"/"+image, new_folder+"/Train/"+folder)
            else:
                shutil.copy(dataset_folder+"/"+folder+"/"+image, new_folder+"/Val/"+folder)
            count += 1


if __name__ == '__main__':
    inp = input('Are you training your model enter \'Y\' if so and \'N\' if evaluating: ')

    if inp.lower().strip() == "y":  # we are training
        # training model directly on SID dataset
        train_size = len(train_dataset)

        model, acc = train_model(model, criterion, optimizer, torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr,
                                                                                                  steps_per_epoch=round(
                                                                                                      train_size / batch_size),
                                                                                                  epochs=1,
                                                                                                  pct_start=0.99), 1,
                                 dataloaders, datasets, plot=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr / 2)

        model, acc = train_model(model, criterion, optimizer,
                                 torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr / 2,
                                                                     steps_per_epoch=round(train_size / batch_size),
                                                                     epochs=epochs, pct_start=0.3), epochs, dataloaders,
                                 datasets,plot=True)
        str_acc = str(round(acc,4))
        print("Please specify a save folder")
        save_folder = tkinter.filedialog.askdirectory()
        torch.save(model.state_dict(), save_folder + "/" + save_name + str_acc + "_ONECYCLE.pkl")
    else:
        Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
        filename = askopenfilename(defaultextension=".pickle")
        if not filename is None:
            print(filename.split("/")[5])
            print("Loading model weights")
            model.load_state_dict(torch.load(filename))
            eval_model(model, filename)