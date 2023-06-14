import copy
import os
import time
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

ds_prompt = 'Provide the name of the folder containing your images. Options include: '
for ds in os.listdir("datasets"):
    ds_prompt = ds_prompt + " "+ds

path = "datasets/"+input(ds_prompt+" \n").strip()

def calc_cwb(beta=0.9):
    #calculates balance factor on effective number of samples
    #count number of images/class
    counts = [0]*21
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
    counts = [0] * 21
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
    r.seed(x)
    np.random.seed(x)
    torch.manual_seed(x)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(x)

#Things to test:
#1. ViT ce - trained on regular images, evaluated on segmented images - ViT_wce_False_seg_False_0.7667_FT_ONECYCLE.pkl - ViT_wce_False_seg_False_0.7667_FT_ONECYCLE_output_on_seg.png
#2. ResNet ce - trained on regular iamges, evalauted on segmented images - resnet_wce_False_seg_False_0.5333_FT_ONECYCLE.pkl - resnet_wce_False_seg_False_0.5333_FT_ONECYCLE_output_on_seg.png
#3. ViT wce - trained on regular images, evalauted on segmented images - ViT_wce_True_seg_False_0.8000_FT_ONECYCLE.pkl - ViT_wce_True_seg_False_0.8000_FT_ONECYCLE_output_on_seg.png
#4. ViT wce - trained on segmented iamges, evaluated on segmented images - ViT_wce_True_seg_True_0.7333_FT_ONECYCLE - ViT_wce_True_seg_True_0.7333_FT_ONECYCLE_output_on_seg.png
#5. ViT ce - trained on segmented images, evaluated on segmented images - ViT_wce_False_seg_True_0.7333_FT_ONECYCLE.pkl - ViT_wce_False_seg_True_0.7333_FT_ONECYCLE_output_on_seg.png


#setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
torch.cuda.empty_cache()
set_seed()

#hyperparameters
batch_size = 16
inp = input('Please provide the model you wish to use \'ViT\' or \'resnet\': ')
model_name = inp.strip() #ViT or resnet
shuffle = False
epochs = 15
lr = 1e-3
inp = input('Are you using weighted ce? (Y) (N): ')
if inp.lower() == 'y':
    weighted_ce = True
else:
    weighted_ce = False

#loading dataset and transforming for input to ViT
train_transforms = T.Compose([
    T.Resize((224,224)),
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
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

save_name = model_name+"_wce_"+str(weighted_ce)+"_"+path.split("/")[1]+"_"

train_dataset = torchvision.datasets.ImageFolder(root=path+"/Train",transform=train_transforms)
val_dataset = torchvision.datasets.ImageFolder(root=path+"/Val",transform=val_transforms)

train_loader = DL.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
val_loader = DL.DataLoader(val_dataset,batch_size=len(val_dataset),shuffle=False)

datasets = {"train":train_dataset, "val":val_dataset}
dataloaders = {"train":train_loader,"val":val_loader}

ft_train_dataset = torchvision.datasets.ImageFolder(root="datasets/iNaturalist-SS/Train", transform=train_transforms)
ft_val_dataset = torchvision.datasets.ImageFolder(root="datasets/iNaturalist-SS/Val", transform=val_transforms)

ft_train_loader = DL.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
ft_val_loader = DL.DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

ft_datasets = {"train": train_dataset, "val": val_dataset}
ft_dataloaders = {"train": train_loader, "val": val_loader}

#######################################################################################################################

#setting up ViT model
if model_name == "ViT":
    model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT,progress=True).to(device)
    model.heads = nn.Sequential(nn.Linear(768,21,bias=True)).to(device)
else:#setting up ResNet101 model
    model = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.DEFAULT,progress=True).to(device)
    model.fc = nn.Linear(in_features=2048,out_features=21,bias=True).to(device)

#print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
if weighted_ce:
    criterion = nn.CrossEntropyLoss(weight=calc_alph()).to(device)
else:
    criterion = nn.CrossEntropyLoss().to(device)

#######################################################################################################################

#model training
def train_model(model,criterion,optimizer,scheduler,num_epochs,dataloader,dataset):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

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
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataset[phase])
            epoch_acc = running_corrects.double() / len(dataset[phase])

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since

    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model, best_acc

#confusion matrix function
def eval_model(model,filename):
    model.to('cpu')
    y_preds = []
    y_true = []
    for inputs, labels in dataloaders["val"]:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_preds.extend(preds)
        y_true.extend(labels.data)

    classes = train_dataset.classes

    cm = confusion_matrix(y_true,y_preds)
    df_cm = pd.DataFrame(cm, index=[i for i in classes], columns=[i for i in classes])

    plt.figure(figsize=(12, 7))
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    sb.heatmap(df_cm, annot=True)
    plt.savefig("eval_outputs/"+filename.split("/")[6].replace(".pkl","")+'_output_on_'+path.split("/")[1]+'.png')
    print("Confusion matrix saved as "+filename.split("/")[6].replace(".pkl","")+'_output_on_'+path.split("/")[1]+'.png')

#https://discuss.pytorch.org/t/feature-extraction-in-torchvision-models-vit-b-16/148029
def get_img_emb(image):
    '''
    Due to ViT's linear projection, the output is a 1x151296 vector
    :param image:
    :return:
    '''
    model.eval()
    if model_name == 'ViT':
        layer = model._modules.get('encoder').ln
        emb = torch.zeros(151296)
    else:
        layer = model._modules.get('avgpool')
        emb = torch.zeros(2048)
    img = Image.open(image).convert('RGB')
    trans_img = Variable((val_transforms(img)).unsqueeze(0)).to(device) #not sure about the unsqueeze

    def copy_data(m,i,o):
        emb.copy_(o.data.reshape(o.data.size(1)*o.data.size(2)))
    h = layer.register_forward_hook(copy_data)
    model(trans_img)
    h.remove()
    return emb.tolist()


if __name__ == '__main__':

    inp = input('Are you training your model enter \'Y\' if so and \'N\' if evaluating: ')

    fine_tune = True

    if inp == "Y":#we are training
        inp = input('Are you training/using a model fine-tuned on the INaturalist dataset? Enter \'Y\' if so and \'N\' if not: ')
        if inp.lower().strip() == 'y':
            fine_tune = True
        else:
            fine_tune = False
        if fine_tune:
            #finetuning on the INaturalist dataset
            inp = input('Are you loading a pre-trained model? enter \'Y\' if so and \'N\' if pre-training a model for the first time: ')
            if inp == "N": #pre-training model for first time
                train_size = len(ft_train_dataset)
                fine_tune_epochs = 15
                print("Pretraining on INat Dataset")
                model, acc = train_model(model, criterion, optimizer,
                                         torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr,
                                                                             steps_per_epoch=round(train_size / batch_size),
                                                                             epochs=1, pct_start=0.99), 1, ft_dataloaders,
                                         ft_datasets)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr / 2)
                model, acc = train_model(model, criterion, optimizer,
                                         torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr / 2,
                                                                             steps_per_epoch=round(train_size / batch_size),
                                                                             epochs=fine_tune_epochs, pct_start=0.3), fine_tune_epochs,
                                         ft_dataloaders, ft_datasets)
                str_acc = str(acc).split(",")[0].split("(")[1]
                torch.save(model.state_dict(), "INat_Models/"+save_name + str_acc + "_INAT_ONECYCLE.pkl")
            else:#using a previously pre-trained model and loading it to fine-tune on SID
                Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
                filename = askopenfilename(defaultextension=".pickle")
                if not filename is None:
                    print(filename.split("/")[5])
                    model.load_state_dict(torch.load(filename))
            #fine-tuning on chosen snake dataset from SID
            train_size = len(train_dataset)
            print("-" * 10)
            print("Fine-tuning on "+path)
            model, acc = train_model(model, criterion, optimizer,
                                     torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr,
                                                                         steps_per_epoch=round(train_size / batch_size),
                                                                         epochs=1, pct_start=0.99), 1, dataloaders,
                                     datasets)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr / 2)

            model, acc = train_model(model, criterion, optimizer,
                                     torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr / 2,
                                                                         steps_per_epoch=round(train_size / batch_size),
                                                                         epochs=epochs, pct_start=0.3), epochs,
                                     dataloaders, datasets)
            str_acc = str(acc).split(",")[0].split("(")[1]

            name = save_name + str_acc + "_FT_ONECYCLE.pkl"
            torch.save(model.state_dict(), "iSIDD_models/"+name)#saving models to folder
        else:
            #fine-tuning model directly on SID dataset without prior pre-training on INat (just default pytorch weights)
            train_size = len(train_dataset)

            model, acc = train_model(model,criterion,optimizer,torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=lr,steps_per_epoch=round(train_size/batch_size),
                                                                                                   epochs=1,pct_start=0.99), 1,dataloaders,datasets)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr/2)

            model, acc = train_model(model,criterion,optimizer,torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=lr/2,steps_per_epoch=round(train_size/batch_size),
                                                                                                   epochs=epochs,pct_start=0.3), epochs,dataloaders,datasets)
            str_acc = str(acc).split(",")[0].split("(")[1]
            torch.save(model.state_dict(),"iSIDD_models/"+save_name+str_acc+"_ONECYCLE.pkl")
    else: #evaluating the model on a given dataset and producing confusion matrix
        Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
        filename = askopenfilename(defaultextension=".pickle")
        if not filename is None:
            print(filename.split("/")[5])
            model.load_state_dict(torch.load(filename))
            eval_model(model,filename)
        # print("Extracting image embeddings")
        # image_vectors = {"Train":{},"Val":{}}
        # for s_class in os.listdir(path + "/Train"):
        #     image_vectors["Train"][s_class] = []
        #     image_vectors["Val"][s_class] = []
        # for phase in ['Train', 'Val']:
        #     for species in os.listdir(path + "/" + phase):
        #         for image in os.listdir(path + "/" +phase+ "/" + species):
        #             image_vectors[phase][species].append(
        #                 get_img_emb(path + "/" + phase + "/" + species + "/" + image))
        # with open(model_name+'_'+path+'_image_vectors.pkl','wb') as fp:
        #     pickle.dump(image_vectors,fp)
        #     print("dict saved")