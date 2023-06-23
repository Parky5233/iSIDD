import PIL
import torch
import torch.nn as nn
from nopdb import nopdb
import torchvision
import torchvision.transforms as T
import numpy as np
import IPython.display as ipd
from PIL import Image
import random as r
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import tkinter

val_transforms = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

#References:
#https://colab.research.google.com/github/cifkao/nopdb/blob/main/docs/pytorch_tutorial.ipynb#scrollTo=TZgFMHMSU305
#https://github.com/tczhangzhi/VisionTransformer-Pytorch/issues/1

def set_seed(x=42):
    r.seed(x)
    np.random.seed(x)
    torch.manual_seed(x)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(x)

def inv_normalize(tensor):
    """Normalize an image tensor back to the 0-255 range."""
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min()) * (256 - 1e-5)
    return tensor


def inv_transform(tensor, normalize=True):
    """Convert a tensor to an image."""
    tensor = inv_normalize(tensor)
    array = tensor.detach().cpu().numpy()
    array = array.transpose(1, 2, 0).astype(np.uint8)
    return PIL.Image.fromarray(array)


def plot_weights(input, patch_weights):
    """Display the image, dimming each patch according to the given weight."""
    # Multiply each patch of the input image by the corresponding weight
    plot = inv_normalize(input.clone())
    print(patch_weights.shape[0])
    for i in range(patch_weights.shape[0]):
        x = i * 16 % 224
        y = i // (224 // 16) * 16
        plot[:, y:y + 16, x:x + 16] *= patch_weights[i]
    vis_att = inv_transform(plot, normalize=False)
    ipd.display(vis_att)
    img_path_arr = img_path.split("/")
    print("Please select a folder to save attention maps to")
    save_folder = tkinter.filedialog.askdirectory()
    vis_att.save(save_folder+"/"+filename.split("/")[6].replace(".pkl","") + "_layer_" + str(l) + "_attention_visual_" + img_path_arr[len(img_path_arr)-1])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
torch.cuda.empty_cache()
set_seed()

model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT,progress=True).to(device)
model.heads = nn.Sequential(nn.Linear(768,21,bias=True)).to(device)

#print(model)

Tk().withdraw()
filename = askopenfilename()
if not filename is None:
    print("select model weights to use")
    model.load_state_dict(torch.load(filename))
    model.eval()
    print("select image to visualize attention on")
    img_path = askopenfilename()
    #img_path = "datasets/iSIDD-regular/Val/cavatappi/cavatappi_2.jpg"
    x = val_transforms(Image.open(img_path)).to(device)
    print(x.size())
    for l in range(len(model.encoder.layers)):
        #grabbing values from forward pass of encoder layer
        with nopdb.capture_calls(model.encoder.layers[l].forward) as calls:
            logits = model(x.unsqueeze(0))

        #saving weights (which have been averaged across all heads) to variable
        avg_attn = calls[0].locals['avg_weights']#(using slightly modified encoder in vision_transformer.py)

        #using weight to adjust image
        img = PIL.Image.open(img_path).convert('RGB')
        input = val_transforms(img).cuda()

        inp_att = avg_attn.mean(axis=-2)
        inp_att = inp_att[0,1:]
        #print(inp_att.size())
        plot_weights(input,inp_att)#avg_attn[-1,1:,1:]