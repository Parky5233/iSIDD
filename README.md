# iSIDD: A Unique Dataset for Individual Snake Identification Within Species

This repository contains code for our paper submitted to NEURIPS Datasets and Benchmarks Track (2023). 

## TLDR
We present a unique dataset for snake individual identification within the Massasauga rattlesnake species.

## Dependencies 
In order to run this project you must have first installed the following dependencies: 
- Python 3
- numpy: https://numpy.org/install/
- matplotlib: https://pypi.org/project/matplotlib/
- torch: https://pytorch.org/
- torchvision: https://pytorch.org/
- seaborn: https://seaborn.pydata.org/installing.html
- PIL: https://pypi.org/project/Pillow/
- scikit-learn: https://scikit-learn.org/stable/install.html
- tkinter: https://docs.python.org/3/library/tkinter.html
- pandas: https://pandas.pydata.org/docs/getting_started/install.html

## Disclaimer
In order for the visualize_attn.py code to work a line in torchvision's vision_transformer.py must be modified
Line 113 in the EncoderBlock forward must be changed:
- From this: x, _ = self.self_attention(x, x, x, need_weights=False)
- To this: x, avg_weights = self.self_attention(x, x, x, need_weights=True)

## Running Python Code
Important Notes:
- Confusion matrix output is saved in an 'eval_outputs' folder
- Model weights pretrained on iNaturalist will be stored in an 'iNat_models' folder
- Model weights fine-tuned on iSIDD will be stored in an 'iSIDD_models' folder
- Attention visualizations are saved in an 'attn_visualizations' folder 

main.py:
- This code is to be run to train or evaluate a given model (ViT, ResNet).
- When running this code follow the following steps:
  1. Provide the dataset folder (should contain two subfolders Train and Val)
  2. Provide model: 'ViT' or 'resnet'
  3. Specify if you're fine-tuning with weighted cross entropy loss 'Y' or 'N'
  4. Specify if you're fine-tuning your model or evaluating it
  
  If you're fine-tuning:
    - Specify if you're first pre-training on iNaturalist dataset
    - If yes:
      * Specify if you're loading the weights of a model already pre-trained on iNaturalist or if you with to pre-train yourself
      * If loading weights: select the model file from your 'INat_Models' folder. The code will then fine-tune on iSIDD and save the model to the 'iSIDD_models'
    - If no:
      * The code will fine-tune on iSIDD and save to 'iSIDD_models

  If you're evaluating:
    - Select your model weights to load. The confusion matrix will be saved with a name including/specifying the model you loaded and what data you evaluated it on in the 'eval_outputs' folder
  
decoupled.py:
- This code is to be run when loading the image vector embeddings for the decoupled method. 
- Currently this code opens the pkl file of the embeddings and then uses Nearest Centroid to train and classify on data
- The performance class-wise and overall accuracy is then printed out

visualize_attn.py:
- This code made use of multiple references:
  * https://colab.research.google.com/github/cifkao/nopdb/blob/main/docs/pytorch_tutorial.ipynb#scrollTo=TZgFMHMSU305
  * https://github.com/tczhangzhi/VisionTransformer-Pytorch/issues/1
- When you run the code, select the model whose weights you want to load and then an image you want to visualize attention on

Future changes to come:
- main.py having folder selection so datasets do not need to be located in the 'datasets' folder'
- decoupled.py having file selection so pkl file specification is not 'hardcoded'
- visualize_attn.py having file selection so image path is not 'hardcoded'
  


## Datasets
- iNaturalist-SS: 
  * Consists of 18,000 images and 12 classes of snake species
  * Each class has 1500 images split between training and test sets using and 85/15 train/test split
- iSIDD-regular: iSIDD dataset using regular images
- iSIDD-segmented: iSIDD dataset using segmented images (https://segment-anything.com/)
- iSIDD-straightened: iSIDD dataset using straightened images (https://ij.imjoy.io/)
