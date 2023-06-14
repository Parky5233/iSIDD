import os
import pickle
import tkinter.filedialog

from sklearn.neighbors import NearestCentroid
from tkinter import Tk
from tkinter.filedialog import askopenfilename

#dictionary for comparison against prediction
class_dict = {}
print("Select image vector file (.pkl)")

pkl_file = askopenfilename()
count = 0

dataset_dir = tkinter.filedialog.askdirectory()
print("Select folder of dataset with the proper classes")
for species in os.listdir(dataset_dir+"/Train"):
    class_dict[species] = count
    count += 1

#importing image vectors
with open(pkl_file,"rb") as f: #resnet_shuffled_snakeimage_vectors.pkl
    vectors = pickle.load(f)

X_train = []
y_train = []

X_test = []
y_test = []

#loading data for Nearest Centroid
for phase in vectors.keys():
    if phase == 'Train':
        for s_class in vectors[phase]:
            for sample in vectors[phase][s_class]:
                X_train.append(sample)
                y_train.append(class_dict[s_class])
    else:
        for s_class in vectors[phase]:
            for sample in vectors[phase][s_class]:
                X_test.append(sample)
                y_test.append(class_dict[s_class])

#classifying using Nearest Centroid
clf = NearestCentroid()
clf.fit(X_train,y_train)

#testing accuracy
classwise_acc = {}
class_count = {}
tot = 0
for i in range(len(os.listdir('datasets/iSIDD-regular/Train'))):
    classwise_acc[i] = 0
    class_count[i] = 0
#for i in range(len(X_test)):
tot = len(X_test)
y_pred = clf.predict(X_test)
for sample in y_test:
    class_count[sample] += 1
for i in range(len(y_pred)):
    if y_pred[i] == y_test[i]:
        classwise_acc[y_test[i]] += 1

tot_true = 0
for s_class in classwise_acc.keys():
    tot_true += classwise_acc[s_class]
    print(str(s_class)+" accuracy = "+str(classwise_acc[s_class]/class_count[s_class]))

print("Overall Accuracy = "+str(tot_true/tot))
#https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestCentroid.html