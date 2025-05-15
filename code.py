import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import os, pathlib, glob, random
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import confusion_matrix
from transformers import WhisperProcessor, WhisperForConditionalGeneration
# from datasets import load_dataset
from transformers.models.whisper.modeling_whisper import WhisperModel, WhisperEncoder
from transformers.models.whisper.configuration_whisper import WhisperConfig
from typing import Optional, Tuple, Union
import torch
import librosa 
import matplotlib.pyplot as plt
import numpy as np
import os, glob, pickle
import scipy.io as sio
from tqdm import tqdm
import multiprocessing as mp 
import torch.optim as optim
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#checking for gpu 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
cuda
print(torch.__version__)
# print(mp.cpu_count())
2.1.2
batch_size = 64
# learning_rate = 0.01
# learning_rate = 0.03
# num_epochs = 40
train_data_path = r"/kaggle/input/34-static-tecc/34_static/training"
valid_data_path = r"/kaggle/input/34-static-tecc/34_static/validation"
test_data_path = r"/kaggle/input/34-static-tecc/34_static/testing"
# drop_amount = 0.2

# class WhisperWordClassifier(nn.Module):

#   def __init__(self):
#         super().__init__()
  
#         self.hidden_size = hidden_size
#         self.lstm1 = nn.LSTM(768, 128, bidirectional=True, batch_first=True)
#         self.dropout1 = nn.Dropout(drop_amount)
#         self.lstm2 = nn.LSTM(768, 128, bidirectional=True, batch_first=True)
#         self.dropout2 = nn.Dropout(drop_amount)
#         self.fc = nn.Linear(hidden_size*2, 155)
#         self.softmax = nn.Softmax(dim=1)

#   def forward(self, out):

#         lstm_out, _ = self.lstm1(out)
#         out = self.dropout1(lstm_out)
#         lstm_out, _ = self.lstm1(out)
#         out = self.dropout1(lstm_out)
#         out = self.fc(out)
#         out = self.softmax(out)
#         return out
# # 
# del model_whisp
# torch.cuda.empty_cache()
# model_whisp = CapsuleNet()
# model_whisp.to(device)
OLD CODE MODEL DEFINED HERE:

# # Define the parameters
# input_size = 512
# hidden_size = 256
# num_layers = 2
# num_classes = 2

# # model_whisp= WhisperWordClassifier()
# model_whisp = BiLSTMClassifier(input_size, hidden_size, num_layers, num_classes)
# # model_whisp = BiGRUAudioClassifier(input_size, num_classes, hidden_size, num_layers)
# model_whisp.to(device)
Custom Dataset
import scipy
# class PtDataset(Dataset):
#     def __init__(self, directory):
#         self.directory = directory
#         self.classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
#         self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
#         self.files = []
#         for c in self.classes:
#             c_dir = os.path.join(directory, c)
#             # c_files = [(f, self.class_to_idx[c]) for f in glob.glob(os.path.join(c_dir, '*.pt'))]
#             c_files = [(os.path.join(c_dir, f), self.class_to_idx[c]) for f in os.listdir(c_dir)]
#             self.files.extend(c_files)
#         random.shuffle(self.files)

#     def __len__(self):
#         return len(self.files)
    
#     def __getitem__(self, idx):
#         filepath, label = self.files[idx]
#         # desired_dimensions = (1, 400, 1024)
        
#         try:
#             data = torch.load(filepath, map_location='cpu').detach()
#             # if(data.shape[1]<400) :
#             #     padding_amounts = (0,0,0,desired_dimensions[1] - data.shape[1])
#             #     data = F.pad(data, padding_amounts, mode='constant',value = 0)
#                 # if(len(data.shape)<3):
#                 #     print(data.shape)
#                 #     print(filepath, " pad")
#             data = data[-1, 0:100 , :]
#                 # if(len(data.shape)<3):
#                 #     print(data.shape)
#                 #     print(filepath, " no pad")
            
#         except Exception as e:
#             print(f"Error loading file {filepath}: {str(e)}")
#             return None
#         return data, label

class PtDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.files = []
        for c in self.classes:
            c_dir = os.path.join(directory, c)
            c_files = [(os.path.join(c_dir, f), self.class_to_idx[c]) for f in os.listdir(c_dir)]
            self.files.extend(c_files)
        random.shuffle(self.files)
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        filepath, label = self.files[idx]
        try:
            mat_vals = scipy.io.loadmat(filepath)
            data = mat_vals['final']
            data = data.T
            max_len=600
            if (max_len > data.shape[0]):
                pad_width = max_len - data.shape[0]
                data = np.pad(data, pad_width=((0, pad_width),(0,0)), mode='constant')
            else:
                data = data[:max_len, :]
        except Exception as e:
            print(f"Error loading file {filepath}: {str(e)}")
            return None
        return data, label
train_dataset = PtDataset(train_data_path)
valid_dataset = PtDataset(valid_data_path)
test_dataset = PtDataset(test_data_path)
Custom Dataloader
class PtDataLoader(DataLoader):
    def __init__(self, directory, batch_size, shuffle=True, num_workers=0):
        dataset = PtDataset(directory)
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
train_dataloader = PtDataLoader(directory=train_data_path, batch_size=batch_size)
valid_dataloader = PtDataLoader(directory=valid_data_path, batch_size=batch_size)
test_dataloader = PtDataLoader(directory=test_data_path, batch_size=batch_size)
train_count = len(train_dataset)
valid_count = len(valid_dataset)
test_count = len(test_dataset)

print(train_count)
print(valid_count)
print(test_count)
53862
10797
4634
TRAINING THE MODEL
# Defining loss and optimizer for BiLSTM Model
loss_function = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model_whisp.parameters(), lr=learning_rate)

# Defining loss and optimizer for CAPSNET Model
# loss_function = CapsuleLoss()
# optimizer = optim.Adam(model_whisp.parameters())
#Model training and testing 

n_total_steps = len(train_dataloader) # n_total_steps * batch size will give total number of training files (consider that last batch may not be fully filled)
train_accuracy_list = []
train_loss_list = []
valid_accuracy_list = []

max_acc = 0
pred_labels =[]
act_labels = []
pred =[]
lab =[]
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Basic Block for ResNet
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# Define the ResNet architecture
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Function to create ResNet-50 model
def resnet50(num_classes=2):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

# Example usage:
# model = resnet50(num_classes=1000) # Change num_classes to match your specific classification task.
# del model_whisp
# torch.cuda.empty_cache()
model_whisp = resnet50().to(device)
num_epochs = 15
optimizer = torch.optim.Adam(model_whisp.parameters(), lr=0.001)
# model_whisp = BiGRUAudioClassifier(input_size, num_classes, hidden_size, num_layers)
# model_whisp.to(device)
# # Create a DataLoader for the dataset
# batch_size = 64
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# # Iterate over the data loader
# for batch_images, batch_labels in train_loader:
#     # Forward pass and loss calculation
#     outputs = model_whisp(batch_images)
#     loss = loss_function(outputs, batch_labels)
#     # Rest of the training loop
#     ...
for epoch in range(num_epochs):
    # Evaluation and training on training dataset
    model_whisp.train()
    train_accuracy = 0.0
    train_loss = 0.0
    
    for batch_idx, (images, labels) in enumerate(tqdm(train_dataloader)):
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        
        optimizer.zero_grad()
        
        # Reshape the input tensor to have 4 dimensions
        images = images.unsqueeze(1)
        images = images.float()  # Convert the tensor to float if needed
        
        outputs = model_whisp(images)
        # print('Outputs shape:', outputs.shape)
        # print('Labels shape:', labels.shape)
        
        loss = loss_function(outputs, labels)
        
        loss.backward()
        optimizer.step()
        train_loss += loss.cpu().data * images.size(0)
        _, prediction = torch.max(outputs.data, 1)
        
        train_accuracy += int(torch.sum(prediction == labels.data))
        
    train_accuracy = train_accuracy / train_count
    train_loss = train_loss / train_count
    
    train_accuracy_list.append(train_accuracy)
    train_loss_list.append(train_loss)
    
    model_whisp.eval()
    valid_accuracy=0.0
    pred = []
    lab = []
    
    for i, (images,labels) in enumerate(tqdm(valid_dataloader)):
        if torch.cuda.is_available():
            images=Variable(images.cuda())
            labels=Variable(labels.cuda())
        images = images.unsqueeze(1)
        images = images.float()
        
        outputs=model_whisp(images)
        
        _,prediction=torch.max(outputs.data,1)
        valid_accuracy+=int(torch.sum(prediction==labels.data))

        pred.extend(prediction.tolist())
        lab.extend(labels.tolist())
    
    valid_accuracy= valid_accuracy / valid_count
    valid_accuracy_list.append(valid_accuracy)
    
    if(max_acc < valid_accuracy):
        pred_labels=[]
        act_labels=[]
        pred_labels = pred
        act_labels = lab
        max_acc = valid_accuracy
        torch.save(model_whisp, 'model.pth')
    
    print('Epoch : '+str(epoch+1)+'/'+str(num_epochs)+'   Train Loss : '+str(train_loss)+'   Train Accuracy : '+str(train_accuracy)+'   Valid Accuracy : '+str(valid_accuracy))
    # print('Epoch:', epoch+1, '/', num_epochs, 'Train Loss:', train_loss.item(), 'Train Accuracy:', train_accuracy)
    
print('Finished Training')
print('max accuracy', max_acc)
100%|██████████| 842/842 [09:28<00:00,  1.48it/s]
100%|██████████| 169/169 [01:38<00:00,  1.72it/s]
Epoch : 1/15   Train Loss : tensor(0.2698)   Train Accuracy : 0.8869889718168653   Valid Accuracy : 0.8234694822635917
100%|██████████| 842/842 [09:35<00:00,  1.46it/s]
100%|██████████| 169/169 [01:27<00:00,  1.92it/s]
Epoch : 2/15   Train Loss : tensor(0.2171)   Train Accuracy : 0.9117931008874531   Valid Accuracy : 0.9132166342502547
100%|██████████| 842/842 [07:56<00:00,  1.77it/s]
100%|██████████| 169/169 [01:09<00:00,  2.44it/s]
Epoch : 3/15   Train Loss : tensor(0.1911)   Train Accuracy : 0.9245664847202109   Valid Accuracy : 0.927479855515421
100%|██████████| 842/842 [08:29<00:00,  1.65it/s]
100%|██████████| 169/169 [01:06<00:00,  2.54it/s]
Epoch : 4/15   Train Loss : tensor(0.1701)   Train Accuracy : 0.9329211689131485   Valid Accuracy : 0.9360933592664629
100%|██████████| 842/842 [06:58<00:00,  2.01it/s]
100%|██████████| 169/169 [01:08<00:00,  2.48it/s]
Epoch : 5/15   Train Loss : tensor(0.1470)   Train Accuracy : 0.9434295050313765   Valid Accuracy : 0.9501713438918218
100%|██████████| 842/842 [06:54<00:00,  2.03it/s]
100%|██████████| 169/169 [01:06<00:00,  2.54it/s]
Epoch : 6/15   Train Loss : tensor(0.1279)   Train Accuracy : 0.9505217036129368   Valid Accuracy : 0.5248680188941373
100%|██████████| 842/842 [07:26<00:00,  1.89it/s]
100%|██████████| 169/169 [01:08<00:00,  2.47it/s]
Epoch : 7/15   Train Loss : tensor(0.1129)   Train Accuracy : 0.9576324681593702   Valid Accuracy : 0.9196999166435121
100%|██████████| 842/842 [08:34<00:00,  1.64it/s]
100%|██████████| 169/169 [01:28<00:00,  1.90it/s]
Epoch : 8/15   Train Loss : tensor(0.1028)   Train Accuracy : 0.9609372099067989   Valid Accuracy : 0.9546170232471983
100%|██████████| 842/842 [07:00<00:00,  2.00it/s]
100%|██████████| 169/169 [01:06<00:00,  2.52it/s]
Epoch : 9/15   Train Loss : tensor(0.0927)   Train Accuracy : 0.9646504028814378   Valid Accuracy : 0.5100490877095489
100%|██████████| 842/842 [06:47<00:00,  2.07it/s]
100%|██████████| 169/169 [01:06<00:00,  2.53it/s]
Epoch : 10/15   Train Loss : tensor(0.0856)   Train Accuracy : 0.9672310719988118   Valid Accuracy : 0.9520237102898953
100%|██████████| 842/842 [06:49<00:00,  2.06it/s]
100%|██████████| 169/169 [01:06<00:00,  2.54it/s]
Epoch : 11/15   Train Loss : tensor(0.0757)   Train Accuracy : 0.9706472095354796   Valid Accuracy : 0.9297953135130129
100%|██████████| 842/842 [06:57<00:00,  2.02it/s]
100%|██████████| 169/169 [01:40<00:00,  1.68it/s]
Epoch : 12/15   Train Loss : tensor(0.0701)   Train Accuracy : 0.973562066020571   Valid Accuracy : 0.9485042141335556
100%|██████████| 842/842 [07:28<00:00,  1.88it/s]
100%|██████████| 169/169 [01:08<00:00,  2.46it/s]
Epoch : 13/15   Train Loss : tensor(0.0636)   Train Accuracy : 0.9757899818053545   Valid Accuracy : 0.8422710012040382
100%|██████████| 842/842 [06:52<00:00,  2.04it/s]
100%|██████████| 169/169 [01:07<00:00,  2.51it/s]
Epoch : 14/15   Train Loss : tensor(0.0566)   Train Accuracy : 0.9789276298689243   Valid Accuracy : 0.8344910623321293
100%|██████████| 842/842 [07:16<00:00,  1.93it/s]
100%|██████████| 169/169 [01:24<00:00,  2.00it/s]
Epoch : 15/15   Train Loss : tensor(0.0500)   Train Accuracy : 0.9812483754780736   Valid Accuracy : 0.969621191071594
Finished Training
max accuracy 0.969621191071594
# import pickle
# torch.save(model_whisp, 'model.pth')
# import pickle
# knnPickle = open('knnpickle_file', 'wb')     
# # source, destination 
# pickle.dump(model_whisp, knnPickle)  
# # close the file
# knnPickle.close()
# Load the best model
best_model = torch.load('model.pth')

# Put the best_model in evaluation mode
best_model.eval()

# Initialize variables to store results
test_accuracy = 0.0
pred_labels = []
act_labels = []

# Pass validation data through the best model
for i, (images, labels) in enumerate(tqdm(test_dataloader)):
    if torch.cuda.is_available():
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
   
    images = images.unsqueeze(1)
    images = images.float()
    outputs = best_model(images)
    _, prediction = torch.max(outputs.data, 1)
   
    test_accuracy += int(torch.sum(prediction == labels.data))
   
    pred_labels.extend(prediction.tolist())
    act_labels.extend(labels.tolist())

# Calculate testing accuracy
test_accuracy = test_accuracy / test_count

# Print the testing accuracy
print("testing Accuracy:", test_accuracy)
100%|██████████| 73/73 [01:35<00:00,  1.30s/it]
testing Accuracy: 0.8996547259387139
plt.plot(train_accuracy_list, label='Train Accuracy')
plt.plot(valid_accuracy_list, label='Valid Accuracy')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train and Valid Accuracy')
plt.legend()

plt.show()

Confusion Matrix
# Calculate the confusion matrix
import seaborn as sns
conf_mat = confusion_matrix(act_labels, pred_labels)
# Plot confusion matrix heat map
sns.heatmap(conf_mat, cmap="flare",annot=True, fmt = "g",
            cbar_kws={"label":"color bar"},
            xticklabels=train_dataset.classes,
            yticklabels=train_dataset.classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
# plt.savefig("ConfusionMatrix_BiLSTM.png")
plt.show()
from sklearn.metrics import f1_score
f1_score = f1_score(pred_labels, act_labels, average='macro')
print('F1 Score : ', f1_score)

F1 Score :  0.8991769446236102
# labels = test_dataset.class_to_idx
# confusion_matrix = torch.zeros(len(labels), len(labels))

# # Iterate through the data and compute the predictions and true labels
# for i in actual_labels:
#     confusion_matrix[actual_labels, pred_labels] += 1

# # Print the confusion matrix
# print(confusion_matrix)

# tuples = list(zip(actual_labels, pred_labels))

# # Define a dictionary to store the frequency of each tuple
# frequency_dict = {}

# # Iterate through the list of tuples and count the frequency of each non-equal tuple
# for t in tuples:
#     if t[0] != t[1]:
#         if t in frequency_dict:
#             frequency_dict[t] += 1
#         else:
#             frequency_dict[t] = 1

# # Sort the dictionary by frequency in descending order and get the top ten tuples
# number_of_top_freq = 50
# top_ten_tuples = sorted(frequency_dict.items(), key=lambda x: x[1], reverse=True)[:number_of_top_freq]

# # Print the top ten tuples and their frequencies
# for t, freq in top_ten_tuples:
#     print(f'Tuple: {t}, Actual Class: {test_dataset.classes[t[0]]} Predicted Class: {test_dataset.classes[t[1]]}, Frequency: {freq}')
import numpy as np
import sklearn.metrics

"""
Python compute equal error rate (eer)
ONLY tested on binary classification

:param label: ground-truth label, should be a 1-d list or np.array, each element represents the ground-truth label of one sample
:param pred: model prediction, should be a 1-d list or np.array, each element represents the model prediction of one sample
:param positive_label: the class that is viewed as positive class when computing EER
:return: equal error rate (EER)
"""
def compute_eer(act_labels, pred_labels, positive_label=1):
    # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
    fpr, tpr, threshold = sklearn.metrics.roc_curve(act_labels, pred_labels)
    fnr = 1 - tpr

    # the threshold of fnr == fpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2
    return eer

# Call the function to compute EER
eer_result = compute_eer(act_labels, pred_labels)

# Print the result
print("Equal Error Rate:", eer_result)
Equal Error Rate: 0.10144976964709038
# from sklearn.metrics import roc_curve

# fpr, tpr, thresholds = roc_curve(pred_labels, actual_labels)

# # Calculate the Equal Error Rate (EER)
# eer = np.interp(0.5, fpr, 1 - tpr)

# print("Equal Error Rate (EER): {:.2f}%".format(eer * 100))
from sklearn.metrics import f1_score, jaccard_score, matthews_corrcoef, hamming_loss,accuracy_score

f1 = f1_score(act_labels, pred_labels, average='weighted')
jaccard = jaccard_score(act_labels, pred_labels, average='weighted')
mcc = matthews_corrcoef(act_labels, pred_labels)
hloss = hamming_loss(act_labels, pred_labels)

print(f1)
print(jaccard)
print(mcc)
print(hloss)
0.899335705812955
0.8171566709370368
0.8021958504043823
0.10034527406128614
# torch.save(model_whisp, r"/home/speechlab/Desktop/Siddharth/ASRUU/ModelsResults/Base/Run1_ep63_lr_03_01_003_drp_02.pt")
# train_feat_save_path="/home/speechlab/Desktop/Siddharth/ASRUU/hubert features/train"
# def pad(input_tensor):
#     if(input_tensor.shape[1]<400):
#         desired_dimensions = (1, 400, 768)
#         padding_amounts = (0,0,0,desired_dimensions[1] - input_tensor.shape[1])
#         padded_tensor = F.pad(input_tensor, padding_amounts, mode='constant',value = 0)
#     else:
#         padded_tensor = input_tensor[-1, 0:400 , :]
#     return padded_tensor
# import os
# import torch
# subfolders = [f.path for f in os.scandir(train_feat_save_path) if f.is_dir()]
# i=0
# mx=[]
# padded_data = []
# for subfolder in subfolders:
#     i=i+1
#     subfolder_name = os.path.basename(subfolder)
#     subfolder_path = os.path.join(train_feat_save_path, subfolder_name)

#     for file in os.listdir(subfolder_path):
      
#         file_path = os.path.join(subfolder_path, file)
#         data = torch.load(file_path, map_location='cpu').detach()
#         mx.append(data.shape[1])
#     #     data = data[-1, 0:400 , :]
#         print(data.shape)
#         break
#     break   
# import numpy as np
# x=np.array(mx)
# print(len(np.unique(x)))
# print(len(x))
# plt.hist(mx, bins=361)
# plt.show()
# import torch
# import torch.nn.functional as F

# # Assuming you have a tensor with dimensions [1, 137, 768]
# input_tensor = torch.randn(1, 137, 768)

# # Define the desired output dimensions
# desired_dimensions = (1, 400, 768)

# # Compute the padding amounts for each dimension
# # padding_amounts = [(0, desired_dimensions[i + 1] - input_tensor.shape[i + 1]) for i in range(len(desired_dimensions) - 1)]
# padding_amounts = (0,0,0,desired_dimensions[1] - input_tensor.shape[1])
# # Pad the tensor
# padded_tensor = F.pad(input_tensor, padding_amounts, mode='constant',value = 0)

# # Verify the new dimensions
# print(padded_tensor.shape)