import pandas as pd
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.nn import Module, Conv2d, Linear, MaxPool2d, ReLU, LogSoftmax, CrossEntropyLoss
import torchvision.transforms as T
# from torchvision import models

from tqdm import tqdm

import os

from BirdNets import BirdNetComplex, BirdNetSimple

class BirdDataset(Dataset):
    N_CLASSES = 450
    """Constructor Function for Class - Loads parameters"""
    def __init__(self, data_dir, csv_file, split='train', transform=False):
        # Load dataset and label file locations to variable 
        self.data_dir = data_dir
        self.transform = transform
        
        df = pd.read_csv(csv_file)
        if split=='test':
            df['filepaths'] = df['image_id'].astype(str) + '.jpg' # self.data_dir + '/' + 
            
            self.df = df
        else: 
            self.df = df.loc[df['data_set'] == split, ['image_id', 'class_id', 'filepaths']]
            self.df = self.df.reset_index()
    
    """Returns the number of records in the dataset"""
    def __len__(self):
        return len(self.df)
    
    """Retrieve a single record from the dataset"""
    def __getitem__(self, idx):
        # Open and Read Image at id `idx`
        
        img = cv2.imread(os.path.join(self.data_dir, self.df['filepaths'][idx]))
        
        # Converts image to Tensor
        img = torch.tensor(img).permute(2,0,1).float()
        transform = T.Compose([
            T.Resize((224,224)),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Image is currently 3x224x224
        # transform1 = T.Compose([
        #     T.Resize((224,224)), # (it is recommended to preserve this transformation unless you plan on modifying the CNN architecture)
        #     T.RandomRotation(15),
        #     T.RandomPerspective(0.3, 0.5),
        #     # T.ToTensor(),
        #     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])
        # transform2 = T.Compose([
        #     T.Resize((224,224)),
        #     # T.ToTensor(),
        #     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])
        # if self.transform:
        #     img = transform1(img)
        # else:
        #     img = transform2(img)
        img = transform(img)
        # Reads label
        label = int(self.df['class_id'][idx])
        
        return img, label


# Dataset Parameters
BATCH_SIZE = 32
DATA_DIR = './data/'
PATH = 'current_model_weights.pth'

# Create Training Dataset
train_ds = BirdDataset(os.path.join(DATA_DIR), os.path.join(DATA_DIR, 'birds.csv'), split='train', transform=True)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# Create Training Dataset
val_ds = BirdDataset(os.path.join(DATA_DIR), os.path.join(DATA_DIR, 'birds.csv'), split='valid')
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True)

# Calcluate # of batches
train_steps = len(train_dl.dataset) // BATCH_SIZE
val_steps = len(val_dl.dataset) // BATCH_SIZE
val_ds.df

# Make sure you are using the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#model = models.resnet18() # weights='IMAGENET1K_V1' or weights=models.ResNet18_Weights.IMAGENET1K_V1
model = BirdNetComplex()
# model = torch.load(PATH) #load entire model !
# model.load_state_dict(torch.load(PATH))

# model = models.resnet50()
# for param in model.parameters():
#     param.require_grad = False
    
# n_features = model.fc.in_features
# model.fc = Linear(n_features, BirdDataset.N_CLASSES)
model = model.to(device)

# Define Training Hyperparameters
LR = 0.01
EPOCHS = 25

# Model Training History
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

# # Define the optimizer and loss function, both used for training
opt = Adam(model.parameters(), lr=LR)
loss_fn = CrossEntropyLoss()

# Prepare the model to train
model.train()

# Outer loop for each epoch
for epoch in range(EPOCHS):
    total_train_loss = 0
    total_val_loss = 0

    train_tp_tn = 0
    val_tp_tn = 0

    # Inner loop for each epoch
    for x, y in tqdm(train_dl):
        # Load the batch
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        # Predict classes of batch
        pred = model(x)
        
        # Calculate loss
        loss = loss_fn(pred, y)

        # Update parameters
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # Update metrics
        total_train_loss += loss
        train_tp_tn += (pred.argmax(1) == y).type(torch.float).sum().item()

    # Evaluate on validation partition
    with torch.no_grad():
        model.eval()
        
        # Loop over validation dataset
        for x, y in tqdm(val_dl):
            # Load the batch
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # Predict val labels
            pred = model(x)
            
            # Update metrics
            total_val_loss += loss_fn(pred, y)
            val_tp_tn += (pred.argmax(1) == y).type(torch.float).sum().item()

    # Calculate Training Final Metrics
    avgTrainLoss = total_train_loss / train_steps
    avgValLoss = total_val_loss / val_steps
    trainCorrect = train_tp_tn / len(train_dl.dataset)
    valCorrect = val_tp_tn / len(val_dl.dataset)
    
    # Update History
    history["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
    history["train_acc"].append(trainCorrect)
    history["val_loss"].append(avgValLoss.cpu().detach().numpy())
    history["val_acc"].append(valCorrect)
    
    # Log Metrics
    print("[INFO] EPOCH: {}/{}".format(epoch + 1, EPOCHS))
    print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(avgTrainLoss, trainCorrect))
    print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(avgValLoss, valCorrect))

# import matplotlib.pyplot as plt

# Plot training history
# plt.figure()
# plt.plot(history["train_loss"], label="train_loss")
# plt.plot(history["val_loss"], label="val_loss")
# plt.plot(history["train_acc"], label="train_acc")
# plt.plot(history["val_acc"], label="val_acc")
# plt.title("Training Loss and Accuracy on Dataset")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend(loc="lower left")
# plt.show()

torch.save(model.state_dict(), PATH)

# Create Dataset and DataLoader for the test partition 

# model.load_state_dict(torch.load(PATH))
# model.to(device)
test_ds = BirdDataset(os.path.join(DATA_DIR, 'test'), os.path.join(DATA_DIR, 'sample_solution.csv'), split='test')

test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

df = pd.read_csv(os.path.join(DATA_DIR, 'sample_solution.csv'))
# print(df.head)

# Disable Gradients to reduce memory usage and speed up computations 
with torch.no_grad():
    # Prepare model for inference
    model.eval()
        
    # Store predictions in a list
    preds = []
    # Iterate through the test dataset using the test dataloader
    for x, _ in tqdm(test_dl):
        # Move input to device
        x = x.to(device)
        
        # Predict labels of batch
        pred = model(x)
        
        # Select highest probability class as the predicted class
        preds.extend(pred.argmax(axis=1).cpu().numpy())
    
    # Convert to Pandas DataFrame and output predictions .csv file
    for i, p in enumerate(preds):
        df['class_id'][i] = p
    df.to_csv('submission.csv', index=False)

df = pd.read_csv('submission.csv')
df