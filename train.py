
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
import math
import matplotlib as plt

#from torchsummary import summary

from BirdNets import BirdNetComplexV1, BirdNetComplexV2, BirdNetComplexV3
from BirdData import BirdDataset

import logging
import re
import warnings

logging.captureWarnings(True)

warnings.filterwarnings("always", category=UserWarning, module=r'^{0}\.'.format(re.escape(__name__)))


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)
    #set backends
    #print(f"CUDNN Version {torch.backends.cudnn.version()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if (torch.backends.cudnn.is_available()):
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    # define Model and Optimizer
    #MODEL_PATH = './models/birdnetcomplexV2_model.pth'
    model = BirdNetComplexV3(dropout = 0.5).to(device)
    #model.load_state_dict(torch.load(MODEL_PATH))
    #model = torch.load(MODEL_PATH)
    #summary(model, input_size=(3, 244, 244))
    # print(model)
    # print(device)
    
    # model = model.to(device)

    LR = 0.001
    #opt = optim.Adam(model.parameters(), lr=LR, weight_decay=0.005)
    opt = optim.SGD(model.parameters(), lr=LR, weight_decay=0.005, momentum=0.9)
    #opt = optim.SGD(model.parameters, momentum=0.9, weight_decay=0.005)
    loss_fn = nn.CrossEntropyLoss()

    
    # Model Training History
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    # Dataset Parameters
    BATCH_SIZE = 32
    DATA_DIR = './data/'
    PATH = 'birdnetcomplexV3_2_weights.pth'

    # Create Training Dataset
    train_ds = BirdDataset(os.path.join(DATA_DIR), os.path.join(DATA_DIR, 'birds.csv'), split='train', transform=True)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True) # num_workers=2,

    # Create Validation Dataset
    val_ds = BirdDataset(os.path.join(DATA_DIR), os.path.join(DATA_DIR, 'birds.csv'), split='valid')
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True) # num_workers=2,

    # Calcluate # of batches
    train_steps = len(train_dl.dataset) // BATCH_SIZE
    val_steps = len(val_dl.dataset) // BATCH_SIZE

    # Define Training Hyperparameters
    EPOCHS = 100
    best_val_acc = 0

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

        if val_tp_tn > best_val_acc:
            best_val_acc = val_tp_tn
            torch.save(model.state_dict(), 'current_best_state_V3_2.pth')


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

    #torch.save(model.state_dict(), PATH)
    torch.save(model, PATH)