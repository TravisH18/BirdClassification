import os
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import cv2
import pandas as pd
import logging
import re
import warnings

logging.captureWarnings(True)

warnings.filterwarnings("always", category=UserWarning, module=r'^{0}\.'.format(re.escape(__name__)))

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
        #img = torch.tensor(img).permute(2,0,1).float()
        img = torch.from_numpy(img).permute(2,0,1).float()
        # transform = T.Compose([
        #     T.Resize((224,224)),
        #     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])

        # Image is currently 3x224x224
        transform1 = T.Compose([
            T.Resize((224,224)), # (it is recommended to preserve this transformation unless you plan on modifying the CNN architecture)
            T.RandomHorizontalFlip(0.3),
            #T.RandomPerspective(0.3, 0.3),
            #T.RandomRotation(15),
            #T.ColorJitter(brightness=(0.5,1.5), contrast=(1), saturation=(0.5,1.5), hue=(-0.1,0.1)),
            #T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        transform2 = T.Compose([
            T.Resize((224,224)),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            #T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        if self.transform:
            img = transform1(img)
        else:
            img = transform2(img)
        #img = transform(img)
        # Reads label
        label = int(self.df['class_id'][idx])
        
        return img, label