import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import cv2
import os
import warnings

from BirdNets import BirdNetComplexV1

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

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    #change these to 64, birdnetcomplex
    BATCH_SIZE = 32
    DATA_DIR = './data/'
    PATH = 'birdnetV2_weights.pth'
    test_ds = BirdDataset(os.path.join(DATA_DIR, 'test'), os.path.join(DATA_DIR, 'sample_solution.csv'), split='test')
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    df = pd.read_csv(os.path.join(DATA_DIR, 'sample_solution.csv'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = BirdNetComplexV1()
    # model.load_state_dict(torch.load(PATH))
    model = torch.load(PATH)
    model.to(device)

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