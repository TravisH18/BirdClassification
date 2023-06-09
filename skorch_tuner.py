import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import creat_search_run, save_best_hyperparam
from skorch import NeuralNetClassifier
#from skorch.dataset import CVSplit
from sklearn.model_selection import GridSearchCV
from BirdNets import BirdNetComplexV1, BirdNetComplexV2, BirdNetComplexV3
from BirdData import BirdDataset

if __name__ == '__main__':
    # Create hyperparam search folder.
    search_folder = creat_search_run()
    # Learning parameters. 
    lr = 0.001
    epochs = 20
    #device='cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    print(device)

    # Create an instance of your custom CNN
    #model = BirdNetComplexV3()
    MODEL_PATH =  './BirdNetComplexV1_weights.pth'
    model = torch.load(MODEL_PATH)
    print(model)

    # Define the hyperparameters to tune and their search space
    param_grid = {
        'lr': [0.0001, 0.001, 0.01, 0.1],
        'optimizer__momentum': [0.0, 0.3, 0.6, 0.9],
        'batch_size': [16, 32, 64, 128],
        'max_epochs': [20, 25, 30, 40],
        #'module__dropout_rate': [0.2, 0.4, 0.6, 0.8]
    }

    # Wrap your model in a skorch NeuralNetClassifier
    net = NeuralNetClassifier(
        module=model,
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.SGD,
        # max_epochs=10,
        # lr=lr,
        # batch_size=32,
        verbose=False,
        train_split=None,
        warm_start=True,
        #device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    BATCH_SIZE = 32
    DATA_DIR = './data/'

    # Create your custom dataset and dataloaders
    # Create Training Dataset
    train_ds = BirdDataset(os.path.join(DATA_DIR), os.path.join(DATA_DIR, 'birds.csv'), split='train', transform=True)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True) # num_workers=2,
    print(len(train_dl))
    # Create Validation Dataset
    val_ds = BirdDataset(os.path.join(DATA_DIR), os.path.join(DATA_DIR, 'birds.csv'), split='valid')
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True) # num_workers=2,
    print(len(val_dl))
    # Create a GridSearchCV object
    #gs = GridSearchCV(net, param_grid, scoring='accuracy', cv=3, refit=False)
    gs = GridSearchCV(
            net, param_grid, refit=False, scoring='accuracy', verbose=1, cv=2, n_jobs=-1
        )

    # Fit the GridSearchCV object on your data
    # gs.fit([train_dl], y=None, iterator_valid=[val_dl])

    # #gs.fit(train_ds.df, y=None, iterator_valid=val_ds.df)

    # # Print the best parameters and the best score
    # print("Best parameters:", gs.best_params_)
    # print("Best score:", gs.best_score_)

    counter = 0
    # Run each fit for 2 batches. So, if we have `n` fits, then it will
    # actually for `n*2` times. We have 672 fits, so total, 
    # 672 * 2 = 1344 runs.
    search_batches = 2
    """
    This will run `n` (`n` is calculated from `params`) number of fits 
    on each batch of data, so be careful.
    If you want to run the `n` number of fits just once, 
    that is, on one batch of data,
    add `break` after this line:
        `outputs = gs.fit(image, labels)`
    Note: This will take a lot of time to run
    """
    for image, labels in train_dl:
        counter += 1
        # image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        outputs = gs.fit(image, labels)
        # GridSearch for `search_batches` number of times.
        if counter == search_batches:
            break
    print('SEARCH COMPLETE')
    print("best score: {:.3f}, best params: {}".format(gs.best_score_, gs.best_params_))
    save_best_hyperparam(gs.best_score_, f"./outputs/{search_folder}/best_param_ComplexV1.yml")
    save_best_hyperparam(gs.best_params_, f"./outputs/{search_folder}/best_param_ComplexV1.yml")
