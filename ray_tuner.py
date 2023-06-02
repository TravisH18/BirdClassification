from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import ray
from ray import tune
from ray.tune import CLIReporter, TuneConfig
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
from ray.air.checkpoint import Checkpoint
from ray.tune.search.optuna import OptunaSearch
from ray._private.runtime_env.conda import get_uri as get_conda_uri
from ray.runtime_env import RuntimeEnv
from torch.utils.data import DataLoader
from BirdNets import BirdNetComplexV1, BirdNetComplexV2, BirdNetComplexV3
from BirdData import BirdDataset

# https://docs.ray.io/en/latest/tune/examples/includes/pbt_convnet_function_example.html


def train_birdnet(config, checkpoint_dir = None):
    DATA_DIR = os.environ.get('DATA_DIR') # './data/'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = BirdNetComplexV3(dropout=config["dropout"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=config["momentum"], weight_decay=config["weight_decay"])
    # To restore a checkpoint, use `session.get_checkpoint()`.
    loaded_checkpoint = session.get_checkpoint()
    if loaded_checkpoint:
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
           model_state, optimizer_state = torch.load(os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    train_ds = BirdDataset(os.path.join(DATA_DIR), os.path.join(DATA_DIR, 'birds.csv'), split='train', transform=True)
    train_dl = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=True) # num_workers=2,

    val_ds = BirdDataset(os.path.join(DATA_DIR), os.path.join(DATA_DIR, 'birds.csv'), split='valid')
    val_dl = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=True) # num_workers=2,

    train_steps = len(train_dl.dataset) // config["batch_size"]
    val_steps = len(val_dl.dataset) // config["batch_size"]

    for epoch in range(10):
        print(epoch)
        total_train_loss = 0
        total_val_loss = 0

        train_tp_tn = 0
        val_tp_tn = 0
        

        # Inner loop for each epoch
        for x, y in tqdm(train_dl):
            # Load the batch
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # Predict classes of batch
            pred = net(x)
            
            # Calculate loss
            loss = criterion(pred, y)

            # Update parameters
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # Update metrics
            total_train_loss += loss
            train_tp_tn += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Evaluate on validation partition
        with torch.no_grad():
            net.eval()
            
            # Loop over validation dataset
            for x, y in tqdm(val_dl):
                # Load the batch
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

                # Predict val labels
                pred = net(x)
                
                # Update metrics
                total_val_loss += criterion(pred, y)
                val_tp_tn += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and can be accessed through `session.get_checkpoint()`
        # API in future iterations.
        os.makedirs("my_model", exist_ok=True)
        torch.save(
            (net.state_dict(), optimizer.state_dict()), "my_model/checkpoint.pt")
        checkpoint = Checkpoint.from_directory("my_model")
        session.report({"loss": (total_val_loss / val_steps), "accuracy": val_tp_tn / len(val_dl.dataset)}, checkpoint=checkpoint)
    print("Finished Training")

def test_best_model(best_result):
    DATA_DIR = os.environ.get('DATA_DIR') # './data/'
    best_trained_model = BirdNetComplexV3(best_result.config["dropout"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_trained_model.to(device)

    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")

    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)
    test_ds = BirdDataset(os.path.join(DATA_DIR, 'test'), os.path.join(DATA_DIR, 'sample_solution.csv'), split='test')
    test_dl = DataLoader(test_ds, batch_size=best_result.config["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_dl:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = best_trained_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    print("Best trial test set accuracy: {}".format(correct / total))
if __name__ == '__main__':
    cwd = os.getcwd()
    print(cwd)
    #runtime_env = {"conda": 'gpu-training', "working_dir": cwd, }
    ENV_VARIABLES = {"DATA_DIR": "./data/", "CSV": "./data/birds.csv"}
    runtime_env = RuntimeEnv(conda='gpu-training', env_vars=ENV_VARIABLES, working_dir='.')
    ray.init(runtime_env=runtime_env, num_gpus=1)
    # ray_config = tune.RayConfig(
    #     num_workers=4,
    #     local_mode=False,  # Set to False to use multiple processes
    #     env={"CONDA_DEFAULT_ENV": "gpu-training"},  # Set the necessary environment variables
    #     local_dir=cwd
    # )
    #ray.init(local_mode=True)
    #ray.init(config=ray_config)
    config = {
        "lr" : tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([8, 16, 32, 64]),
        "dropout": tune.uniform(0.1, 0.9),
        "momentum": tune.uniform(0.1, 0.9), 
        "weight_decay": tune.loguniform(1e-5, 1e-2)
    }
    checkpoint_dir = './my_model'
    num_samples = 10
    max_epochs = 25
    #scheduler = ASHAScheduler(max_t=max_epochs, grace_period=1, reduction_factor=2)
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_epochs,
        grace_period=1,
        reduction_factor=2
    )
    tuner = tune.Tuner(
        train_birdnet,
        #tune.with_resources(tune.with_parameters(train_birdnet), resources={"gpu": 1}),
        #tune_config=TuneConfig(metric="loss", mode="min", scheduler=scheduler, num_samples=num_samples, chdir_to_trial_dir=False),
        tune_config=TuneConfig(scheduler=scheduler, num_samples=num_samples, chdir_to_trial_dir=False),
        param_space=config
    )
    results = tuner.fit()
    
    best_result = results.get_best_result("loss", "max") #maybe change to get_results

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))

    test_best_model(best_result)
