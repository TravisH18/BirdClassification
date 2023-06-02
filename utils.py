# https://debuggercafe.com/hyperparameter-search-with-pytorch-and-skorch/

import matplotlib
import matplotlib.pyplot as plt
import glob as glob
import os

matplotlib.style.use('ggplot')

def save_plots(
    train_acc, valid_acc, train_loss, valid_loss, 
    acc_plot_path, loss_plot_path
):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # Accuracy plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(acc_plot_path)
    
    # Loss plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(loss_plot_path)

def save_hyperparam(text, path):
    """
    Function to save hyperparameters in a `.yml` file.

    :param text: The hyperparameters dictionary.
    :param path: Path to save the hyperparmeters.
    """
    with open(path, 'w') as f:
        keys = list(text.keys())
        for key in keys:
            f.writelines(f"{key}: {text[key]}\n")

def create_run():
    """
    Function to create `run_<num>` folders in the `outputs` folder for each run.
    """
    num_run_dirs = len(glob.glob('../outputs/run_*'))
    run_dir = f"../outputs/run_{num_run_dirs+1}"
    os.makedirs(run_dir)
    return run_dir

def creat_search_run():
    """
    Function to save the Grid Search results.
    """
    num_search_dirs = len(glob.glob('../outputs/search_*'))
    search_dirs = f"../outputs/search_{num_search_dirs+1}"
    os.makedirs(search_dirs)
    return search_dirs

def save_best_hyperparam(text, path):
    """
    Function to save best hyperparameters in a `.yml` file.
    :param text: The hyperparameters dictionary.
    :param path: Path to save the hyperparmeters.
    """
    with open(path, 'a') as f:
        f.write(f"{str(text)}\n")