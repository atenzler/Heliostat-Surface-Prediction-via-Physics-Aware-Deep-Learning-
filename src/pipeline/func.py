"""
func.py

Helper functions for data preparation, logging, and visualization
used in the DeepLARTS training pipeline.

Includes:
- Dataset wrapper (`pack_dataset`)
- Data preparation and splitting
- Folder structure creation (local & cluster versions)
- Model summary logging
- Loss plotting and example visualization
- Simple metrics (MAE)

Original base: Jing Sun & Tiexing Wang
Adapted and extended for heliostat surface prediction in this thesis.
"""
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import argparse
import torch.nn as nn
import logging
import math
import shutil

class pack_dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, sun_positions, heliostat_positions, transform=None):
        self.inputs  = inputs
        self.sun_positions = sun_positions
        self.heliostat_positions = heliostat_positions
        self.transform = transform

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index):
        inputs  = torch.Tensor(self.inputs[index]).float()
        sun_positions = torch.Tensor(self.sun_positions).float()
        heliostat_positions = torch.Tensor(self.heliostat_positions).float()
        if self.transform:
            inputs, sun_positions, heliostat_positions = self.transform(inputs, sun_positions, heliostat_positions)
        return inputs, sun_positions, heliostat_positions

def prepare_data(inputs, sun_positions, heliostat_positions):
    print(inputs.shape)
    print("inputs shape")
    print(inputs.size)
    inputs  = inputs.reshape(inputs.shape[0], 4, 500, 540) #bisschen willkuerlich gewahlt sodass code passt
    #kernels = kernels.reshape(kernels.shape[0], 1, *kernels.shape[1:])
    bag = pack_dataset(inputs, sun_positions, heliostat_positions, transform=None)
    return bag

def split_dataset(data, train_percentage=0.8):
    len_train = int(round(len(data) * train_percentage))
    len_valid = len(data) - len_train
    train_data, valid_data = torch.utils.data.random_split(data, [len_train, len_valid])
    return train_data, valid_data

def create_folder_struct_cluster(suffix="", base_logdir=None):
    from datetime import datetime
    import os, shutil

    if base_logdir is None:
        base_logdir = "/mnt"  # fallback default

    # Add date subfolder inside the specified base_logdir
    date = datetime.now().strftime('%Y-%m-%d')
    base_dir = os.path.join(base_logdir, date)

    # Use suffix (e.g., trial number) to separate each run
    folder_name = f"run_{suffix}" if suffix else "run_default"
    full_path = os.path.join(base_dir, folder_name)

    # Clean if exists
    if os.path.exists(full_path):
        print(f'⚠ Overwriting existing folder: "{full_path}"')
        shutil.rmtree(full_path)

    os.makedirs(os.path.join(full_path, "tensorboard_logs"), exist_ok=True)
    print(f'✅ Folder created: "{full_path}"')

    return full_path

def create_folder_struct(suffix="", base_logdir=None):
    """
    Create a folder structure for storing output.
    If a folder already exists for the day and suffix, it will be cleared and recreated.

    Args:
        suffix (str): Optional experiment name to make the folder unique.
        base_logdir (str): Root directory for logs. Defaults to local path for Windows,
                           can be overridden (e.g. for /mnt on cluster).

    Returns:
        str: Path to the created directory.
    """
    # Default base path
    if base_logdir is None:
        base_logdir = '/mnt/tensorboard_logs_5000'

        # Use date-based folder
    date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    base_dir = os.path.join(base_logdir, date)

    # Add suffix-based experiment folder
    folder_name = f"run_{suffix}" if suffix else "run_default"
    full_path = os.path.join(base_dir, folder_name)

    # Clean up existing folder
    if os.path.exists(full_path):
        print(f'⚠ Overwriting existing folder: "{full_path}"')
        shutil.rmtree(full_path)

    # Create structure
    os.makedirs(os.path.join(full_path, "tensorboard_logs"), exist_ok=True)
    print(f'✅ Folder created: "{full_path}"')

    return full_path


def write_summary(model, output_dir):
    """
    Write model summary to a text file.
    """
    filepath = os.path.join(output_dir, 'model_print.txt')
    with open(filepath, 'a') as f:
        f.write(f'Model architecture: {model}\n')
        for param_name, param_tensor in model.state_dict().items():
            # Move tensor to CPU and convert to NumPy array
            param_array = param_tensor.cpu().numpy()  
            f.write(f'{param_name}\n{np.array(param_array)}\n')

    print('Model summary written to file!')
    return

def self_logging(model, device, pre_dirs, dirs, which_net,
                        nbr_train_pairs, nbr_valid_pairs,
                        in_channels, out_channels, kernel_size, features, nbr,
                        criterion, optimizer_type, learning_rate, weight_decay,
                        batch_size, max_epochs, patience_epochs, patience_loss,
                        sun_positions, heliostat_positions,
                        heliostat_aim_point):
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    print(f'Log file will be saved at {dirs}/loss_record.log')
    logging.basicConfig(level=logging.INFO, filename=dirs+'/loss_record.log', filemode='a',
                        format='%(asctime)s   %(levelname)s   %(message)s')
    
    parser = argparse.ArgumentParser(description='Your description here')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    args   = parser.parse_args()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        assert args.batch_size % torch.cuda.device_count() == 0, 'batch-size %g not multiple of GPU count %g' % (args.batch_size, torch.cuda.device_count())
        logging.info('DDP Mode')

    assert device != 'cpu', 'You are in CPU mode, please switch to GPU'
    model.to(device)
    
    # Log GPU information
    logging.info('Use GPU: %s' % (torch.cuda.is_available()))
    logging.info('Number of GPU: %d' % (torch.cuda.device_count()))
    if torch.cuda.is_available():
        logging.info('Device name: %s' % (torch.cuda.get_device_name()))
    else:
        logging.info('CUDA is not available.')
    
    logging.info('pre_dirs: %s' % (pre_dirs))
    logging.info('which_net: %s' % (which_net))
    logging.info('number of training pairs: %d' % (nbr_train_pairs))
    #logging.info('number of validation pairs: %d' % (nbr_valid_pairs))
    logging.info('in_channels: %d' % (in_channels))
    logging.info('out_channels: %d' % (out_channels))
    logging.info('kernel_size: %d' % (kernel_size))
    logging.info('features: %s' % (features))
    if which_net == 'UNet':
        logging.info('number of poolings: %s' % (nbr))
    elif which_net == 'ResNet':
        logging.info('number of blocks: %s' % (nbr))
    logging.info('criterion: %s' % (criterion))
    logging.info('optimizer_type: %s' % (optimizer_type))
    logging.info('initial_learning_rate: %f' % (learning_rate))
    logging.info('weight_decay: %f' % (weight_decay))
    logging.info('batch_size: %d' % (batch_size))
    logging.info('max_epochs: %d' % (max_epochs))
    logging.info('patience_epochs: %d' % (patience_epochs))
    logging.info('patience_loss: %f' % patience_loss)
    logging.info('sun_positions: %s' % (sun_positions))
    logging.info('heliostat_positions: %s' % (heliostat_positions))
    logging.info('heliostat_aim_point: %s' % (heliostat_aim_point))

def plot_loss(train_losses, valid_losses, dirs):
    """
    Plots the training and validation losses and saves the plot to a file.
    """

    # Convert to CPU + NumPy if they're tensors
    if torch.is_tensor(train_losses[0]):
        train_losses = [loss.detach().cpu().item() for loss in train_losses]
    if torch.is_tensor(valid_losses[0]):
        valid_losses = [loss.detach().cpu().item() for loss in valid_losses]


    epoch = np.arange(1, len(train_losses) + 1)
    plt.plot(epoch, train_losses, 'b-', label='Train Total Loss')
    plt.plot(epoch, valid_losses, 'r--', label='Validation Total Loss')
    
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    # Join the directory and filename using os.path.join()
    loss_path = os.path.join(dirs, 'loss_plot.png')

    # Add error handling in case the directory does not exist
    if not os.path.exists(dirs):
        os.makedirs(dirs)
        print(f"Created directory {dirs}")

    plt.savefig(loss_path, dpi=300)
    plt.close()

    print('Loss written and plotted')
    return

def mae(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred)).item()

def plot_example(h1,   magh1_modeled,
                 magin,  upwarded_dnn,
                 target, dnn_out,
                 dirs, first_example_num, total_num, interval):
    ## difference between input (from forward modeling) and upwarded_target (obtained by sending target into upward continuation function)
    #diff_in_uptarget = magin  - upwarded_target

    ## difference between input (obtained by sending target into upward continuation function) and magh1_modeled (from forward modeling)
    diff_in_modeled = magin  - magh1_modeled
    ## difference between input and upwarded_dnn (obtained by sending dnn output into upward continuation function)
    diff_in_updnn   = magin  - upwarded_dnn
    ## difference between target (ground truth) and dnn output
    diff_out        = target - dnn_out

    titles = ['h1'                     , 'magh1_modeled'  , 'diff (input - magh1_modeled)',
              'input (upwarded_target)', 'upwarded_dnn'   , 'diff (input - upwarded_dnn)',
              'target'                 , 'dnn'            , 'diff (target - dnn)']
    sets   = [h1                       , magh1_modeled    , diff_in_modeled,
              magin                    , upwarded_dnn     , diff_in_updnn,
              target                   , dnn_out          , diff_out]
    '''
    titles = ['h1'                   , 'upwarded_target', 'diff (input - upwarded_target)',
              'input (magh1_modeled)', 'upwarded_dnn'   , 'diff (input - upwarded_dnn)',
              'target'               , 'dnn'            , 'diff (target - dnn)']
    sets   = [h1                     , upwarded_target  , diff_in_uptarget,
              magin                  , upwarded_dnn     , diff_in_updnn,
              target                 , dnn_out          , diff_out]
    '''

    for n in range(first_example_num, first_example_num + total_num * interval, interval):
        vmin_input    = np.min(magh1_modeled[n])
        vmax_input    = np.max(magh1_modeled[n])
        vmin_target   = np.min([np.min(target[n]), np.min(dnn_out[n])])
        vmax_target   = np.max([np.max(target[n]), np.max(dnn_out[n])])
        vmin_diff_in  = np.min([np.min(diff_in_updnn[n]), np.min(diff_in_modeled[n])])
        vmax_diff_in  = np.max([np.max(diff_in_updnn[n]), np.max(diff_in_modeled[n])])
        vmin_diff_out = np.min(diff_out[n])
        vmax_diff_out = np.max(diff_out[n])

        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        for i, ax in enumerate(axes.flat):
            kwargs = {}
            if titles[i] in ['magh1_modeled', 'input (upwarded_target)', 'upwarded_dnn']:
                kwargs['vmin'] = vmin_input
                kwargs['vmax'] = vmax_input

            if titles[i] in ['target', 'dnn']:
                kwargs['vmin'] = vmin_target
                kwargs['vmax'] = vmax_target

            if titles[i] in ['diff (input - magh1_modeled)', 'diff (input - upwarded_dnn)']:
                kwargs['vmin'] = vmin_diff_in
                kwargs['vmax'] = vmax_diff_in

            if titles[i] in ['diff (target - dnn)']:
                kwargs['vmin'] = vmin_diff_out
                kwargs['vmax'] = vmax_diff_out

            im = ax.imshow(sets[i][n], cmap='rainbow', aspect='auto', origin='upper', **kwargs)
            ax.set_title(titles[i], size=17)
            ax.set_ylabel('Y axis', fontsize=17)
            ax.set_xlabel('X axis', fontsize=17)
            ax.tick_params(axis='x', which='major', labelsize=17, pad=2)
            ax.tick_params(axis='y', which='major', labelsize=17, pad=2)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            plt.colorbar(im)
        
        short_dirs = dirs[-16:] 
        info_title = f"{short_dirs} - example No.{n}"

        fig.suptitle(f"{info_title}\n", fontsize=18)
        plt.tight_layout()
        plt.savefig(f"{dirs}/example_plot_{n}.png", dpi=200)
        plt.close()

    print('Example plotted!')
