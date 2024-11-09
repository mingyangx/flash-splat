from termcolor import colored
import os
import psutil
import torch
import numpy as np
import random
import os, random
from termcolor import colored
import torch
import warnings, logging
import numpy as np
import torch, time
import logging, warnings, wandb
from pynvml import *
import wandb


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)



def gpuinfo(text=None, use_wandb=False, wandb_iter=None):
    nvmlInit()
    for i in range(torch.cuda.device_count()):
        h = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(h)
        if not use_wandb:
            print(colored(f'{text} :GPU {i}, total {info.total/1024**3}G, free {info.free/1024**3}G, used: {info.used/1024**3}G', 'red'))
        
        if use_wandb:
            wandb.log({f'GPU Memory Used No. {i} (GB)': info.used/1024**3}, step=wandb_iter)


def info(vector=None, name='', precision=4):
    """
    check info
    :param name: name
    :param vector: torch tensor or numpy array or list of tensor/np array
    """
    if torch.is_tensor(vector):
        if torch.is_complex(vector):
            print(colored(name, 'red') + f' tensor size: {vector.size()}, mean: {torch.mean(vector).item():.{precision}f}, dtype: {vector.dtype}, on Cuda: {vector.is_cuda}')
        else:
            try:
                print(colored(name, 'red') + f' tensor size: {vector.size()}, min: {torch.min(vector).item():.4f},  max: {torch.max(vector).item():.4f}, mean: {torch.mean(vector).item():.{precision}f}, dtype: {vector.dtype}, on Cuda: {vector.is_cuda}')
            except:
                print(colored(name, 'red') + f' tensor size: {vector.size()}, min: {torch.min(vector).item():.4f},  max: {torch.max(vector).item():.4f}, dtype: {vector.dtype}, on Cuda: {vector.is_cuda}')
    elif isinstance(vector, np.ndarray):
        try:
            print(colored(name, 'red') + f' numpy size: {vector.shape}, min: {np.min(vector):.4f},  max: {np.max(vector):.4f}, mean: {np.mean(vector):.4f}, dtype: {vector.dtype}')
        except:
            print(colored(name, 'red') + f' numpy size: {vector.shape}, min: {np.min(vector):.4f},  max: {np.max(vector):.4f}, dtype: {vector.dtype}')
    elif isinstance(vector, list):
        info(vector[0], f'{name} list of length: {len(vector)}, {name}[0]')
    else:
        print(colored(name, 'red') + 'Neither a torch tensor, nor a numpy array, nor a list of tensor/np array.' + f' type{type(vector)}')


def init_env(use_wandb=True):
    seed_torch(0)
    warnings.filterwarnings("ignore")

    if use_wandb:
        logger = logging.getLogger("wandb")
        logger.setLevel(logging.ERROR)


def cpuinfo(text=None):
    
    print(colored(f'{text} : CPU Percent {psutil.cpu_percent()}%, RAM Percent {psutil.virtual_memory().percent}%, Disk Percent {psutil.disk_usage(os.sep).percent}%', 'red'))
    print(colored(f'{text} : RAM Total {psutil.virtual_memory().total / (1024.0 ** 3)}, \
                  USED {psutil.virtual_memory().used / (1024.0 ** 3)}, \
                  Free {psutil.virtual_memory().free / (1024.0 ** 3)}, \
                  Available {psutil.virtual_memory().available / (1024.0 ** 3)},\
                  Percent {psutil.virtual_memory().percent}%,', 'red'))



def normalize(x):
    '''
    normalize the max value to 1, and min value to 0
    '''
    return (x - torch.min(x)) / (torch.max(x) - torch.min(x))