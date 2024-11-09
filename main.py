from train import nerflash
from config import *


for param_dict in [
    param_dict_poster, 
    param_dict_lens_stage, 
    param_dict_shelf, 
    param_dict_ps5, 
    param_dict_outdoor, 
    param_dict_office
    ]:

    nerflash(param_dict)

'''
Example Usage:

python main.py --exp_name new_experiment --save_dir experiments --gif_save_dir gif_demo

'''