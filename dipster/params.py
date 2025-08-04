import os
import sys
import numpy as np
import torch 
import logging
class Params():
    def __init__(self, ts=None):
        # Data Parameters
        if ts is not None:
            self.proj_size: int = ts.data.shape[0]
            self.frames: int = ts.data.shape[2]
            self.noise_regularizer = 0.0001
            self.dev = 0
            self.seed: int = 0

            # Model Parameters
            self.opt_over: str = "net"
            self.latent_dim: int = 4
            self.hidden_dim: int = 512
            self.style_size: int = 8
            self.depth: int = 1
            self.Nr: int = 1
            self.input_nch: int = 1
            self.output_nch: int = 1
            self.need_bias: bool = False
            self.up_factor: int = 16
            self.upsample_mode: str = "nearest"

            # Training Parameters
            self.lr: float = 1e-3
            self.step_size: int = 2000
            self.gamma: float = 0.5
            self.batch_size: int = 1
            self.max_steps: int = 2000

            # Testing Parameters
            self.wandb_local_dir = "..\..\wandb"
            self.wandb_project = 'Default'
            self.wandb_name = 'Default'
            self.evaluate: bool = True
            self.eval_vol: bool = False
            self.save_period: int = 100
            self.hypertrain: bool = False
        
    def set_env(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.dev)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def to_dict(self):
        _dict = {}
        _dict['proj_size'] = self.proj_size
        _dict['frames'] = self.frames
        _dict['dev'] = self.dev
        _dict['seed'] = self.seed
        _dict['opt_over'] = self.opt_over
        _dict['latent_dim'] = self.latent_dim
        _dict['hidden_dim'] = self.hidden_dim
        _dict['style_size'] = self.style_size
        _dict['depth'] = self.depth
        _dict['Nr'] = self.Nr
        _dict['input_nch'] = self.input_nch
        _dict['output_nch'] = self.output_nch
        _dict['need_bias'] = self.need_bias
        _dict['up_factor'] = self.up_factor
        _dict['upsample_mode'] = self.upsample_mode
        _dict['lr'] = self.lr
        _dict['step_size'] = self.step_size
        _dict['gamma'] = self.gamma
        _dict['batch_size'] = self.batch_size
        _dict['max_steps'] = self.max_steps
        _dict['evaluate'] = self.evaluate
        _dict['eval_vol'] = self.eval_vol
        _dict['save_period'] = self.save_period
        _dict['wandb_local_dir'] = self.wandb_local_dir
        _dict['wandb_project'] = self.wandb_project
        _dict['wandb_name'] = self.wandb_name
        _dict['hypertrain'] = self.hypertrain
        return _dict

    def log(self):
        #set up logging in juypter notebook
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', stream=sys.stdout)

        dict = self.to_dict()
        for dict_key in dict:
            logging.info(f"{dict_key}: {dict[dict_key]}")


    @classmethod
    def from_dict(cls, state_dict):
        params = cls()
        params.proj_size = state_dict['proj_size']
        params.frames = state_dict['frames']
        params.dev = state_dict['dev']
        params.seed = state_dict['seed']
        params.opt_over = state_dict['opt_over']
        params.latent_dim = state_dict['latent_dim']
        params.hidden_dim = state_dict['hidden_dim']
        params.style_size = state_dict['style_size']
        params.depth = state_dict['depth']
        params.Nr = state_dict['Nr']
        params.input_nch = state_dict['input_nch']
        params.output_nch = state_dict['output_nch']
        params.need_bias = state_dict['need_bias']
        params.up_factor = state_dict['up_factor']
        params.upsample_mode = state_dict['upsample_mode']
        params.lr = state_dict['lr']
        params.step_size = state_dict['step_size']
        params.gamma = state_dict['gamma']
        params.batch_size = state_dict['batch_size']
        params.max_steps = state_dict['max_steps']
        params.evaluate = state_dict['evaluate']
        params.eval_vol = state_dict['eval_vol']
        params.save_period = state_dict['save_period']
        params.wandb_local_dir = state_dict['wandb_local_dir']
        params.wandb_project = state_dict['wandb_project']
        params.wandb_name = state_dict['wandb_name']
        params.hypertrain = state_dict['hypertrain']
        return params
