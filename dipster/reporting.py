import time
import wandb
import os 

import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from dipster import util

class Report():
    def __init__(self, params):
        # Set Up reporting 
        self.hypertrain = params.hypertrain
        self._tables = {}
        if not params.hypertrain:
            os.environ["WANDB_DIR"] = params.wandb_local_dir
            run = wandb.init(project=params.wandb_project)
            self.name = run.name
            self._log = {}
        else:
            
            self.name = 'report'
        self.start = time.time()

    def quantify(self, rec, ref):
        rec = util.torch_to_np(rec)
        ref = util.torch_to_np(ref)

        psnr_val = psnr(rec, ref, data_range= ref.max()-ref.min())
        ssim_val = ssim(rec, ref, data_range= ref.max()-ref.min())

        return psnr_val, ssim_val

    def update(self, step, rec, ref, title):
        psnr_val, ssim_val = self.quantify(rec, ref)
        self._update_values(title, psnrs = [step, psnr_val], ssims = [step, ssim_val])

        rec = (rec - rec.min())/(rec.max()-rec.min())
        ref = (ref - ref.min())/(ref.max()-ref.min())

        self._update_images(title, [ref, rec])


    def _update_values(self, title, **kwargs):
        if not self.hypertrain:
            for key, value in kwargs.items():
                dict_key = title + "_" + key
                if not dict_key in self._tables:
                    self._tables[dict_key] = [value]
                else:
                    self._tables[dict_key].append(value)
            
            self._log[dict_key] = wandb.plot.line(wandb.Table(data=self._tables[dict_key], columns = ["steps", dict_key]), "steps", dict_key, title=dict_key)
        else:
            for key, value in kwargs.items():
                dict_key = title + "_" + key
                if not dict_key in self._tables:
                    self._tables[dict_key] = []
                self._tables[dict_key] = value[1]

    def _update_images(self, title,  value):
        if not self.hypertrain:
            self._log[title] = [wandb.Image(value[0], caption='reference'), wandb.Image(value[1], caption='reconstruction')]
        else:
            self._tables[title +'_image'] = value

    def publish(self):
        wandb.log(self._log)