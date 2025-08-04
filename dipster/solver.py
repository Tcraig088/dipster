
import os
import sys
import time
from collections.abc import Iterable
import numpy as np
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia import losses as L
from ray import train, tune
from dipster import grad, nets, tomo, params, util
from dipster.reporting import Report

class Solver():
    def __init__(self, ts = None, fromdict = False):
        self.setup=False
        if not  fromdict:
            self.params = params.Params(ts)


    def eval(self, ref=None, save_period=100):
        self.params.evaluate = True
        if not ref is None:
            self.ref_vol = ref
            self.params.eval_vol = True
        self.params.save_period = save_period
        self.results = Report(self.params)
        self.params.wandb_name = self.results.name
        return self
    
    def _setfromconfig(self, config, params):
        self.params = params
        self.params.depth = config['depth']
        self.params.lr  = config['lr']
        self.params.gamma = config['gamma']
        self.params.batch_size = config['batch_size']

        self.params.style_size = int(np.sqrt(self.params.proj_size/config['option']))
        self.params.up_factor = int(np.sqrt(self.params.proj_size*config['option']))
        self.params.hypertrain = True

    def _setnet(self, config):
        self.params.depth = config['depth']
        self.params.lr  = config['lr']
        self.params.gamma = config['gamma']
        self.params.style_size = config['style_size']
        self.params.up_factor = config['up_factor']

    def _setup(self, ts= None):
        #working on the Assumption CUDA is always available
        self.params.dev = torch.device(self.params.dev) 
        self.params.set_env()
        self.params.log()

        self.grad = grad.CustomGradient(self.params.proj_size,self.params.batch_size,self.params.output_nch)
        self.loss_fn = L.GemanMcclureLoss(reduction="mean")

        self.net = nets.Net(self.params).to(self.params.dev)
        if ts is not None:
            self.manifold = nets.Manifold(ts.times[self.params.frames-1], self.params.proj_size, self.params.dev)
        else:
            self.manifold = nets.Manifold(self.params.frames, self.params.proj_size, self.params.dev)

        p = nets.get_params(self.net)         
        self.mapnet = nets.MappingNet(self.params).to(self.params.dev)
        p += self.mapnet.parameters()              

        self.optimizer = torch.optim.Adam(p, lr=self.params.lr)  
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.params.step_size, gamma=self.params.gamma)

        self.step = 0 
        self.setup = True 

    def train(self, ts):
        if self.setup == False:
            self._setup(ts)

        if len(ts.data.shape) < 4:
            ts.data = ts.data[:,:,:, None]
        while self.step < self.params.max_steps:

            self.optimizer.zero_grad()

            i_batched_frames = torch.randint(0, self.params.frames, (self.params.batch_size,)).to(self.params.dev)
            i_batched_depths = torch.randint(0, self.params.proj_size, (self.params.batch_size,)).to(self.params.dev)
            #check if step is in between 3800 and 4200
            out_set = self.reconstruct_slices(ts.angles[i_batched_frames], i_batched_depths, ts.times[i_batched_frames])
            angles = ts.angles[i_batched_frames]

            data_1d_ref = torch.zeros([self.params.proj_size, self.params.batch_size,1, self.params.output_nch]).to(self.params.dev)
            for i in range(self.params.batch_size):
                data_1d_ref[:,i,:,:] = ts.data[:,i_batched_depths[i], i_batched_frames[i],:][:,None,:]

            self.grad.X = out_set
            tv_array = torch.permute(out_set, [3,1,2,0])
            tv = L.total_variation(tv_array)
            data_1d_rec = self.grad(angles)

            
            # Backpropogate and update weights
            total_loss = self.loss_fn(data_1d_rec,data_1d_ref) + (tv*self.params.noise_regularizer)
            total_loss = total_loss.sum()
            total_loss.backward() 

            self.optimizer.step()
            self.scheduler.step()
            self.step += 1

            if self.params.evaluate==True:
                if self.step % self.params.save_period == 0: 
                    self.results._update_values('losses',times = [self.step, time.time()-self.results.start], losses = [self.step, total_loss])
                    self._evaluate(ts)

                        

    
    @torch.no_grad()             
    def _evaluate(self, ts): 
        rand = torch.randint(0, self.params.frames, (1,)).item()
        rec = np.zeros((self.params.proj_size, self.params.proj_size)) 
        rand = torch.randint(0, self.params.frames,(1,)).to(self.params.dev)
        for i in range(self.params.proj_size):
            val = tomo.fp(self.reconstruct_slices(ts.angles[rand], [i], ts.times[rand]),ts.angles[rand]).squeeze()
            rec[:,i] = util.torch_to_np(val)
        ref = ts.data[:,:,rand,0].squeeze()
        self.results.update(self.step, rec, ref, 'tiltseries')

        if self.params.eval_vol == True:
            # reconstruct a slice of the volume at halfway through the centre and at time = 0 
            rec = self.reconstruct_slices(ts.angles[0], self.params.proj_size//2,ts.times[0]).squeeze()
            rec = util.torch_to_np(rec).squeeze()
            self.results.update(self.step,rec, self.ref_vol[:,self.params.proj_size//2,:].squeeze(), 'volume')


        if self.params.hypertrain == True:
            pass
        else:
            self.results.publish()

    def reconstruct_slices(self, angles, depths, times):
        #Note all arrays must be same lenghth
        batch_val = 1
        if isinstance(depths, Iterable):
            if len(depths) > 1:
                batch_val = len(depths)
        manifold_output = self.manifold.get_value(angles,depths, times)
        mapnet_output = self.mapnet(manifold_output).reshape((batch_val,self.params.output_nch, self.params.style_size,self.params.style_size))
        out = self.net(mapnet_output)
        out= torch.permute(out, (2,0,3,1))
        return out
        
    
    def reconstruct(self, **kwargs):
        #define kwargs ts is a tiltseries object
        # or you can just input the angles and times
        usesetangle = kwargs.get('usesetangle', False)
        ts = kwargs.get('ts', None)
        angles =  kwargs.get('angles', ts.angles)
        times = kwargs.get('times', ts.times)
        if usesetangle:
            angles = torch.zeros_like(times)
        depths = kwargs.get('depths', range(self.params.proj_size))
        save_func = kwargs.get('save_func', None)
        saveas = kwargs.get('saveas', None)
        for i in range(len(times)):
            rec = np.zeros((self.params.proj_size, self.params.proj_size, len(depths)))
            tiled_angles = torch.full((len(depths),), angles[i]).to(self.params.dev)
            tiled_times = torch.full((len(depths),), times[i]).to(self.params.dev)
            rec[:,:,:] = util.torch_to_np(self.reconstruct_slices(tiled_angles, depths, tiled_times)).squeeze()
            print(tiled_angles.shape, tiled_times.shape, len(depths))
            if (save_func is not None) and (saveas is not None):
                save_func(rec, i, saveas)
            else:
                yield rec, util.torch_to_np(times[i])

    def state_dict(self):
        return {
            'scheduler': self.scheduler.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'net': self.net.state_dict(),
            'mapnet': self.mapnet.state_dict(),
            'manifold': self.manifold.to_dict(), 
            'params':   self.params.to_dict(),
            'step': self.step
            # Add any other state you want to save
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        solver = cls(fromdict=True)
        solver.params = params.Params.from_dict(state_dict['params'])
        solver._setup()

        solver.scheduler.load_state_dict(state_dict['scheduler'])
        solver.optimizer.load_state_dict(state_dict['optimizer'])
        solver.net.load_state_dict(state_dict['net'])
        solver.mapnet.load_state_dict(state_dict['mapnet'])
        solver.manifold.load_state_dict(state_dict['manifold'])
        
        solver.step = state_dict['step']
        solver.params.evaluate = False
        solver.params.eval_vol  = False
        solver.params.hypertrain = False
        
        return solver


