import torch
from torch import nn, optim
from torch import Tensor
import numpy as np
from dipster import tomo, util

class custom_grad_func(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing 
    torch.autograd.Function and implementing the forward and backward passes 
    which operate on Tensors.
    NuFFT from https://github.com/jyhmiinlin/pynufft
    """

    @staticmethod
    def forward(ctx,input_r, angle):
        """
        In the forward pass we receive a Tensor containing the input and return 
        a Tensor containing the output. ctx is a context object that can be used 
        to stash information for backward computation. You can cache arbitrary 
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.dev = input_r.get_device()
        ctx.angle=angle
        x_val = input_r
        y = tomo.fp(x_val, angle)
        y_cut = torch.zeros(y.shape[0],y.shape[1],1,y.shape[3]).to(ctx.dev)
        for i in range(y.shape[3]):
            for j in range(y.shape[1]):
                y_cut[:,j,0,i] = y[:,j,j,i]
        return y_cut
        

    @staticmethod
    def backward(ctx,grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss 
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        angle = ctx.angle
        if torch.numel(angle) == 1:
            angle = torch.tensor([angle.item()])

        yc = grad_output
        grad_output = tomo.bp(yc, angle, 1)

        #return grad_output
        return grad_output, None, None, None, None, None, None

class CustomGradient(nn.Module):
    def __init__(self,ImageSize,batch,channels):
        super(CustomGradient,self).__init__()
        self.X=Tensor(ImageSize,batch, ImageSize,channels).fill_(0).cuda()
        #self.X=Tensor(ImageSize,ImageSize).fill_(0)
        
    def forward(self,angles):
        return  custom_grad_func.apply(self.X, angles)