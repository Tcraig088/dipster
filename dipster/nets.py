import numpy as np
import torch
import torch.nn as nn 
from dipster import util

def get_params(net):
    params = []
    params += [x for x in net.parameters() ]
    return params

class Manifold():
    def __init__(self, times, depth, device):
        #cast times and depth to float
        self.times = float(times)
        self.depth = float(depth)-1
        self.device = device

        
    def get_value(self, ang,z,t):
        x = torch.cos((ang + 90)* torch.pi / 180).to(self.device)
        y = torch.sin((ang + 90)* torch.pi / 180).to(self.device)
        if not isinstance(t,torch.Tensor):
            t = torch.tensor(t).to(self.device)
        if not isinstance(z,torch.Tensor):
            z = torch.tensor(z).to(self.device).to(torch.float32)
        t_norm = t / self.times
        z_norm = z / self.depth
        print(x, y, z_norm, t_norm)
        manifold = torch.stack([x, y, z_norm, t_norm])
        if len(manifold.shape)<2:
            manifold = manifold[:, None]
        manifold = manifold.permute(1, 0).float().to(self.device)
        return manifold 

    def to_dict(self):
        return {
            'times': self.times,
            'depth': self.depth,
            'device': self.device
        }
    
    def load_state_dict(self, d):
        self.times = d['times']
        self.depth = d['depth']
        self.device = d['device']   

    
def conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class MappingNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        
        latent_dim = opt.latent_dim
        style_dim = opt.style_size**2
        hidden_dim = opt.hidden_dim
        depth = opt.depth
        
        layers = []
        layers += [nn.Linear(latent_dim, hidden_dim)]
        layers += [nn.ReLU()]
        for _ in range(depth):
            layers += [nn.Linear(hidden_dim, hidden_dim)]
            layers += [nn.ReLU()]
        layers += [nn.Linear(hidden_dim, style_dim)]
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, z):
        out = self.net(z)        
        return out


class Net(nn.Module):
    def __init__(self, opt):
        super().__init__() 
        inp_ch=opt.input_nch
        ndf=opt.proj_size
        out_ch=opt.output_nch
        Nr=opt.Nr
        num_ups=int(np.log2(opt.up_factor))
        need_bias=opt.need_bias
        upsample_mode=opt.upsample_mode
        
        layers = [conv(inp_ch, ndf, 3, bias=need_bias), 
                  nn.BatchNorm2d(ndf),
                  nn.ReLU(True)]
        
        for _ in range(Nr):
            layers += [conv(ndf, ndf, 3, bias=need_bias),
                       nn.BatchNorm2d(ndf),
                       nn.ReLU(True)]

        for _ in range(num_ups):
            layers += [nn.Upsample(scale_factor=2, mode=upsample_mode),
                       conv(ndf, ndf, 3, bias=need_bias),                                         
                       nn.BatchNorm2d(ndf),
                       nn.ReLU(True)]
            for _ in range(Nr):
                layers += [conv(ndf, ndf, 3, bias=need_bias),
                           nn.BatchNorm2d(ndf),
                           nn.ReLU(True)]

        layers += [conv(ndf, out_ch, 3, bias=need_bias)]


        self.net = nn.Sequential(*layers)
        
    def forward(self, z, s=None):
        out = self.net(z)        
        return out