#import tomosipo as tp

import astra
import os 
import torch 
import astra
import sys
from collections.abc import Iterable
from dipster import util
import numpy as np

def _create_projector(x, y, z, angles):
    print('Creating projector with geometry (x, y, z, n):', x, y, z, angles.shape[0])
    proj_geom = astra.creators.create_proj_geom('parallel3d', 1, 1,  y, x, angles * np.pi / 180)
    vol_geom = astra.creators.create_vol_geom(y,x,z)
    proj_id = astra.creators.create_projector('cuda3d', proj_geom, vol_geom)
    return proj_id

def fp(rec, ang):
    """Create a sinogram from a volume using forward projection. The GPU Context is overriden due to underlying astra gpu usage. 
    Args:
        volume (Volume): The input volume to be projected.
        angles (Union[Tuple[TiltSchemeAbstract, slice], np.ndarray]): The angles at which to project the volume. Can be a TiltSchemeAbstract with a slice of angles or a numpy array of angles.
        use_gpu (bool): Whether to use GPU for projection. Default is True.
    Returns:
        Sinogram: The resulting sinogram.

    """
    device = rec.device
    dim, batch, channels = rec.shape[0], rec.shape[1], rec.shape[3]
    print('rec shape before permutation', rec.shape)
    #rec = torch.permute(rec, (1,0,2,3))
    print('rec shape before projection', rec.shape)
    rec = util.torch_to_np(rec)
    ang = util.torch_to_np(ang)

    proj_id = _create_projector(dim, batch, dim, ang)
    W = astra.OpTomo(proj_id)
    sino = np.zeros((batch, batch, dim, channels))
    
    for i in range(channels):
        print('Reconstruction Gemoetry (sino, rec)',sino[:, :, :, i].shape, rec[:, :, :, i].shape)
        print('Projection Geometry (sino, rec)',W.sshape, W.vshape)
        sino[:, :, :, i] = W.FP(rec[:, :, :, i])

    print('Sino shape after reconstruction', sino.shape)
    sino = sino.transpose(2, 0, 1, 3)  # ASTRA gives (z, n, d)
    print('Sino shape after transpose', sino.shape)
    sino = util.np_to_torch(sino, device)
    astra.astra.delete(proj_id)
    return sino

def bp(sino, ang):
    
    device = sino.device
    dim, batch, channels = sino.shape[0], sino.shape[1], sino.shape[3]
    
    print('sino shape before permutation', sino.shape)
    sino = util.torch_to_np(sino)
    sino = sino.transpose(3, 2, 0, 1)  # ASTRA gives (z, n, d)
    print('sino shape after transpose', sino.shape)
    ang = util.torch_to_np(ang)

    rec = np.zeros((dim, channels, dim, batch))
    
    for i in range(batch):
        proj_id = _create_projector(dim, channels, dim, np.array([ang[i]]))
        W = astra.OpTomo(proj_id)
        print( 'Data Geometry (sino, rec)',sino[:,:,:, i].shape,rec[:,:,:,i].shape)
        print('Reconstruction Geometry (sino, rec)',W.sshape, W.vshape)
        rec[:,:,:,i] = W.BP(sino[:,:,:, i])
        astra.astra.delete(proj_id)
    rec = rec.transpose(0,3,2,1)  # ASTRA gives (z, n, d)
    rec = util.np_to_torch(rec, device)
    return rec


