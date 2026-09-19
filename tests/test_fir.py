"""Independent 6D-padding reference from the source repository."""
import pytest
import torch
from torch.nn import functional as F
from model.backbones.up_or_down_sampling import upfirdn2d_native


def original(input,kernel,ux,uy,dx,dy,px0,px1,py0,py1):
    _,C,H,W=input.shape
    x=input.reshape(-1,H,W,1).view(-1,H,1,W,1,1)
    x=F.pad(x,[0,0,0,ux-1,0,0,0,uy-1]).view(-1,H*uy,W*ux,1)
    x=F.pad(x,[0,0,max(px0,0),max(px1,0),max(py0,0),max(py1,0)])
    x=x[:,max(-py0,0):x.shape[1]-max(-py1,0),max(-px0,0):x.shape[2]-max(-px1,0),:]
    x=x.permute(0,3,1,2).reshape(-1,1,H*uy+py0+py1,W*ux+px0+px1)
    x=F.conv2d(x,kernel.flip((0,1)).reshape(1,1,*kernel.shape))
    x=x.permute(0,2,3,1)[:,::dy,::dx,:]
    return x.reshape(-1,C,x.shape[1],x.shape[2])

@pytest.mark.parametrize('scales',[(1,1,1,1),(2,2,1,1),(1,1,2,2),(2,1,1,2)])
@pytest.mark.parametrize('pad',[(1,2,0,1),(-1,2,1,-1)])
def test_fir_value_gradient(scales,pad):
    x=torch.randn(2,3,8,10,dtype=torch.float64,requires_grad=True); k=torch.randn(3,3,dtype=torch.float64)
    got=upfirdn2d_native(x,k,*scales,*pad); ref=original(x,k,*scales,*pad)
    torch.testing.assert_close(got,ref,atol=0,rtol=0)
    probe=torch.randn_like(got)
    g1=torch.autograd.grad((got*probe).sum(),x,retain_graph=True)[0]
    g2=torch.autograd.grad((ref*probe).sum(),x)[0]
    torch.testing.assert_close(g1,g2,atol=0,rtol=0)
