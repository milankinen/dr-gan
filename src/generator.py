from collections import OrderedDict
from pyrsistent import m
from torch import nn
from ops import conv_bn, comp_layers, avgpool, sigmoid, fconv_bn
from misc import using_gpu
import torch

def fused_avg(emb, w):
  normalized_w = w / w.sum(dim=-1, keepdim=True)
  return (emb * normalized_w.unsqueeze(-1)).sum(dim=-2)

#
# G_enc :: image -> embedding + coefficient
#
class Encoder(nn.Module):
  def __init__(self, params):
    super(Encoder, self).__init__()
    self.multi_image = params.N_images > 1
    self.params = params
    self.conv_layers = comp_layers(
      3,  # num channels
      conv_bn("Conv11", 32),
      conv_bn("Conv12", 64),
      conv_bn("Conv21", 64, downsample=True),
      conv_bn("Conv22", 64),
      conv_bn("Conv23", 128),
      conv_bn("Conv31", 128, downsample=True),
      conv_bn("Conv32", 96),
      conv_bn("Conv33", 192),
      conv_bn("Conv41", 192, downsample=True),
      conv_bn("Conv42", 128),
      conv_bn("Conv43", 256),
      conv_bn("Conv51", 256, downsample=True),
      conv_bn("Conv52", 160))

    self.emb_layers = comp_layers(
      160,
      conv_bn("Conv53", params.Nf),
      avgpool(6, 1))

    self.conv = nn.Sequential(OrderedDict(self.conv_layers))
    self.emb = nn.Sequential(OrderedDict(self.emb_layers))

    if self.multi_image > 1:
      self.w_layers = comp_layers(
        160,
        conv_bn("Conv53_W", 1),
        avgpool(6, 1),
        sigmoid)
      self.coefficient = nn.Sequential(OrderedDict(self.w_layers))

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.02)

    if using_gpu():
      self.cuda()

  def forward(self, x):
    if self.multi_image > 1:
      assert self.multi_image
      batch_size = x.size(0)
      x = x.view(batch_size * self.params.N_images, 3, 96, 96)
      x = self.conv(x)
      emb = self.emb(x).view((batch_size, self.params.N_images, self.params.Nf))
      w = self.coefficient(x).view((batch_size, self.params.N_images))
      fused = fused_avg(emb, w) if x.size[1] > 1 else emb.view((-1, self.params.Nf))
      return fused, emb, w
    else:
      x = self.conv(x.squeeze())
      emb = self.emb(x).view((-1, 1, self.params.Nf))
      return emb.view((-1, self.params.Nf)), emb, None

#
# G_dec :: embedding + noise + pose -> image
#
class Decoder(nn.Module):
  def __init__(self, params):
    super(Decoder, self).__init__()
    self.params = params

    # Paper says that FC layer should have output of 6x6x620 - however, the decompiled Tensorflow
    # model tells that the output is actually 3x3x320 directly followed by upsampling FConv52 layer.
    # We'll be using the same network topology as the de-compiled mode
    self.fc = nn.Linear(params.Nf + params.Nz + params.Np, 3 * 3 * params.Nf)
    self.bn = nn.BatchNorm2d(3 * 3 * params.Nf)

    self.fconv_layers = comp_layers(
      params.Nf,
      fconv_bn("FConv52", 160, upsample=True),
      fconv_bn("FConv51", 256),
      fconv_bn("FConv43", 256, upsample=True),
      fconv_bn("FConv42", 128),
      fconv_bn("FConv41", 192),
      fconv_bn("FConv33", 192, upsample=True),
      fconv_bn("FConv32", 96),
      fconv_bn("FConv31", 128),
      fconv_bn("FConv23", 128, upsample=True),
      fconv_bn("FConv22", 64),
      fconv_bn("FConv21", 64),
      fconv_bn("FConv13", 64, upsample=True),
      fconv_bn("FConv12", 32),
      fconv_bn("FConv11", 3, activation=nn.Tanh))

    self.x_h = nn.Sequential(OrderedDict(self.fconv_layers))

    for m in self.modules():
      if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.02)

    if using_gpu():
      self.cuda()

  def forward(self, f_x, c, z):
    x = torch.cat([f_x, c, z], dim=1)
    x = self.fc(x).view((-1, self.params.Nf, 3, 3))
    x = self.x_h(x)
    return x

"""
from ops import float_tensor
from torch.autograd import Variable
import numpy as np
d = Decoder(m(Nz=100, Np=13, Nf=320))
f_x = Variable(float_tensor(np.random.uniform(-1, 1, (3, 320))))
z = Variable(float_tensor(np.random.uniform(-1, 1, (3, 100))))
c = Variable(float_tensor(np.random.uniform(-1, 1, (3, 13))))
print d(f_x, z, c)
"""

class Generator(nn.Module):
  def __init__(self, params):
    super(Generator, self).__init__()
    self.enc = Encoder(params)
    self.dec = Decoder(params)

  def forward(self, x, c, z, emb_only=False):
    f_x, emb, w = self.enc(x)
    return m(f_x=f_x, emb=emb, w=w, x=self.dec(f_x, c, z) if emb_only is False else None)
