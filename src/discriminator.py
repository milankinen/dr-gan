from pyrsistent import m
from collections import OrderedDict
from torch import nn
from ops import conv_bn, comp_layers, avgpool
from misc import using_gpu

#
# G_enc :: image -> embedding + coefficient
#
class Discriminator(nn.Module):
  def __init__(self, params):
    super(Discriminator, self).__init__()
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
      conv_bn("Conv52", 160),
      conv_bn("Conv53", params.Nf),
      avgpool(6, 1))

    self.conv = nn.Sequential(OrderedDict(self.conv_layers))
    self.fc = nn.Linear(params.Nf, params.Nd + params.Np + 1)

    for m in self.modules():
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.02)

    if using_gpu():
      self.cuda()

  def forward(self, x):
    Nf, Nd, Np = self.params.Nf, self.params.Nd, self.params.Np
    x = self.conv(x).view((-1, Nf))
    y = self.fc(x)
    id = y[:, :Nd]
    pose = y[:, Nd:Nd + Np]
    gan = y[:, -1]
    return m(id=id, pose=pose, gan=gan)
