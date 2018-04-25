from torch import nn, FloatTensor, LongTensor
from misc import using_gpu
import torch

class conv_bn(object):
  def __init__(self, name, next_filters, downsample=False, activation=nn.ELU):
    self.name = name
    self.next_filters = next_filters
    self.downsample = downsample
    self.activation = activation
  def __call__(self, prev_filters):
    conv = nn.Conv2d(prev_filters, self.next_filters,
                     kernel_size=3,
                     stride=(2 if self.downsample else 1),
                     padding=1,
                     bias=True)
    bn = nn.BatchNorm2d(self.next_filters)
    activation = self.activation()
    conv._name = self.name
    bn._name = "%s_bn" % self.name
    activation._name = "%s_activation" % self.name
    return self.next_filters, [conv, bn, activation]

class FConv(nn.ConvTranspose2d):
  def __init__(self, upsample, *args, **kwargs):
    super(FConv, self).__init__(*args, **kwargs)
    self._upsample = upsample
  def forward(self, input, **kwargs):
    if self._upsample:
      in_size = input.size()
      out_size = (in_size[0], self.out_channels, in_size[2] * 2, in_size[3] * 2)
      return super(FConv, self).forward(input, output_size=out_size)
    else:
      return super(FConv, self).forward(input, **kwargs)

class fconv_bn(object):
  def __init__(self, name, next_filters, upsample=False, activation=nn.ELU):
    self.name = name
    self.next_filters = next_filters
    self.upsample = upsample
    self.activation = activation
  def __call__(self, prev_filters):
    conv = FConv(self.upsample, prev_filters, self.next_filters,
                 kernel_size=3,
                 stride=(2 if self.upsample else 1),
                 padding=1,
                 bias=True)
    bn = nn.BatchNorm2d(self.next_filters)
    activation = self.activation()
    conv._name = self.name
    bn._name = "%s_bn" % self.name
    activation._name = "%s_activation" % self.name
    layers = [conv, bn, activation]
    return self.next_filters, layers

class avgpool(object):
  def __init__(self, kernel_size, strides):
    self.kernel_size = kernel_size
    self.strides = strides
  def __call__(self, *args):
    ap = nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.strides)
    ap._name = "AvgPool"
    return None, [ap]

def sigmoid(_):
  s = nn.Sigmoid()
  s._name = "Sigmoid"
  return None, [s]

def comp_layers(input, *layers):
  res = []
  for lrs in layers:
    input, ls = lrs(input)
    res = res + [(l._name, l) for l in ls]
  return res

def init(x):
  return x.cuda() if using_gpu() else x

def float_tensor(val):
  return init(FloatTensor(val))

def long_tensor(val):
  return init(LongTensor(val))

def one_hot(label, num_classes):
  ones = init(torch.sparse.torch.eye(num_classes))
  return init(ones.index_select(0, label))

def ones(size):
  return init(torch.ones(size))

def zeros(size):
  return init(torch.zeros(size))
