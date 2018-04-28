import os, torch, torchvision
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from hopenet import Hopenet

_weights_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "/hopenet_robust_alpha1.pkl")
_model = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
_model.load_state_dict(torch.load(_weights_path, map_location="cpu"))
# Change model to 'eval' mode (BN uses moving mean/var)
_model.eval()

def detect_pose(face_img):
  transformations = transforms.Compose([transforms.Resize(224),
                                        transforms.CenterCrop(224), transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
  idx_tensor = [idx for idx in xrange(66)]
  idx_tensor = torch.FloatTensor(idx_tensor)
  face_img = transformations(face_img)
  face_shape = face_img.size()
  face_img = face_img.view(1, face_shape[0], face_shape[1], face_shape[2])
  face_img = Variable(face_img)

  yaw_logit, pitch_logit, roll_logit = _model(face_img)
  yaw_p = F.softmax(yaw_logit, dim=-1)
  #pitch_p = F.softmax(pitch_logit, dim=-1)
  #roll_p = F.softmax(roll_logit, dim=-1)
  # convert predictions to degrees
  yaw = torch.sum(yaw_p.data[0] * idx_tensor) * 3 - 99
  #pitch = torch.sum(pitch_p.data[0] * idx_tensor) * 3 - 99
  #roll = torch.sum(roll_p.data[0] * idx_tensor) * 3 - 99
  return yaw #, pitch, roll
