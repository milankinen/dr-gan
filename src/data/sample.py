from os import path
from glob import glob
from PIL import Image
import numpy as np

def _load_image(path):
  img = np.array(Image.open(path, "r").convert("RGB"))
  assert img.shape[0] == 96
  assert img.shape[1] == 96
  return np.transpose(img / 127.5 - 1., (2, 0, 1)).reshape((1, 3, 96, 96))

#
# Load samples for image generation
#
def load_gen_samples():
  image_paths = [path.abspath(g) for g in glob("%s/sample/*.png" % path.dirname(__file__))]
  return [(path.basename(p), _load_image(p)) for p in image_paths]
