from PIL import Image
import random, math
import numpy as np

def pose_class(Np, yaw):
  return max(0, min(Np - 1, int((90. + float(yaw)) / (180. / Np))))

def load_image(image_path, resize=None, crop=None, randomize=False):
  def _resize(img, method=Image.ANTIALIAS):
    img.thumbnail(resize, method)
    offset = (int((resize[0] - img.size[0]) / 2), int((resize[1] - img.size[1]) / 2))
    back = Image.new("RGB", resize, (0, 0, 0))
    back.paste(img, offset)
    return back

  def _random_crop(img):
    width, height = img.size
    top, left = random.randint(0, width - crop[0] - 1), random.randint(0, height - crop[1] - 1)
    return img.crop((left, top, left + crop[0], top + crop[1]))

  def _center_crop(img):
    width, height = img.size
    top, left = math.ceil((width - crop[0]) / 2), math.ceil((height - crop[1]) / 2)
    return img.crop((left, top, left + crop[0], top + crop[1]))

  img = Image.open(image_path, mode="r").convert(mode="RGB")
  if resize is not None:
    img = _resize(img)
  if crop is not None:
    img = _random_crop(img) if randomize else _center_crop(img)
  img = np.array(img) / 127.5 - 1. # normalize
  return np.transpose(img, (2, 0, 1))  # -> C, W, H