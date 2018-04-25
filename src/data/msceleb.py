from torch.utils.data import Dataset
from pyrsistent import m, freeze
from toolz import pipe
from PIL import Image
from cStringIO import StringIO
from threading import Lock
import toolz.curried as tz
import math, mmap, json, random, util, numpy as np

#
# Load MS-Celeb low-shot training data using pre-generated metadata info
#
def load_training_data(tsv_path, args, limit=None):
  with open(tsv_path + ".meta.json", "r") as mjf:
    meta = json.load(mjf)

  tsv_file = open(tsv_path, "r+b")
  mm_lock = Lock()
  mm = mmap.mmap(tsv_file.fileno(), 0)

  def _pose_class(yaw):
    return util.pose_class(args.Np, yaw)

  def _crop(img):
    w, h = img.size
    if w == h:
      return img
    elif w > h:
      s = math.ceil((w - h) / 2)
      return img.crop((s, 0, s + h, h))
    else:
      s = math.ceil((h - w) / 4 * 3)
      return img.crop((0, s, w, s + w))

  def _resize(img):
    if img.size[0] == 96:
      return img
    else:
      return img.resize((96, 96), Image.LANCZOS)

  def _shuffled(images):
    images = images[:]
    random.shuffle(images)
    return images

  def _load_image(image):
    mm_lock.acquire()
    try:
      mm.seek(image.pos)
      face_data = mm.readline().split('\t')[1]
    finally:
      mm_lock.release()
    img = Image.open(StringIO(face_data.decode("base64")), "r").convert("RGB")
    img = _resize(_crop(img))
    img = np.array(img) / 127.5 - 1.  # normalize
    return np.transpose(img, (2, 0, 1))  # -> C, W, H

  def _make_samples(meta, shuffle):
    def _to_sample(person, images):
      return m(id=person["id_class"] - 1, images=freeze(list(images)))
    samples = pipe(
      meta["persons"],
      tz.take(limit) if limit is not None else tz.identity,
      tz.map(lambda p: m(p=p, i=tz.partition(args.N_images, _shuffled(p["images"]) if shuffle else p["images"]))),
      tz.mapcat(lambda s: [_to_sample(s.p, i) for i in s.i]),
      tz.take(limit) if limit is not None else tz.identity,
      list)
    if shuffle:
      random.shuffle(samples)
    return samples

  class MSCelebDataset(Dataset):
    def __init__(self, samples):
      self.samples = samples
      self.tsv_file = tsv_file  # keep reference to file descriptor
    def __len__(self):
      return len(self.samples)
    def __getitem__(self, idx):
      s = self.samples[idx]
      images = np.array([_load_image(i) for i in s.images])
      poses = np.array([_pose_class(i.yaw) for i in s.images])
      return [images, s.id, poses]
  class MSCelebTrainData(object):
    @property
    def Nd(self):
      return meta["num_id_classes"]
    def get_ds(self, shuffle=True):
      return MSCelebDataset(_make_samples(meta, shuffle))

  return MSCelebTrainData()
