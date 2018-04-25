from torch.utils.data import Dataset
from pyrsistent import m
from util import load_image
import os

def load_validation_dataset(val_pairs_tsv_path):
  image_dir = os.path.join(os.path.dirname(val_pairs_tsv_path), "images")

  def _img_path(person_id, image_num):
    return os.path.join(image_dir, "%s__%04d.png" % (person_id, image_num))

  def _load_image(image_path):
    img = load_image(image_path, resize=(110, 110), crop=(96, 96))
    return img.reshape((1,) + img.shape)

  samples_same, samples_not_same = [], []
  with open(val_pairs_tsv_path, "r") as f:
    num = int(f.readline())
    for _ in range(num):
      pid, anum, bnum = f.readline().split("\t")
      samples_same.append(m(aid=pid, bid=pid, a=int(anum), b=int(bnum)))
    for _ in range(num):
      aid, anum, bid, bnum = f.readline().split("\t")
      samples_not_same.append(m(aid=aid, bid=bid, a=int(anum), b=int(bnum)))

  class LFWDataset(Dataset):
    def __init__(self, samples):
      self.samples = samples
      self.a = [_load_image(_img_path(s.aid, s.a)) for s in samples]
      self.b = [_load_image(_img_path(s.bid, s.b)) for s in samples]
    def __len__(self):
      return len(self.samples)
    def __getitem__(self, idx):
      return [self.a[idx], self.b[idx]]

  return LFWDataset(samples_same), LFWDataset(samples_not_same)
