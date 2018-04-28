from pyrsistent import m
from six.moves import xrange
from torch import optim, nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from ops import float_tensor, long_tensor, one_hot, ones, zeros, init
import scipy.spatial.distance as distance
import models
import numpy as np

_exit = False

def sigint_handler(signal, frame):
  global _exit
  print "Early shutdown requested!"
  _exit = True

def train_gan(args, train_data, val_dataset, callbacks):
  G = models.Generator(args)
  D = models.Discriminator(args)
  g_optim = optim.Adam(G.parameters(), lr=args.learning_rate, betas=(.5, .999))
  d_optim = optim.Adam(D.parameters(), lr=args.learning_rate, betas=(.5, .999))
  bce = nn.BCEWithLogitsLoss()
  sce = nn.CrossEntropyLoss()

  def _batch_data(batch):
    images = float_tensor(batch[0].float())
    bsize = len(images)
    return m(images=images,
             x1=float_tensor(batch[3].float()),
             x2=float_tensor(batch[4].float()),
             id_labels=init(batch[1]),
             pose_labels=init(batch[2]),
             fake_pose_labels=long_tensor(np.random.randint(args.Np, size=bsize)),
             ones=ones(bsize),
             zeros=zeros(bsize),
             noise=float_tensor(np.random.uniform(-1., 1., (bsize, args.Nz))))

  def _interpolate(x1, x2, c, z):
    return G.dec((G.enc(x1)[0] + G.enc(x2)[0]) / 2., c, z)

  def _train_g(step, tensors):
    # Define training vars
    images = Variable(tensors.images)
    x1 = Variable(tensors.x1).unsqueeze(dim=0)
    x2 = Variable(tensors.x2).unsqueeze(dim=0)
    fake_pose_one_hot = Variable(one_hot(tensors.fake_pose_labels, args.Np))
    fake_pose_labels = Variable(tensors.fake_pose_labels)
    noise = Variable(tensors.noise)
    one_labels = Variable(tensors.ones)
    id_labels = Variable(tensors.id_labels)

    # Run models and get losses
    image_fused = G(images, fake_pose_one_hot, noise).x
    image_interpolated = _interpolate(x1, x2, fake_pose_one_hot, noise)  # Representation Interpolation (3.5)
    logits_fused = D(image_fused)
    logits_interpolated = D(image_interpolated)
    L_gan = bce(logits_fused.gan, one_labels) + bce(logits_interpolated.gan, one_labels)
    L_id = sce(logits_fused.id, id_labels)
    L_pose = sce(logits_fused.pose, fake_pose_labels) + sce(logits_interpolated.pose, fake_pose_labels)
    loss = L_id + L_pose + L_gan

    # Update models
    loss.backward()
    g_optim.step()
    callbacks.on_step_end(step, G, D, {
      "g_loss_total": loss,
      "g_loss_gan": L_gan,
      "g_loss_id": L_id,
      "g_loss_pose": L_pose
    })

  def _train_d(step, tensors):
    # Define training vars
    images = Variable(tensors.images)
    x1 = Variable(tensors.x1).unsqueeze(dim=0)
    x2 = Variable(tensors.x2).unsqueeze(dim=0)
    fake_pose_one_hot = Variable(one_hot(tensors.fake_pose_labels, args.Np))
    noise = Variable(tensors.noise)
    one_labels = Variable(tensors.ones)
    zero_labels = Variable(tensors.zeros)
    id_labels = Variable(tensors.id_labels)
    pose_labels = Variable(tensors.pose_labels)

    # TODO: proper multi image handling??
    # Run models and get losses
    logits_real = D(images[:, 0])
    logits_fake = D(G(images, fake_pose_one_hot, noise).x.detach())
    logits_interpolated = D(_interpolate(x1, x2, fake_pose_one_hot, noise))  # Representation Interpolation (3.5)
    L_gan_fake = bce(logits_fake.gan, zero_labels) + bce(logits_interpolated.gan, zero_labels)
    L_gan = bce(logits_real.gan, one_labels)
    L_id = sce(logits_real.id, id_labels)
    L_pose = sce(logits_real.pose, pose_labels[:, 0])
    loss = L_id + L_pose + L_gan + L_gan_fake

    # Update models
    loss.backward()
    d_optim.step()
    callbacks.on_step_end(step, G, D, {
      "d_loss_total": loss,
      "d_loss_gan": L_gan,
      "d_loss_gan_fake": L_gan_fake,
      "d_loss_id": L_id,
      "d_loss_pose": L_pose
    })

  def _validate(step):
    def _val(ds):
      val_data = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
      for batch in val_data:
        a = Variable(init(batch[0].float()))
        b = Variable(init(batch[1].float()))
        emb_a = G(a, None, None, emb_only=True).f_x.detach().data.cpu().numpy()
        emb_b = G(b, None, None, emb_only=True).f_x.detach().data.cpu().numpy()
        return [distance.cosine(emb_a[i], emb_b[i]) for i in range(emb_a.shape[0])]
    dist_same = _val(val_dataset[0])
    dist_not_same = _val(val_dataset[1])
    acc = np.mean([1. - d for d in dist_same] + dist_not_same)
    callbacks.on_val_end(step, acc)

  step = 0
  for epoch in xrange(1, args.epochs + 1):
    epoch_data = DataLoader(train_data.get_ds(shuffle=True), batch_size=args.batch_size, shuffle=False, num_workers=1)
    for batch in epoch_data:
      step += 1
      D.train()
      G.train()
      D.zero_grad()
      G.zero_grad()
      if step % args.train_ratio == 0:
        _train_d(step, _batch_data(batch))
      else:
        _train_g(step, _batch_data(batch))
      if val_dataset is not None and step % args.val_period == 0:
        _validate(step)
      if _exit: return
    callbacks.on_epoch_end(epoch, G, D)
  return G, D
