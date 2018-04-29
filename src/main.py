from pyrsistent import m
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from training import train_gan, sigint_handler
from data.msceleb import load_training_data
from data.sample import load_gen_samples
from data.lfw import load_validation_dataset
from ops import float_tensor, long_tensor, one_hot
import argparse, json, misc, signal, os, math
import torch, numpy as np
import torchvision.utils as vutils

def _main():
  parser = argparse.ArgumentParser(description="Train DR-GAN model")
  parser.add_argument("--dataset", required=True)
  parser.add_argument("--take", type=int)
  parser.add_argument("--val_dataset")
  parser.add_argument("--logdir", required=True)
  parser.add_argument("--outdir")
  parser.add_argument("--epochs", type=int, default=10)
  parser.add_argument("--batch_size", type=int, default=64)
  parser.add_argument("--Nf", type=int, default=320)
  parser.add_argument("--Np", type=int, default=13)
  parser.add_argument("--Nz", type=int, default=50)
  parser.add_argument("--N_images", type=int, default=1)
  parser.add_argument("--learning_rate", type=float, default=.0002)
  parser.add_argument("--train_ratio", type=int, default=2)
  parser.add_argument("--d_precision_threshold", type=float, default=.85)
  parser.add_argument("--val_period", type=int, default=5000)
  parser.add_argument("--gen_images_period", type=int, default=2500)
  parser.add_argument("--save_period", type=int, default=50000)
  parser.add_argument("--model_switch_period", type=int, default=0)
  parser.add_argument("--D_weights")
  parser.add_argument("--G_weights")
  parser.add_argument("--gpu", action="store_true", default=False)
  args = parser.parse_args()
  print "Model params:"
  print json.dumps(vars(args), sort_keys=True, indent=2, separators=(',', ': '))

  sample_images = load_gen_samples()
  train_data = load_training_data(args.dataset, args, limit=args.take)
  val_dataset = load_validation_dataset(args.val_dataset) if args.val_dataset else None
  args.Nd = train_data.Nd
  signal.signal(signal.SIGINT, sigint_handler)
  writer = SummaryWriter(log_dir=args.logdir)

  print "Num IDs: %d" % args.Nd

  def _gen_sample_images(step, G):
    print "Generating %d sample images..." % len(sample_images)
    G.eval()
    x = float_tensor(np.array([si[1] for si in sample_images]))
    z = float_tensor(np.random.uniform(-1., 1., (x.shape[0], args.Nz)))
    c = one_hot(long_tensor(np.array([misc.pose_class(args.Np, 0.)]).repeat(x.shape[0])), args.Np)
    gen = G(Variable(x), Variable(c), Variable(z)).x.detach().data
    for i, s in enumerate(sample_images):
      grid = vutils.make_grid(gen[i:i + 1], 1, normalize=True, scale_each=True, padding=0)
      writer.add_image(s[0], grid, step)
    if args.outdir:
      misc.mkdir_p(os.path.join(args.outdir, "generated_images"))
      img_path = os.path.join(args.outdir, "generated_images", "%d.jpg" % step)
      vutils.save_image(gen, img_path, normalize=True)
    print "Sample images generated!"

  def _save_weights(step, G, D):
    weight_path = os.path.join(args.outdir, "weights")
    print "Saving model weights..."
    misc.mkdir_p(weight_path)
    torch.save(G.state_dict(), os.path.join(weight_path, "%s.g.pt" % str(step)))
    torch.save(D.state_dict(), os.path.join(weight_path, "%s.d.pt" % str(step)))
    print "Model weights saved!"

  def on_train_begin(G, D):
    if args.D_weights:
      d_weights = torch.load(args.D_weights)
      D.load_state_dict(d_weights)
      print "Loaded D initial weights from %s" % args.D_weights
    if args.G_weights:
      g_weights = torch.load(args.G_weights)
      G.load_state_dict(g_weights)
      print "Loaded G initial weights from %s" % args.G_weights

  def on_step_end(step, G, D, vals):
    for n in vals:
      writer.add_scalar(n, vals[n], step)
    if step % args.gen_images_period == 0:
      _gen_sample_images(step, G)
    if args.outdir:
      if step % args.save_period == 0:
        _save_weights(step, G, D)
      elif step % 1000 == 0:
        _save_weights("latest", G, D)

  def on_epoch_end(epoch, G, D):
    print "Epoch ended: %d" % epoch

  def on_val_end(step, accuracy):
    writer.add_scalar("val_accuracy", accuracy, step)

  with misc.gpu_flag(args.gpu):
    callbacks = m(on_step_end=on_step_end,
                  on_epoch_end=on_epoch_end,
                  on_val_end=on_val_end,
                  on_train_begin=on_train_begin)
    G, D, = train_gan(args, train_data, val_dataset, callbacks)
    if args.outdir:
      _save_weights("final", G, D)

  writer.close()

if __name__ == "__main__":
  _main()
