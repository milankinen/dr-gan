from toolz import pipe
from PIL import Image
from cStringIO import StringIO
from pose_detection import detect_pose
from multiprocessing import Process, Queue
from datetime import datetime
from tqdm import tqdm
import toolz.curried as tz
import os, mmap, json, argparse, signal

def _pad_image(img):
  if img.size[0] == img.size[1]:
    return img
  longer_side = max(img.size)
  hpad = (longer_side - img.size[0]) / 2
  vpad = (longer_side - img.size[1]) / 2
  return img.crop((-hpad, -vpad, img.size[0] + hpad, img.size[1] + vpad))

def _process(q, worker_idx, tsv_path, num_par):
  meta_path = tsv_path + (".meta.%d-%d.json" % (worker_idx, num_par))

  meta = {"persons": []}
  if os.path.isfile(meta_path):
    meta = json.load(open(meta_path, "r"))

  id_class_counter = 1
  p_index = {}
  i_index = {}
  for p in meta["persons"]:
    p_index[p["ext_id"]] = p
    if p["id_class"] >= id_class_counter:
      id_class_counter = p["id_class"] + 1
    for i in p["images"]:
      key = "%s:%s" % (p["ext_id"], i["id"])
      i_index[key] = i

  def _save_meta():
    with open(meta_path, "w") as mf:
      json.dump({"num_id_classes": id_class_counter - 1, "persons": list(p_index.values())}, mf)

  with open(tsv_path, "r+b") as tsv:
    mm = mmap.mmap(tsv.fileno(), 0)
    step = 0
    while True:
      step += 1
      if step % 1000 == 0:
        _save_meta()
      pos = mm.tell()
      line = mm.readline()
      if (step - 1 + worker_idx) % num_par != 0:
        continue  # not belonging to this worker
      if not line:
        break
      image_id, face_data, mid, _, _ = line.split('\t')
      key = "%s:%s" % (mid, image_id)
      if key in i_index:
        q.put(1)
        continue  # already processed
      try:
        face_data = face_data.decode("base64")
        face_img = _pad_image(Image.open(StringIO(face_data), "r"))
        yaw = detect_pose(face_img)
        if not mid in p_index:
          p_index[mid] = {"ext_id": mid, "id_class": id_class_counter, "images": []}
          id_class_counter += 1
        p = p_index[mid]
        i = i_index[key] = {"id": image_id, "yaw": yaw, "pos": pos}
        p["images"].append(i)
      except Exception, e:
        print "ERROR (%s, %s): %s" % (mid, image_id, str(e))
      finally:
        q.put(1)

    _save_meta()
    mm.close()

_terminated = False

def sigterm_handler(signal, frame):
  global _terminated
  _terminated = True

def _progressbar(q, total):
  signal.signal(signal.SIGTERM, sigterm_handler)
  if total is None:
    print "No --lines given, can't print progress bar"
    print "Check tsv file line number with wc -l <file.tsv> and pass it as --line=<val>"
    print ""
    return
  with tqdm(total=total) as pbar:
    while not _terminated:
      pbar.update(q.get())

def _main():
  start = datetime.now()
  parser = argparse.ArgumentParser("MSCeleb parallelized pre-processing")
  parser.add_argument("--num_parallelism", type=int, default=1)
  parser.add_argument("--lines", type=int)  # TrainData_Base.tsv = 1155175
  parser.add_argument("tsv_file")
  args = parser.parse_args()
  tsv_path = args.tsv_file
  num_par = args.num_parallelism
  q = Queue(2 * num_par)
  workers = [Process(target=_process, args=(q, i, tsv_path, num_par)) for i in range(num_par)]
  pb_proc = Process(target=_progressbar, args=(q, args.lines))

  print "Start data processing..."
  pb_proc.start()
  for w in workers:
    w.start()
  for w in workers:
    w.join()

  def _combine(persons):
    images = pipe(
      persons,
      tz.mapcat(lambda m: m["images"]),
      tz.groupby(lambda p: p["id"]),
      tz.valmap(tz.first)).values()
    return {"images": images}

  # All workers done, combine
  metas = [json.load(open(tsv_path + (".meta.%d-%d.json" % (i, num_par)), "r")) for i in range(num_par)]
  persons = pipe(
    metas,
    tz.mapcat(lambda m: m["persons"]),
    tz.groupby(lambda p: p["ext_id"]),
    tz.valmap(_combine)).items()

  pp = []
  for idx, (ext_id, p) in enumerate(persons):
    pp.append({
      "ext_id": ext_id,
      "id_class": idx + 1,
      "images": list(p["images"])
    })

  with open(tsv_path + ".meta.json", "w") as mf:
    json.dump({"num_id_classes": len(pp), "persons": pp}, mf, sort_keys=False, indent=2, separators=(',', ': '))

  q.put(1)
  pb_proc.terminate()
  pb_proc.join()

  end = datetime.now()
  print "Done, took %f seconds" % (end - start).total_seconds()

if __name__ == "__main__":
  _main()
