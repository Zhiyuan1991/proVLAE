import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import copy
import scipy

def make_gif(images, fname, duration=2, true_image=False):
  import moviepy.editor as mpy

  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8)

  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_gif(fname, fps = len(images) / duration)

def save_images(images, size, image_path):
    images=np.array(images)
    images=np.reshape(images,(images.shape[0],images.shape[1],images.shape[2],1))
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(image_path, image)

def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  if (images.shape[3] in (3,4)):
    c = images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img
  elif images.shape[3]==1:
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
    return img
  else:
    raise ValueError('in merge(images,size) images parameter '
                     'must have dimensions: HxW or HxWx3 or HxWx4')

def load_checkpoints(sess,model,flags):
  if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")
  saver = tf.train.Saver(max_to_keep=1)
  checkpoint = tf.train.get_checkpoint_state(flags.checkpoint_dir)
  if checkpoint and checkpoint.model_checkpoint_path:
    #saver.restore(sess, checkpoint.model_checkpoint_path)
    try:
      saver.restore(sess, checkpoint.model_checkpoint_path)
    except:
      print("direct restoration failed, try loading existing parameters only")
      if flags.mode!=1:
        print("Not in train mode, better check model structure")
      optimistic_restore(sess,checkpoint.model_checkpoint_path)
    print("loaded checkpoint: {0}".format(checkpoint.model_checkpoint_path))
  else:
    print("Could not find old checkpoint")
    if not os.path.exists(flags.checkpoint_dir):
      os.mkdir(flags.checkpoint_dir)
  return saver

def optimistic_restore(session, save_file):
    #Adapt code from https://github.com/tensorflow/tensorflow/issues/312
    #only load those variables that exist in the checkpoint file
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = [(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                        if var.name.split(':')[0] in saved_shapes]
    restore_vars = []
    name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)

def compute_mean_loss(sess,model,manager,flags):
  #for a given dataset, compute average reconstrcution loss and KL divergence
  n_samples = manager.sample_size
  indices = list(range(n_samples))

  total_batch = n_samples // flags.batch_size
  print(n_samples,total_batch,flags.batch_size)

  recon_total=0
  latent_total=0
  for i in range(total_batch):
    batch_indices = indices[flags.batch_size*i : flags.batch_size*(i+1)]
    batch_xs = manager.get_images(batch_indices)
    recons_loss,latent_loss = model.get_recons_loss(sess, batch_xs)
    recon_total+=recons_loss*flags.batch_size
    latent_total+=latent_loss*flags.batch_size
  recon_total=np.array(recon_total)
  latent_total=np.array(latent_total)
  print("recon:",recon_total/float(n_samples),"latent:",latent_total/float(n_samples))