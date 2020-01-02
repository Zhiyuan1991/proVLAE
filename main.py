from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import random
import os
from scipy.misc import imsave

from data_manager import *
from utils import *
import copy

from metric_MIG import *
from shutil import copyfile
import glob

tf.app.flags.DEFINE_integer("epoch_size", 15, "epoch size")
tf.app.flags.DEFINE_integer("batch_size", 100, "batch size")
tf.app.flags.DEFINE_float("beta", 1.0, "beta for the KL loss")
tf.app.flags.DEFINE_float("learning_rate", 5e-4, "learning rate")
tf.app.flags.DEFINE_string("checkpoint_dir", "checkpoints/3dshapes_progress_z3_b8_2", "checkpoint directory")
tf.app.flags.DEFINE_string("log_file", "./log", "log file directory")
tf.app.flags.DEFINE_integer("train_seq",3,"training sequence number")
tf.app.flags.DEFINE_boolean("KL", False, "pre-trained KL loss or not")
tf.app.flags.DEFINE_boolean("fadein", False, "fadein new blocks or not")
tf.app.flags.DEFINE_float("coff", 1., "coff for pre-trained KL loss") #reduce this to decrease TF GPU space for pytorch
tf.app.flags.DEFINE_float("gpu_usage", 1., "gpu usage fraction, reduce this to save ")
tf.app.flags.DEFINE_integer("z_dim", 3, "dimensions for each latent variable")
tf.app.flags.DEFINE_integer("mode", 1, "mode. 1: training one step; 2: display results; 3: compute metrics")
flags = tf.app.flags.FLAGS

test_img_ind=0 #image indice for visualization

#settings
z_dim = flags.z_dim
print("z_dim:",z_dim)

if 'Celeb' in flags.checkpoint_dir or '3dshapes' in flags.checkpoint_dir:
  chn_num=3
  image_shape = [64, 64, 3]
else: #dsprite
  chn_num = 1
  image_shape = [64, 64]

#models
if 'Celeb' in flags.checkpoint_dir: #bigger model, L=4
  from model_ladder_pro_celbA import VAE_ladder
else:# L=3
  from model_ladder_progress import VAE_ladder

def train(sess,model,manager,saver):
  summary_writer = tf.summary.FileWriter(flags.log_file, sess.graph)
  n_samples = manager.sample_size
  indices = list(range(n_samples))

  iter = 0

  for epoch in range(flags.epoch_size):
    recon_epoch = 0
    KL_epoch = 0
    KL_list_epoch=[]

    random.shuffle(indices)
    
    total_batch = n_samples // flags.batch_size
    
    for i in range(total_batch):
      batch_indices = indices[flags.batch_size*i : flags.batch_size*(i+1)]
      batch_xs = manager.get_images(batch_indices)

      reconstr_loss, latent_loss, summary_str, KL_list, = model.partial_fit(sess, batch_xs, iter)
      summary_writer.add_summary(summary_str, iter)
      iter += 1

      recon_epoch += reconstr_loss
      KL_epoch += latent_loss
      KL_list_epoch=KL_list

    print("e:", epoch, "recon_loss: ", recon_epoch/float(total_batch), "latent_loss: ", KL_epoch/float(total_batch),
          "loss:",recon_epoch/float(total_batch)+KL_epoch/float(total_batch))

    #print KL loss for each layer
    KL_string = ''
    for e_ind, e_KL in enumerate(KL_list_epoch):
      KL_string += str(e_ind) + ":" + str(e_KL) + ", "
    print(KL_string)

    #Save reconstruction for random images
    reconstruct_check_images = manager.get_random_images(10)
    reconstruct_check(sess, model, reconstruct_check_images)

    # Save checkpoint
    saver.save(sess, flags.checkpoint_dir + '/' + 'checkpoint', global_step = iter) #save for each epoch

  #After training, copy checkpoint to a new folder
  if not os.path.exists(os.path.join(flags.checkpoint_dir,str(flags.train_seq))):
    os.mkdir(os.path.join(flags.checkpoint_dir,str(flags.train_seq)))
  for file in glob.glob(os.path.join(flags.checkpoint_dir,"checkpoint*")):
    copyfile(file, os.path.join(flags.checkpoint_dir,str(flags.train_seq),os.path.basename(file)))

def reconstruct_check(sess, model, images):
  x_reconstruct = model.reconstruct(sess, images)

  if not os.path.exists("reconstr_img"):
    os.mkdir("reconstr_img")

  for i in range(10):
    try:
      org_img = images[i].reshape(image_shape)
    except:
      org_img = images[0][i].reshape(image_shape)
    org_img = org_img.astype(np.float32)
    reconstr_img = x_reconstruct[i].reshape(image_shape)
    imgs_comb = np.hstack((org_img, reconstr_img))
    imsave("reconstr_img/check_r{0}.png".format(i), imgs_comb)

def compute_mean_loss(sess,model,manager):
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

def disentangle_check_image_row(sess, model, manager, save_original=False, plot_flag=True, step=0,):
  if not os.path.exists("disentangle_img_row"):
    os.mkdir("disentangle_img_row")
  if not os.path.exists("disentangle_img"):
    os.mkdir("disentangle_img")
  img = manager.get_images([test_img_ind])
  if save_original:
    try:
      imsave("disentangle_img_row/original.png", img[0].reshape(image_shape).astype(np.float32))
    except:
      imsave("disentangle_img_row/original.png", img[0][0].reshape(image_shape).astype(np.float32))
    try:
      org_img = img[0].reshape(image_shape)
    except:
      org_img = img[0][0].reshape(image_shape)
    x_reconstruct = model.reconstruct(sess, img)
    reconstr_img=x_reconstruct[0].reshape(image_shape)
    imgs_comb = np.hstack((org_img, reconstr_img))
    imsave("disentangle_img/check_recon.png", imgs_comb)

  qz = model.inference(sess, img)
  code = model.inference_z(sess, img)
  select_dim = []
  samples_allz = []
  all_keys=sorted(qz.keys())
  print(all_keys)
  gif_nums = 8
  for key in all_keys:
    z_mean = qz[key][0][0]
    z_sigma_sq = np.exp(qz[key][1][0])

    for ind in range(len(z_sigma_sq)):
      if z_sigma_sq[ind] < 0.2:
        select_dim.append(key+"_"+str(ind))

    if plot_flag:
      n_z = z_mean.shape[0]
      for target_z_index in range(n_z):
        samples = []
        for ri in range(gif_nums + 1):
          maxi = 2.5
          value = -maxi + 2 * maxi / gif_nums * ri
          code2 = copy.deepcopy(code)
          for i in range(n_z):
            if i == target_z_index:
              code2[key][0][i] = value
            else:
              code2[key][0][i] = code[key][0][i]
          reconstr_img = model.generate(sess,code2)
          rimg = reconstr_img[0].reshape(image_shape)
          samples.append(rimg * 255)
        samples_allz.append(samples)
        imgs_comb = np.hstack((img for img in samples))
        imsave("disentangle_img_row/check_" + key + "_z{0}.png".format(target_z_index), imgs_comb)
        make_gif(samples, "disentangle_img/"+key+"_z_%s.gif" % (target_z_index), true_image=True)
  if plot_flag:
    final_gif = []
    for i in range(gif_nums):
      gif_samples = []
      for j in range(z_dim*len(qz.keys())):
        gif_samples.append(samples_allz[j][i])
      gif_samples.reverse() #now the order from left to right is high layer to low layer
      imgs_comb = np.hstack((img for img in gif_samples))
      final_gif.append(imgs_comb)
    make_gif(final_gif, "disentangle_img/all_z_step{0}.gif".format(step), true_image=True)
    make_gif(final_gif, flags.checkpoint_dir+"/all_z_step{0}.gif".format(step), true_image=True)

  return select_dim

def disentangle_layer_sample(sess, model, manager, save_original=True, step=1):
  if not os.path.exists("disentangle_img_row"):
    os.mkdir("disentangle_img_row")
  if not os.path.exists("disentangle_img"):
    os.mkdir("disentangle_img")
  img = manager.get_images([test_img_ind])
  if save_original:
    try:
      imsave("disentangle_img_row/original.png", img[0].reshape(image_shape).astype(np.float32))
    except:
      imsave("disentangle_img_row/original.png", img[0][0].reshape(image_shape).astype(np.float32))
    try:
      org_img = img[0].reshape(image_shape)
    except:
      org_img = img[0][0].reshape(image_shape)
    x_reconstruct = model.reconstruct(sess, img)
    reconstr_img=x_reconstruct[0].reshape(image_shape)
    imgs_comb = np.hstack((org_img, reconstr_img))
    imsave("disentangle_img/check_recon.png", imgs_comb)

  qz = model.inference(sess, img)
  code = model.inference_z(sess, img)

  for key in qz.keys():
    image_comb_row = []
    for img_i in range(8):
      samples = []
      for img_j in range(8):
        code2 = copy.deepcopy(code)
        code2[key]=np.random.normal(0,1,z_dim).reshape(1,z_dim)
        reconstr_img = model.generate(sess, code2)
        rimg = reconstr_img[0].reshape(image_shape)
        samples.append(rimg * 255)
      imgs_comb = np.hstack((img for img in samples))
      image_comb_row.append(imgs_comb)
    final_comb=np.vstack((img for img in image_comb_row))
    imsave(flags.checkpoint_dir+"/step_" + str(step) + "_disentangle_" + key + "_seed" + str(0) + ".png",final_comb)

def load_checkpoints(sess,model):
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

def main(argv):
  if "Celeb" in flags.checkpoint_dir:
    manager = DataManager_Celeb64()
  elif '3dshape' in flags.checkpoint_dir:
    manager = DataManager_3dshapes()
  else:
    print('set to default dataset: dsprite')
    manager = DataManager()

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=flags.gpu_usage)
  sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,gpu_options=gpu_options))
  model = VAE_ladder(z_dim=z_dim,beta=flags.beta,
              learning_rate=flags.learning_rate,flags=flags,chn_num=chn_num,train_seq=flags.train_seq,)

  sess.run(tf.global_variables_initializer())

  saver = load_checkpoints(sess,model)

  if flags.mode==1: #training for one step. Check "train_pro-VLAE.sh" for progressive training.
    manager.load()
    print("training data size:",manager.n_samples)
    train(sess, model, manager, saver)
    disentangle_layer_sample(sess, model, manager, step=flags.train_seq)
  elif flags.mode==2: #visulazation
    manager.load()
    print("training data size:", manager.n_samples)
    reconstruct_check_images = manager.get_random_images(10)
    reconstruct_check(sess, model, reconstruct_check_images)
    disentangle_layer_sample(sess, model, manager, step=flags.train_seq)
    disentangle_check_image_row(sess,model,manager)
    compute_mean_loss(sess, model, manager)
  elif flags.mode==3: #compute MIG and MIG-sup
    manager.load()
    if "3dshapes" in flags.checkpoint_dir:
      mutual_info_metric_3dshapes(sess, model, manager)
    else: #for dsprite
      mutual_info_metric_shapes(sess, model, manager)

if __name__ == '__main__':
  tf.app.run()