from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import glob
import os

from PIL import Image

class DataManager(object):
  def load(self):
    # Load dataset
    dataset_zip = np.load('/home/zl7904/Documents/projects/hierarchical/disentangled_vae-master/data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz',
                          encoding='latin1')

    self.imgs = dataset_zip['imgs']
    self.latents_values = dataset_zip['latents_values']
    latents_classes = dataset_zip['latents_classes']
    metadata = dataset_zip['metadata'][()]

    latents_sizes = metadata['latents_sizes']
    self.n_samples = latents_sizes[::-1].cumprod()[-1]

    self.latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],
                                         np.array([1, ])))

    for j in range(1,self.latents_values.shape[1]):
      v=self.latents_values[:,j]
      self.latents_values[:, j] = ((v - v.min()) / (v.max() - v.min())-0.5)*6.
    self.latents_values=self.latents_values[:,1:]

  @property
  def sample_size(self):
    return self.n_samples

  def get_image(self, shape=0, scale=0, orientation=0, x=0, y=0):
    latents = [0, shape, scale, orientation, x, y]
    index = np.dot(latents, self.latents_bases).astype(int)
    return self.get_images([index])[0]

  def get_images(self, indices):
    images = []
    for index in indices:
      img = self.imgs[index]
      img = img.reshape(64,64,1)
      images.append(img)
    return images

  def get_images_with_label(self, indices):
    images = []
    labels = []
    for index in indices:
      img = self.imgs[index]
      label = self.latents_values[index]
      img = img.reshape(64,64,1)
      images.append(img)
      labels.append(label)
    return images,labels

  def get_random_images(self, size,with_label=False):
    indices = [np.random.randint(self.n_samples) for i in range(size)]
    if not with_label:
      return self.get_images(indices)
    else:
      return self.get_images_with_label(indices)


class DataManager_Celeb64(object):
  def load(self):
    self.img_folder = "/home/zl7904/Documents/Data/celebA" #"path/to/celebA"
    self.all_path_list = glob.glob(os.path.join(self.img_folder,"*.png"))
    self.imgs = None
    self.n_samples = len(self.all_path_list)
  @property
  def sample_size(self):
    return self.n_samples

  def get_image(self, index=0):
    image_name = self.img_folder+'/{:06d}.png'.format(index + 1)
    image = Image.open(image_name).crop((15, 45, 163, 193)).resize((64,64),Image.ANTIALIAS) #crop and resize
    return np.array(image)

  def get_images(self, indices):
    images = []
    for index in indices:
      img = self.get_image(index)
      img = img/255.
      images.append(img)
    return images

  def get_random_images(self, size):
    indices = [np.random.randint(self.n_samples) for i in range(size)]
    return self.get_images(indices)

class DataManager_3dshapes(object):
  #3dshapes dataset: https://github.com/deepmind/3d-shapes
  def load(self):
    import h5py
    dataset = h5py.File('/home/zl7904/Documents/Data/3dshapes/3dshapes.h5', 'r')
    images = dataset['images']
    self.imgs=np.array(images)
    self.n_samples = self.imgs.shape[0]

  @property
  def sample_size(self):
    return self.n_samples

  def get_image(self, index=0):
    return self.imgs[index]

  def get_images(self, indices):
    images = np.zeros([len(indices),64,64,3])
    count=0
    for index in indices:
      img = self.imgs[index]
      img = img/255.
      images[count]=img
      count+=1
    return images

  def get_random_images(self, size):
    indices = [np.random.randint(self.n_samples) for i in range(size)]
    return self.get_images(indices)