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
    dataset_zip = np.load('data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', #"path/to/dsprite"
                          encoding='latin1')
    self.imgs = dataset_zip['imgs']
    self.n_samples = len(self.imgs)

  @property
  def sample_size(self):
    return self.n_samples

  def get_images(self, indices):
    images = []
    for index in indices:
      img = self.imgs[index]
      img = img.reshape(64,64,1)
      images.append(img)
    return images

  def get_random_images(self, size):
    indices = [np.random.randint(self.n_samples) for i in range(size)]
    return self.get_images(indices)


class DataManager_Celeb64(object):
  def load(self):
    self.img_folder = "data/celebA" #"path/to/celebA"
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
  def load(self):
    import h5py
    dataset = h5py.File('data/3dshapes.h5', 'r') #"path/to/3Dshapes"
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
