from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import lib.dist as dist
import torch
from torch.autograd import Variable

class VAE_ladder(object):
    def __init__(self,
                 z_dim=10,
                 beta=1.0,
                 learning_rate=5e-4,
                 fade_in_duration=5000,
                 flags=None,
                 chn_num=1,
                 train_seq=1,
                 image_size=64):
        self.flags=flags
        self.activation=tf.nn.leaky_relu
        self.z_dim = z_dim
        self.layer_num=4
        self.learning_rate = learning_rate
        self.beta=beta
        self.chn_num=chn_num
        self.fade_in_duration = fade_in_duration
        self.train_seq=train_seq
        self.image_size=image_size
        self.pre_KL=flags.KL
        self.fadein=flags.fadein

        self.q_dist = dist.Normal()
        self.x_dist = dist.Bernoulli()
        self.prior_dist = dist.Normal()
        self.prior_params = torch.zeros(self.z_dim, 2)

        self._create_network()
        self._create_loss_optimizer()

    def _get_prior_params(self, batch_size=1):
        expanded_size = (batch_size,) + self.prior_params.size()
        prior_params = Variable(self.prior_params.expand(expanded_size))
        return prior_params

    def _sample_z(self, z_mean, z_log_sigma_sq):
        eps_shape = tf.shape(z_mean)
        eps = tf.random_normal(eps_shape, 0, 1, dtype=tf.float32)
        # z = mu + sigma * epsilon
        z = tf.add(z_mean,
                   tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps))
        return z

    def _KL(self,z_mean,z_log_sigma_sq):
        latent_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq
                                           - tf.square(z_mean)
                                           - tf.exp(z_log_sigma_sq), 1)
        latent_loss = tf.reduce_mean(latent_loss)
        return latent_loss

    def fade_in_alpha(self, step):
        if step > self.fade_in_duration:
            a = 1.
        else:
            a = 1. * (step / self.fade_in_duration)
        return a

    def inference_h1(self, x, reuse=False):
        with tf.variable_scope("qh1", reuse=reuse) as scope:
            conv1 = tf.layers.batch_normalization(tf.layers.conv2d(x, 64, 4, strides=(2, 2), padding='same', activation=self.activation))
        return conv1

    def inference_h2(self, h1, reuse=False):
        with tf.variable_scope("qh2", reuse=reuse) as scope:
            conv1 = tf.layers.batch_normalization(tf.layers.conv2d(h1, 128, 4, strides=(2, 2), padding='same', activation=self.activation))
        return conv1

    def inference_h3(self, h2, reuse=False):
        with tf.variable_scope("qh3", reuse=reuse) as scope:
            conv1 = tf.layers.batch_normalization(tf.layers.conv2d(h2, 256, 4, strides=(2, 2), padding='same', activation=self.activation))
        return conv1

    def inference_h4(self, h3, reuse=False):
        with tf.variable_scope("qh4", reuse=reuse) as scope:
            conv1 = tf.layers.batch_normalization(tf.layers.conv2d(h3, 512, 4, strides=(2, 2), padding='same', activation=self.activation))
        return conv1

    def ladder1(self,h1,reuse=False):
        with tf.variable_scope("qladder1", reuse=reuse) as scope:
            if self.train_seq == 3 and self.fadein:
                h1 = h1 * self.fade_in
            conv1 = tf.layers.batch_normalization(tf.layers.conv2d(h1, 64, 4, strides=(2, 2), padding='same', activation=self.activation))
            conv1 = tf.layers.batch_normalization(tf.layers.conv2d(conv1, 64, 4, strides=(1, 1), padding='same', activation=self.activation))
            print('ladder1',conv1.shape)
            fc1 = tf.layers.flatten(conv1)
            z_mean = tf.layers.dense(fc1, self.z_dim)
            z_log_sigma_sq = tf.layers.dense(fc1, self.z_dim)
            return z_mean,z_log_sigma_sq

    def ladder2(self,h2,reuse=False):
        with tf.variable_scope("qladder2", reuse=reuse) as scope:
            if self.train_seq == 2 and self.fadein:
                h2=h2*self.fade_in
            conv1 = tf.layers.batch_normalization(tf.layers.conv2d(h2, 128, 4, strides=(2, 2), padding='same', activation=self.activation))
            conv1 = tf.layers.batch_normalization(tf.layers.conv2d(conv1, 256, 4, strides=(2, 2), padding='same', activation=self.activation))
            print('ladder2', conv1.shape)
            fc1 = tf.layers.flatten(conv1)
            z_mean = tf.layers.dense(fc1, self.z_dim)
            z_log_sigma_sq = tf.layers.dense(fc1, self.z_dim)
            return z_mean,z_log_sigma_sq

    def ladder3(self,h3,reuse=False):
        with tf.variable_scope("qladder3", reuse=reuse) as scope:
            conv1 = tf.layers.batch_normalization(tf.layers.conv2d(h3, 256, 4, strides=(2, 2), padding='same', activation=self.activation))
            conv1 = tf.layers.batch_normalization(tf.layers.conv2d(conv1, 512, 4, strides=(2, 2), padding='same', activation=self.activation))
            print('ladder3', conv1.shape)
            fc1 = tf.layers.flatten(conv1)
            z_mean = tf.layers.dense(fc1, self.z_dim)
            z_log_sigma_sq = tf.layers.dense(fc1, self.z_dim)
            return z_mean,z_log_sigma_sq

    def ladder4(self,h4,reuse=False):
        with tf.variable_scope("qladder4", reuse=reuse) as scope:
            print('ladder4', h4.shape)
            fc1 = tf.layers.batch_normalization(tf.layers.dense(tf.layers.flatten(h4), 1024, activation=self.activation))
            fc2 = tf.layers.batch_normalization(tf.layers.dense(fc1, 1024, activation=self.activation))
            z_mean = tf.layers.dense(fc2, self.z_dim)
            z_log_sigma_sq = tf.layers.dense(fc2, self.z_dim)
            return z_mean,z_log_sigma_sq

    def generative4(self,z4_sample,reuse=False):
        with tf.variable_scope("gen4", reuse=reuse) as scope:
            fc1 = tf.layers.batch_normalization(tf.layers.dense(z4_sample, 1024, activation=self.activation))
            fc1 = tf.layers.batch_normalization(tf.layers.dense(fc1, 1024, activation=self.activation))
            fc2 = tf.layers.batch_normalization(tf.layers.dense(fc1, 4 * 4 * 512, activation=self.activation))
            fc2_reshaped = tf.reshape(fc2, [-1, 4, 4, 512])
            return fc2_reshaped

    def generative3(self,z3_sample,g4,reuse=False):
        with tf.variable_scope("gen3", reuse=reuse) as scope:
            fc2 = tf.layers.batch_normalization(tf.layers.dense(z3_sample, 4 * 4 * 512, activation=self.activation))
            fc2_reshaped = tf.reshape(fc2, [-1, 4, 4, 512])
            if self.train_seq==2 and self.fadein:
                fc2_reshaped=fc2_reshaped*self.fade_in
            if self.train_seq<2:
                fc2_reshaped = fc2_reshaped * 0.
            fc2_reshaped = tf.concat(values=[fc2_reshaped, g4], axis=len(fc2_reshaped.get_shape()) - 1)
            deconv1 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(fc2_reshaped, 512, 4, strides=(1, 1), padding='same',
                                                 activation=self.activation))
            deconv1 = tf.layers.batch_normalization(
                tf.layers.conv2d_transpose(deconv1, 256, 4, strides=(2, 2), padding='same',activation=self.activation))
            return deconv1

    def generative2(self,z2_sample,g3,block_z=False,reuse=False):
        with tf.variable_scope("gen2", reuse=reuse) as scope:
            fc2 = tf.layers.batch_normalization(tf.layers.dense(z2_sample, 8 * 8 * 256, activation=self.activation))
            fc2_reshaped = tf.reshape(fc2, [-1, 8, 8, 256])
            if self.train_seq==3 and self.fadein:
                fc2_reshaped=fc2_reshaped*self.fade_in
            if self.train_seq<3 or block_z:
                fc2_reshaped = fc2_reshaped * 0.
            fc2_reshaped = tf.concat(values=[fc2_reshaped, g3], axis=len(fc2_reshaped.get_shape()) - 1)
            deconv1 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(fc2_reshaped, 128, 4, strides=(2, 2), padding='same',
                                                 activation=self.activation))
            deconv1 = tf.layers.batch_normalization(
                tf.layers.conv2d_transpose(deconv1, 64, 4, strides=(1, 1), padding='same',activation=self.activation))
            return deconv1

    def generative1(self,z1_sample,g2,block_z=False,reuse=False):
        with tf.variable_scope("gen1", reuse=reuse) as scope:
            fc2 = tf.layers.dense(z1_sample, 16 * 16 * 64, activation=self.activation)
            fc2_reshaped = tf.reshape(fc2, [-1, 16, 16, 64])
            if self.train_seq==4 and self.fadein:
                fc2_reshaped = fc2_reshaped * self.fade_in
            if self.train_seq<4 or block_z:
                fc2_reshaped = fc2_reshaped*0.
            fc2_reshaped = tf.concat(values=[fc2_reshaped, g2], axis=len(fc2_reshaped.get_shape()) - 1)
            deconv1 = tf.layers.conv2d_transpose(fc2_reshaped, 64, 4, strides=(2, 2), padding='same',
                                                 activation=self.activation)  # 32*32,64
            return deconv1

    def generative0(self,g1,reuse=False):
        with tf.variable_scope("gen0", reuse=reuse) as scope:
            deconv5 = tf.layers.conv2d_transpose(g1, self.chn_num, 4, strides=(2, 2), padding='same', activation=None)
            return deconv5

    def _create_network(self):
        self.x = tf.placeholder(tf.float32, shape=[None, self.image_size,self.image_size,self.chn_num])
        self.fade_in = tf.placeholder(tf.float32,shape=[])
        with tf.variable_scope("vae_ladder"):
            #inference
            self.h1 = self.inference_h1(self.x)
            print('h1',self.h1.shape)
            self.h2 = self.inference_h2(self.h1)
            print('h2', self.h2.shape)
            self.h3 = self.inference_h3(self.h2)
            print('h3', self.h3.shape)
            self.h4 = self.inference_h4(self.h3)
            print('h4', self.h4.shape)

            self.z_mean4, self.z_log_sigma_sq4 = self.ladder4(self.h4)
            self.z_sample4 = self._sample_z(self.z_mean4, self.z_log_sigma_sq4)
            self.z_log_sigma_sq4 = tf.clip_by_value(self.z_log_sigma_sq4, -1e2, 3)

            self.z_mean3, self.z_log_sigma_sq3 = self.ladder3(self.h3)
            self.z_sample3 = self._sample_z(self.z_mean3, self.z_log_sigma_sq3)
            self.z_log_sigma_sq3 = tf.clip_by_value(self.z_log_sigma_sq3, -1e2, 3)

            self.z_mean2, self.z_log_sigma_sq2 = self.ladder2(self.h2)
            self.z_sample2 = self._sample_z(self.z_mean2, self.z_log_sigma_sq2)
            self.z_log_sigma_sq2= tf.clip_by_value(self.z_log_sigma_sq2, -1e2, 3)


            self.z_mean1, self.z_log_sigma_sq1 = self.ladder1(self.h1)
            self.z_sample1 = self._sample_z(self.z_mean1, self.z_log_sigma_sq1)
            self.z_log_sigma_sq1 = tf.clip_by_value(self.z_log_sigma_sq1, -1e2, 3)

            #gneration
            self.g4 = self.generative4(self.z_sample4)
            print('g4', self.g4.shape)

            self.g3 = self.generative3(self.z_sample3,self.g4)
            print('g3',self.g3.shape)

            self.g2 = self.generative2(self.z_sample2,self.g3)
            print('g2', self.g2.shape)

            self.g1 = self.generative1(self.z_sample1, self.g2)
            print('g1', self.g1.shape)

            self.x_out_logit = self.generative0(self.g1)
            print('x_out_logit', self.x_out_logit.shape)

            self.x_out = tf.nn.sigmoid(self.x_out_logit)

            #store latent variables
            self.ladders = {}
            self.ladders['ladder1'] = [self.z_sample1, self.z_mean1, self.z_log_sigma_sq1]
            self.ladders['ladder2'] = [self.z_sample2, self.z_mean2, self.z_log_sigma_sq2]
            self.ladders['ladder3'] = [self.z_sample3, self.z_mean3, self.z_log_sigma_sq3]
            self.ladders['ladder4'] = [self.z_sample4, self.z_mean4, self.z_log_sigma_sq4]

    def _create_loss_optimizer(self):
        # Reconstruction loss
        self.x_recon = self.x
        reconstr_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.x_recon,
                                                                logits=self.x_out_logit)
        reconstr_loss = tf.reduce_sum(reconstr_loss, [1,2,3],)
        self.reconstr_loss = tf.reduce_mean(reconstr_loss)

        # Latent loss
        self.latent_loss1 = self._KL(self.z_mean1, self.z_log_sigma_sq1)
        self.latent_loss2 = self._KL(self.z_mean2, self.z_log_sigma_sq2)
        self.latent_loss3 = self._KL(self.z_mean3, self.z_log_sigma_sq3)
        self.latent_loss4 = self._KL(self.z_mean4, self.z_log_sigma_sq4)

        self.latent_loss=self.latent_loss1+self.latent_loss2+self.latent_loss3+self.latent_loss4

        # summary
        reconstr_loss_summary_op = tf.summary.scalar('reconstr_loss', self.reconstr_loss)
        latent_loss_summary_op = tf.summary.scalar('latent_loss', self.latent_loss)
        self.summary_op = tf.summary.merge([reconstr_loss_summary_op, latent_loss_summary_op])

        self.loss = self.reconstr_loss
        coff=self.flags.coff

        if self.train_seq==1:
            self.loss += self.beta * self.latent_loss4
            if self.pre_KL:
                self.loss+=(self.latent_loss3+self.latent_loss2+self.latent_loss1)*coff
        elif self.train_seq==2:
            self.loss+= self.beta * (self.latent_loss4+self.latent_loss3)
            if self.pre_KL:
                self.loss += (self.latent_loss2+ self.latent_loss1)*coff
        elif self.train_seq == 3:
            self.loss+= self.beta * (self.latent_loss4+self.latent_loss3+self.latent_loss2)
            if self.pre_KL:
                self.loss+= self.latent_loss1*coff
        elif self.train_seq == 4:
            self.loss+= self.beta * (self.latent_loss4+self.latent_loss3+self.latent_loss2+self.latent_loss1)

        self.KL_list = [self.latent_loss1, self.latent_loss2, self.latent_loss3,self.latent_loss4,self.latent_loss]

        t_vars = tf.trainable_variables()

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)


    def partial_fit(self, sess, xs, iter):
        a = self.fade_in_alpha(iter)

        _, reconstr_loss, latent_loss, summary_str, KL_list = sess.run((self.train_op,
                                                               self.reconstr_loss,
                                                               self.latent_loss,
                                                               self.summary_op,
                                                               self.KL_list,
                                                               ),
                                                              feed_dict={
                                                                  self.x: xs,
                                                                  self.fade_in:a,
                                                              })

        return reconstr_loss, latent_loss, summary_str, KL_list,

    def reconstruct(self, sess, xs):
        return sess.run(self.x_out,
                        feed_dict={self.x: xs,self.fade_in:1.})

    def transform(self, sess, xs, zlayer=0):
        m1,log_var1,m2,log_var2,m3,log_var3,m4,log_var4=sess.run([self.z_mean1, self.z_log_sigma_sq1,
                              self.z_mean2, self.z_log_sigma_sq2,
                              self.z_mean3, self.z_log_sigma_sq3,
                              self.z_mean4, self.z_log_sigma_sq4],
                        feed_dict={self.x: xs,self.fade_in:1.})
        if zlayer==0:
            return np.concatenate([m1,m2,m3,m4],axis=1),np.concatenate([log_var1,log_var2,log_var3,log_var4],axis=1)
        elif zlayer==4:
            return m4, log_var4
        elif zlayer==3:
            return m3, log_var3
        elif zlayer==2:
            return m2, log_var2
        else:
            return m1, log_var1

    def inference(self, sess, xs):
        tensor_handle = [self.ladders[key][1:] for key in self.ladders]
        tensor_value = sess.run(tensor_handle,
                                feed_dict={self.x: xs,self.fade_in:1.})
        return {name: value for name, value in zip(self.ladders, tensor_value)}

    def inference_z(self, sess, xs):
        tensor_handle = [self.ladders[key][0] for key in self.ladders]
        tensor_value = sess.run(tensor_handle,
                            feed_dict={self.x: xs,self.fade_in:1.})
        return {name: value for name, value in zip(self.ladders, tensor_value)}

    def generate(self, sess, codes):
        return sess.run(self.x_out,
                        feed_dict={self.z_sample1: codes['ladder1'],
                                   self.z_sample2: codes['ladder2'],
                                   self.z_sample3: codes['ladder3'],
                                   self.z_sample4: codes['ladder4'],
                                   self.fade_in: 1.
                                   })

    def get_recons_loss(self, sess, xs):
        reconstr_loss, latent_loss = sess.run((self.reconstr_loss, self.latent_loss,),
                                              feed_dict={
                                                  self.x: xs,
                                                  self.fade_in: 1.
                                              })
        return reconstr_loss, latent_loss