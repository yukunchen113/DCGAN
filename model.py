import tensorflow as tf 
import numpy as np 
import os
from PIL import Image
import math

import util as ut
'''
params = [#model has to be retrained if any of these are changed
	batch_size,#size of batch----0
	sample_size,#size of sample noise-1
	image_size,#size of the generated image----2
	n_classes,#number of classes for discriminator, should be 2 ---3
	n_channels,#number of colour channels----4
	n_dis_layers#number of layers for the discriminator----5
	]
'''
#default params (for mnist) = [128,100,28,2,1,3]

class dcgan():
	def __init__(self, images, params, istrain=False):
		'''
		Args: 
			images: batch of images, real images, uint8 array of size 
				[batch_size, height, width, channels]
			params: list of important values
			sample: takes in list of size [sample_size], sample noise to generate image from
			istrain: sets model to training state, type, bool
		'''
		self.pm = params
		batch_size = params[0]
		#--scaling of images to -1 and 1
		images = tf.cast(images, tf.float32)
		offset_value = tf.cast(128, tf.float32)
		normalize_value = tf.cast(130, tf.float32)
		images = tf.divide(tf.subtract(images, offset_value),normalize_value)
		self.images = images
		self.istrain = istrain
		self.gen_labels = tf.zeros([batch_size],dtype = tf.int64)
		self.labels = tf.ones([batch_size],dtype = tf.int64)

	def get_sample(self, sample=None):
		sample_size = self.pm[1]
		if sample is None:
			sample = tf.truncated_normal([sample_size], stddev=1.0)
		else: 
			if not len(sample) == sample_size:
				print 'sample must be sample size'
			sample = tf.cast(sample, tf.float32)
		self.sample = sample

	def generator(self, scope = 'generator',reuse = True,sample = None):
		'''
		Generates image, from sample
		'''
		with tf.variable_scope(scope,reuse=reuse):
			batch_size = self.pm[0]
			image_size = self.pm[2]
			self.get_sample(sample)
			i = image_size
			n_gen_layers = 0
			while not i%2:
				n_gen_layers+=1
				i/=2
				if n_gen_layers == 5:
					break
			start_size = image_size/(2**(n_gen_layers-1))
			n_channels = self.pm[4]
			#-reshape sample------------------------
			if image_size % 2**(n_gen_layers-1):
				print 'please pick a image_size that is divisible by %d'%(2**(n_gen_layers-1))
				print image_size
			out = ut.reshape(
				self.sample,
				[batch_size, start_size,start_size,2**(n_gen_layers+5)],
				'reshape1')
			out = ut.batch_normalization(out,'batch_normalization1')
			out = tf.nn.relu(out)
			#-Conv layers---------------------------
			new_size = start_size
			new_depth = out.get_shape()[-1].value
			for i in range(1,n_gen_layers):
				new_size *= 2
				new_depth /= 2
				activation = tf.nn.relu

				if i == n_gen_layers-1:
					new_depth = n_channels
					activation = tf.tanh
				out = ut.tconv2d(out, [batch_size,new_size,new_size,new_depth],5,2,
					'conv2d_transpose%d'%i)
				if not i == n_gen_layers-1:
					out = ut.batch_normalization(out, 'batch_normalization%d'%(i+1))
				out = activation(out)
			if sample is None:
				self.generator_output = out
			else:
				return out

	def discriminator(self, isgen, scope='discriminator',reuse = True):
		'''
		gen_im: bool, see if use generator inputs or not
		'''
		with tf.variable_scope(scope,reuse=reuse):
			batch_size = self.pm[0]
			image_size = self.pm[2]
			n_dis_layers = self.pm[5]
			n_classes = self.pm[3]
			self.isgen = isgen
			if isgen:
				images = self.generator_output
			else:
				images = self.images
			#---convolutional Layers-----
			new_size = image_size
			for i in range(n_dis_layers-1):
				out = ut.conv2d(images, image_size*2,3,2,'conv2d%d'%(i+1))
				out = ut.batch_normalization(out,'batch_normalization%d'%(i+1))
				out = ut.lrelu(out)
			#--fully connected layer---
				#should be changed, there are better methods.
			out = ut.fully_connected(out,1000,scope = 'fully_connected1')
			out = ut.fully_connected(out,n_classes,False,'fully_connected2')
			self.discriminator_output = out

	def loss(self):
		self.discriminator(False, reuse = False)
		logits = self.discriminator_output
		labels = self.labels
		loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
		self.generator(reuse = False)
		self.discriminator(True)
		logits = self.discriminator_output
		labels = self.gen_labels
		loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
		self.dis_loss = loss1+loss2
		labels = self.labels
		self.gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
		
	def train(self, global_step):
		self.loss()
		gen_var = [i for i in tf.trainable_variables() if i.name.startswith('g')]
		dis_var = [i for i in tf.trainable_variables() if i.name.startswith('d')]
		gen_opt = tf.train.AdamOptimizer(learning_rate = 0.0002,
			beta1=0.5).minimize(self.gen_loss,var_list=gen_var,global_step=global_step)
		dis_opt = tf.train.AdamOptimizer(learning_rate = 0.0002,
			beta1=0.5).minimize(self.dis_loss,var_list=dis_var,global_step=global_step)
		with tf.control_dependencies([gen_opt,dis_opt]):
			self.train_opt = tf.no_op(name='train')