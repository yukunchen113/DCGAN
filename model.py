import tensorflow as tf 
import numpy as np 
import os
import util as ut
import matplotlib.pyplot as plt 

class dcgan():
	def __init__(self, images, params, test_sample=None):
		self.pm = params
		images = tf.cast(images, tf.float32)
		images = tf.image.resize_images(images, [self.pm['image_resize'], self.pm['image_resize']])
		if not self.pm['Sets'] =='MNIST':
		#	images = images*2-1.0
		#else:
			images = images*2.0/255.0-1.0
		self.real_images = images
		self.test_sample = ut.make_sample(self.pm['batch_size'],
			self.pm['sample_size'])

	def generator(self, scope='generator',reuse=True,test_sample=False):
		with tf.variable_scope(scope, reuse=reuse):
			# get/calculate parameters-----
			generator_params = self.pm['gen_params']
			batch_size =self.pm['batch_size']
			sample_size =self.pm['sample_size']
			gen_conv_params = generator_params[1]
			n_gen_conv_layers = len(gen_conv_params) + 1
			fm_length = self.pm['image_resize']/2**(n_gen_conv_layers)
			#get sample-----------
			test_sample = tf.cast(test_sample,tf.bool)
			self.sample = tf.cond(test_sample, lambda:self.test_sample, 
				lambda: ut.make_sample(batch_size,sample_size))
			#reshape sample-------
			out = ut.reshape(self.sample,
				[batch_size,fm_length,fm_length,generator_params[0][-1]],
				self.pm['sample_size'],'reshape0')
			#convoutional layers-----
			for i in range(n_gen_conv_layers - 1):
				if not fm_length > self.pm['image_resize']:	
					fm_length*=2
				current_gen_conv_params = gen_conv_params[i]
				out = ut.tconv2d(out,
					[batch_size,fm_length,fm_length,
					current_gen_conv_params[-1]],
					current_gen_conv_params[0],
					current_gen_conv_params[1],
					'conv2d_transpose%d'%i)
				
				out = ut.batch_normalization(out,'batch_norm%d'%i)
				out = tf.nn.relu(out)
			#last convolutional layer (makes the image)----
			current_gen_conv_params = generator_params[-1]
			out = ut.tconv2d(out,
				[batch_size,self.pm['image_resize'],self.pm['image_resize'],
				self.pm['n_channels']],
				current_gen_conv_params[0],
				current_gen_conv_params[1],
				'final_conv2d_transpose')
			out = tf.tanh(out)
			return out

	def discriminator(self,images,scope='discriminator',reuse=True):
		with tf.variable_scope(scope, reuse=reuse):
			#get/calculate parameters:
			discriminator_params = self.pm['dis_params']
			n_dis_conv_layers = len(discriminator_params)
			image_size = self.pm['image_resize']
			#determine if use generated images------
			self.images= images
			out = images
			#build allcnn with leaky relu-----------
			for i in range(n_dis_conv_layers):
				# get individual conv layer parameters-----
				current_dis_conv_params = discriminator_params[i]
				if i >= n_dis_conv_layers-1:
					depth = self.pm['n_classes']
				else:
					depth = current_dis_conv_params[2]
				#make conv layers-----------------------
				out = ut.conv2d(out, depth, current_dis_conv_params[0],
					current_dis_conv_params[1],'conv2d%d'%i)
				if i:
					out = ut.batch_normalization(out, 'batch_norm%d'%i)
				out = ut.lrelu(out,self.pm['lrelu_const'])
				#height, width = out.get_shape().as_list()[1:-1]
			out = tf.reduce_mean(out, [1,2])
			#print out.get_shape().as_list()
			#out = ut.fully_connected(out,self.pm['n_classes'],False)
			'''
			out = tf.nn.avg_pool(out,[1,height,width,1],
				[1,image_size,image_size,1],'SAME')
			out = tf.squeeze(out,[1,2])
			'''
			return out

	def loss(self):
		#get/calculate parameters:
		batch_size = self.pm['batch_size']
		#calculate loss:
		self.rilogits = self.discriminator(self.real_images, reuse= False)
		self.gilogits = self.discriminator(self.generator(reuse=False))

		self.rilabels = ut.make_label(batch_size,self.pm['label_stddev'])
		loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
			logits=self.rilogits, labels=self.rilabels))
		self.gilabels = ut.make_label(batch_size,self.pm['label_stddev'],False)
		loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
			logits=self.gilogits, labels=self.gilabels))
		self.dis_loss =loss1+loss2
		self.gilabels2 = ut.make_label(batch_size,self.pm['label_stddev'])
		self.gen_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
			logits=self.gilogits, labels=self.gilabels2))


	def train(self, global_step):
		#get/calculate parameters:
		train_params = self.pm['opt_params']
		dlr, dbeta1 = train_params[0]
		glr, gbeta1 = train_params[1]
		#train model:
		self.loss()
		gen_var = [var for var in tf.trainable_variables() if var.name.startswith('g')]
		dis_var = [var for var in tf.trainable_variables() if var.name.startswith('d')]
		gen_opt = tf.train.AdamOptimizer(learning_rate = glr,
			 beta1=gbeta1).minimize(self.gen_loss,var_list=gen_var,global_step=global_step)
		dis_opt = tf.train.AdamOptimizer(learning_rate = dlr,
			 beta1=dbeta1).minimize(self.dis_loss,var_list=dis_var,global_step=global_step)
		with tf.control_dependencies([gen_opt,dis_opt]):
			self.train_opt = tf.no_op(name='train')