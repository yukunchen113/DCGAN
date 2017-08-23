import tensorflow as tf 
import numpy as np 
import os
from PIL import Image
import math

def variable(name, shape, initializer, trainable = True, dtype = tf.float32):
	out = tf.get_variable(name, shape, dtype, initializer, trainable = True)
	return out

def reshape(input, out_shape, n_in, scope = 'reshape'):
	with tf.variable_scope(scope):
		n_out = reduce(lambda x,y: x*y, out_shape[1:])
		out= tf.reshape(input, [-1, n_in])
		stddev = 0.02#math.sqrt(n_in)

		initializer = tf.random_normal_initializer(stddev=stddev)
		weights = variable('weights', [n_in,n_out], initializer)
		out = tf.matmul(out, weights)
		out = tf.reshape(out,out_shape)
		return out
def lrelu(input, const=None):
	if const is None:
		const = 0.2
	const = tf.cast(const,input.dtype)
	out = tf.maximum(input, tf.multiply(input, const))
	return out

def conv2d(input, n_out, k, s, scope = 'conv2d', padding ='SAME'):
	with tf.variable_scope(scope):
		n_in = input.get_shape()[-1].value
		stddev = 0.02#math.sqrt(2.0/(n_in*k*k))

		initializer = tf.random_normal_initializer(stddev=stddev)
		kernel = variable('weights', [k,k,n_in,n_out], initializer)
		strides = [1,s,s,1]
		out = tf.nn.conv2d(input, kernel, strides, padding)
		return out

def tconv2d(input, out_shape, k, s, scope = 'conv2d_transpose', padding ='SAME'):
	with tf.variable_scope(scope):
		n_in = input.get_shape()[-1].value
		n_out = out_shape[-1]
		stddev = 0.02#math.sqrt(2.0/(n_in*k*k))

		initializer = tf.random_normal_initializer(stddev=stddev)
		kernel = variable('weights', [k,k,n_out,n_in], initializer)
		strides = [1,s,s,1]
		out = tf.nn.conv2d_transpose(input, kernel,out_shape, strides, padding)
		return out

def batch_normalization(input, scope='batch_normalization'):
	with tf.variable_scope(scope):
		n_in = input.get_shape()[-1].value
		beta = variable('biases',[n_in],tf.constant_initializer(0.0))
		gamma = variable('weights',[n_in],tf.constant_initializer(1.0))
		variance_epsilon = 1e-8
		cur_mean, cur_var = tf.nn.moments(input,[0,1,2]) 
		out = tf.nn.batch_normalization(input, cur_mean,cur_var,beta,gamma,variance_epsilon)
		return out

def fully_connected(input, n_out, activation = True, scope = 'fully_connected'):
	with tf.variable_scope(scope):
		n_in = reduce(lambda x,y: x*y, input.get_shape().as_list()[1:])
		out = tf.reshape(input,[-1,n_in])
		stddev = 0.02#math.sqrt(2.0/n_in)

		initializer = tf.random_normal_initializer(stddev=stddev)
		weights = variable('weights',[n_in,n_out],initializer)
		biases = variable('biases',[n_out],tf.constant_initializer(0.01))
		out = tf.add(tf.matmul(out, weights),biases)
		if activation:
			out = lrelu(out)
		return out

def make_label(batch_size, stddev, isones=True):
		labels = tf.random_normal([batch_size],1,stddev,tf.float32)
		labels = tf.clip_by_value(labels,0,1)
		op_labels = tf.subtract(tf.ones([batch_size],tf.float32),labels)
		if not stddev:
			labels = tf.ones([batch_size],tf.float32)
			op_labels = tf.zeros([batch_size],tf.float32)
		labels = tf.reshape(labels, [batch_size,1])
		op_labels = tf.reshape(op_labels,[batch_size,1])
		if isones:
			return tf.concat([labels,op_labels],1)
		return tf.concat([op_labels,labels],1)

def make_sample(batch_size, sample_size, sample=None):
	if sample is None:
		sample = tf.truncated_normal([batch_size, sample_size], stddev=0.5)
	else: 
		if not np.asarray(sample).shape == (batch_size, sample_size):
			raise TypeError('sample must be shape, [batch_size, sample_size]')
		sample = tf.cast(sample, tf.float32)
	return sample

def timer(total_seconds):
	ts = total_seconds%60
	tm = (total_seconds/60)%60
	th = (total_seconds/3600)%24
	td = (total_seconds/864000)%7
	tw = total_seconds/604800
	time_list = [tw,td,th,tm,ts]
	unit_list = ['w','d','h','m','s']
	string = ''
	for i in range(len(time_list)):
		if int(time_list[i]) or unit_list[i] == 's':
			string += '%d'%time_list[i] + unit_list[i] +' '
	return string
	
def make_info(i,gl,dl,spb,td):
	string = 'step:%d, generator loss:%.3f, discriminator loss:%.3f, seconds per batch:%f, '
	return string%(i,gl,dl,spb) + 'total run time: ' + td