import tensorflow as tf 
import numpy as np 
import os
from PIL import Image
import math

def variable(name, shape, initializer, trainable = True, dtype = tf.float32):
	out = tf.get_variable(name, shape, dtype, initializer, trainable = True)
	return out

def reshape(input, out_shape, scope = 'reshape'):
	with tf.variable_scope(scope):
		n_in = reduce(lambda x,y: x*y, input.get_shape().as_list())
			#gets total input size
		n_out = reduce(lambda x,y: x*y, out_shape)
		input.set_shape([n_in])
		stddev = math.sqrt(n_in)
		initializer = tf.truncated_normal_initializer(stddev=stddev)
		weights = variable('weights', [n_in,n_out], initializer)
		out = tf.matmul([input], weights)
		out = tf.reshape(out,out_shape)
		return out
def lrelu(input, const=None):
	if const is None:
		const = 0.02
	const = tf.cast(const,input.dtype)
	out = tf.maximum(input, tf.multiply(input, const))
	return out
def conv2d(input, n_out, k, s, scope = 'conv2d', padding ='SAME'):
	with tf.variable_scope(scope):
		n_in = input.get_shape()[-1].value
		stddev = math.sqrt(2.0/(n_in*k*k))
		initializer = tf.truncated_normal_initializer(stddev=stddev)
		kernel = variable('weights', [k,k,n_in,n_out], initializer)
		strides = [1,s,s,1]
		out = tf.nn.conv2d(input, kernel, strides, padding)
		return out

def tconv2d(input, out_shape, k, s, scope = 'conv2d_transpose', padding ='SAME'):
	with tf.variable_scope(scope):
		n_in = input.get_shape()[-1].value
		n_out = out_shape[-1]
		stddev = math.sqrt(2.0/(n_in*k*k))
		initializer = tf.truncated_normal_initializer(stddev=stddev)
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
		stddev = math.sqrt(2.0/n_in)
		initializer = tf.truncated_normal_initializer(stddev=stddev)
		weights = variable('weights',[n_in,n_out],initializer)
		biases = variable('biases',[n_out],tf.constant_initializer(0.01))
		out = tf.add(tf.matmul(out, weights),biases)
		if activation:
			out = lrelu(out)
		return out
