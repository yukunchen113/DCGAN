import tensorflow as tf 
import numpy as np 
import os
import math
import model as mp
import cifar10_input as ip 
import time
import cPickle as pickle
from PIL import Image
import shutil
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("Software/Python/MyCode/DCGAN/Data/",one_hot = True)
write_frequency = 10
Sets = ip.Sets
#Sets = 'MNIST'
last_step =100000
train_dir = '/media/yukun/Barracuda Hard Drive 2TB/Data/DCGAN/'+ Sets +'/ModelCheckpoint'
main_dir ='/media/yukun/Barracuda Hard Drive 2TB/Data/DCGAN/'+ Sets
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
		if int(time_list[i]):
			string += '%d'%time_list[i] + unit_list[i] +' '
	return string
def print_string(i,gl,dl,spb,td):
	string = 'step:%d, generator loss:%.3f, discriminator loss:%.3f, seconds per batch:%f, '
	print string%(i,gl,dl,spb) + 'total run time: ' + td

def train_data(params = [128,50,32,2,3,3]):
	global_step = tf.contrib.framework.get_or_create_global_step()
	batch_size = params[0]
	if not Sets == 'MNIST':
		images = ip.input(batch_size) 
	else:
	#----for mnist
		with tf.variable_scope('input'):
			x = tf.placeholder(tf.float32, [batch_size, params[2]*params[2]])
			images = tf.reshape(x, [-1,params[2],params[2],1])
	#--------
	model = mp.dcgan(images, params, istrain = True)
	model.train(global_step)
	class log_hook(tf.train.SessionRunHook):
		def begin(self):
			self.time = time.time()
			self.total_time = time.time()
			self.step = -1
			self.step_points = 1
			self.sample_images = []
		def before_run(self, run_context):
			self.step += 1
			mnist_batch,_ = mnist.train.next_batch(params[0])
			if Sets == 'MNIST':
				return tf.train.SessionRunArgs([model.gen_loss,model.dis_loss,
					 model.generator_output],feed_dict = {x:mnist_batch})
			else: 
				return tf.train.SessionRunArgs([model.gen_loss,model.dis_loss,
					 model.generator_output])
		def after_run(self, run_context, run_values):
			if not self.step%write_frequency:
				curtime = time.time()
				duration = curtime - self.time
				total_dur = curtime - self.total_time
				total_dur = timer(total_dur)
				self.time = curtime
				gen_loss, dis_loss, _ = run_values.results
				sec_per_batch = float(duration)/write_frequency
				print_string(self.step,gen_loss,dis_loss,sec_per_batch,total_dur)
			if (not self.step% self.step_points) or (self.step == last_step) or (not self.step%5000):
				self.step_points *= 2
				test = run_values.results[-1]
				test = test[0]*130+128
				test = np.clip(test, 0,255)
				test = np.asarray(test,np.uint8)
				if params[4] ==1:
					test = np.squeeze(test, axis = 2)
				img = Image.fromarray(test)
				img = img.resize((150,150))
				img.save('/media/yukun/Barracuda Hard Drive 2TB/Data/DCGAN/'+ Sets +'/trainGen/step%d.jpg'%self.step)
	hooks = [
		tf.train.StopAtStepHook(last_step = (last_step*2)),
		tf.train.NanTensorHook(model.dis_loss),
		log_hook()]
	with tf.train.MonitoredTrainingSession(checkpoint_dir= train_dir,hooks = hooks) as sess:
		while not sess.should_stop():
			sess.run(model.train_opt)

def main(argv=None):
	train_data()

if __name__ == '__main__':
	if tf.gfile.Exists(main_dir):
		tf.gfile.DeleteRecursively(main_dir)	
	os.makedirs(train_dir)
	os.makedirs(main_dir + '/trainGen')
	tf.app.run()