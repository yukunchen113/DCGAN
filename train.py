import tensorflow as tf 
import numpy as np 
import os
import math
import util as ut
import model as mp
import time
import cPickle as pickle
from PIL import Image
import shutil
import matplotlib.pyplot as plt
def train_data(params,filewriter):
	#--------set important variables
	global_step = tf.contrib.framework.get_or_create_global_step()
	if params['Sets'] == 'MNIST':
		from tensorflow.examples.tutorials.mnist import input_data
		mnist = input_data.read_data_sets("Software/Python/MyCode/DCGAN/Data/",one_hot = True)
		params['n_channels'] = 1
	batch_size = params['batch_size']
	Sets = params['Sets']
	if Sets == 'CIFAR10':
		import cifar10_input as ip 
		images = ip.input(batch_size)
	if Sets == 'MNIST':
		x = tf.placeholder(tf.float32, [batch_size, 28*28])
		images = tf.reshape(x, [-1,28,28,1])
	if Sets == 'LSUN':
		import LSUN_input as ip
		images = ip.input(batch_size)
	#---------model__________________
	model = mp.dcgan(images,params)
	model.train(global_step)
	test_images = model.generator(test_sample=True)
	class log_hook(tf.train.SessionRunHook):
		def begin(self):
			self.step = -1
			self.previous_time = time.time()
			self.start_time = time.time()
			self.step_points = 1
		def before_run(self, run_context):
			self.step += 1
			feed_dict = None
			if Sets == 'MNIST':
				mnist_batch,_ = mnist.train.next_batch(batch_size)
				feed_dict={x:mnist_batch}
			return tf.train.SessionRunArgs([model.dis_loss,model.gen_loss,test_images],
				feed_dict=feed_dict)
		def after_run(self, run_context, run_values):

			if not self.step%params['print_frequency']:
				curtime = time.time()
				duration = curtime - self.previous_time
				total_dur = curtime - self.start_time
				total_dur = ut.timer(total_dur)
				self.previous_time = curtime
				dis_loss, gen_loss, _ = run_values.results
				sec_per_batch = float(duration)/params['print_frequency']
				string = ut.make_info(self.step,gen_loss,dis_loss,sec_per_batch,total_dur)
				print string
				if not self.step%params['write_frequency']:
					filewriter.write(string+'\n')
			if (not self.step% self.step_points) or (self.step == params['last_step']) or (not self.step%100):
				self.step_points *= 2
				test = run_values.results[-1]
				if params['Sets'] == 'MNIST':
					test = (1-test)*255
				else:
					test = ((test+1)*255)/2
				test = np.clip(test, 0,255)
				test = np.asarray(test,np.uint8)
				combined_img = Image.new('RGB', (150*params['n_saved_samples'],
					150))
				for i in range(params['n_saved_samples']):
					ind_test = test[i,:,:,:]
					if params['n_channels'] == 1:
						ind_test = np.squeeze(ind_test, axis = 2)
					img = Image.fromarray(ind_test)
					img = img.resize((150,150))
					combined_img.paste(img,(i*150,0))
				combined_img.save(params['info_dir']+'/step%d.jpg'%(self.step))

	hooks = [tf.train.NanTensorHook(model.dis_loss),
	tf.train.StopAtStepHook(last_step=params['last_step']*2),
	log_hook()]
	with tf.train.MonitoredTrainingSession(checkpoint_dir = params['train_dir'],
		hooks = hooks) as sess:
		while not sess.should_stop():
			sess.run(model.train_opt)

def main(argv=None):
	#set/create parameters----------------------------------------------------
	params = {
		'batch_size':128,#batch size, int
		'sample_size':100,#sample size, int
		'image_resize':32,#has to be at least divisible by the  2^(len(gen_params[1])+1)
		'n_saved_samples':8,#: Number of samples that are saved per batch (on prespecified step)
		'label_stddev':0.0,#: tinstance noise standard deviation (for labels)
		'gen_params':[[1024],[[5,2,512],[5,2,256],[5,2,128]],[5,2]],#: generator layer info [[reshape depth],[[conv1 kernel, conv1 stride, conv1 depth]...],[last conv kernel, last conv stride]] (last conv depth will be n_channels)
		'dis_params':[[3,1,96],[3,1,96],[3,2,96],[3,1,192],[3,1,192],[3,2,192],[3,1,192],[1,1,192],[1,1]],#: discrimonator layer depths [[conv1 kernel, conv1 stride, conv1 depth]...[last conv kernel, last conv stride]] (last conv depth will be n_classes)]
		'n_channels':3,#: number of colour channels
		'n_classes':2,#: number of classes for discriminator
		'print_frequency':10,#: print frequency to console
		'write_frequency':100,#: write frequency to info file
		'Sets':'CIFAR10',#ip.Sets #: Dataset
		'last_step':100000,#total number of iterations
		'opt_params':[[0.0002,0.5],[0.0002,0.5]],#: optimizer hyperparameters, [[dicriminator lr, discriminator beta], [generator lr, generator beta]]
		'lrelu_const':0.2,#: leaky rely constant
		'main_dir':'/media/yukun/Barracuda Hard Drive 2TB/Data/DCGAN/'#: directory of where you want to generate images and log to
		}
	#should only change directory above--------------------------------------
	params['main_dir'] = os.path.join(params['main_dir'], params['Sets'])
	params['train_dir'] = os.path.join(params['main_dir'], 'ModelCheckpoint')
	i = 1
	while tf.gfile.Exists(params['main_dir'] + '/'+params['Sets']+'trainGen%d'%i):
		i+=1
		time.sleep(0.100)
	params['info_dir'] = os.path.join(params['main_dir'],params['Sets']+'trainGen%d'%i)
	time.sleep(0.100)
	if params['image_resize']%(2**(len(params['gen_params'][1])+1)):
		print 'image resize should be divisible by 2^(len(gen_params[1])+1)'
		quit()
	#---running the training
	train_dir = params['train_dir']
	if tf.gfile.Exists(train_dir):
		time.sleep(0.100)
		tf.gfile.DeleteRecursively(train_dir)
		time.sleep(0.100)	
	os.mkdir(train_dir)
	time.sleep(0.100)
	os.mkdir(params['info_dir'])
	with open(os.path.join(params['info_dir'], 'info.txt'),'w') as f:
		string = ''
		for i in params:
			string += i+':'+str(params[i]) +'\n'
		string +='\n\n\nRun Info: \n'
		f.write(string)
		train_data(params,f)

if __name__ == '__main__':
	tf.app.run()