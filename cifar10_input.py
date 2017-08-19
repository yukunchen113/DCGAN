import os
from six.moves import xrange
import tensorflow as tf
data_dir='/media/yukun/Barracuda Hard Drive 2TB/Data/CIFAR10/cifar-10-batches-bin'#Path to data directory
Sets = 'CIFAR10'
IMAGE_RESIZE =32
NUM_CLASSES= 10
N_EXAMPLES_TRAIN_EPOCH = 50000
N_EXAMPLES_EVAL_EPOCH =10000
num_tffile = 6
IMAGE_CUT_SIZE =24
num_threads = 8
Queue_Fraction = 0.4

def read_file(filename_queue):
	label_bytes = 1
	record_bytes = 32*32*3 + label_bytes#image data length + label data length
	reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
	key, data = reader.read(filename_queue)
	record_bytes = tf.decode_raw(data, tf.uint8)
	image = tf.reshape(
		tf.strided_slice(record_bytes,[label_bytes],[label_bytes + 32*32*3]),
		[3, IMAGE_RESIZE, IMAGE_RESIZE]
	)
	image = tf.transpose(image, [1,2,0])
	return image

def batch_data(batch_size, image, min_dequeue_examples, shuffle = True, allow_smaller_final_batch= False):
	
	capacity = min_dequeue_examples + 3*batch_size
	if shuffle:
		images = tf.train.shuffle_batch(
			[image],
			min_after_dequeue = min_dequeue_examples, 
			capacity = capacity,
			batch_size = batch_size,
			num_threads = num_threads,
			allow_smaller_final_batch = allow_smaller_final_batch)
		return images
	images = tf.train.batch(
		[image],
		capacity = capacity,
		batch_size = batch_size,
		num_threads = num_threads,
		allow_smaller_final_batch = allow_smaller_final_batch
		)
	images = tf.cast(images,tf.float32)
	return images


def input(batch_size):
	filename_list = [os.path.join(
		data_dir,
		'data_batch_%d.bin'%i) for i in range(1,num_tffile)]
	filename_queue = tf.train.string_input_producer(filename_list)
	image = read_file(filename_queue)
	image = tf.cast(image, tf.float32)
	image = tf.image.resize_images(image, [IMAGE_RESIZE,IMAGE_RESIZE])
	min_dequeue_examples = int(N_EXAMPLES_TRAIN_EPOCH*Queue_Fraction)

	return batch_data(batch_size, image, min_dequeue_examples)