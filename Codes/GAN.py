from os import listdir
from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed
import PIL
import numpy as np
import matplotlib.pyplot as plt
global g_model,d_model,gan_model,gan

from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from matplotlib import pyplot

from keras.models import load_model
from numpy import load
from numpy import vstack
from matplotlib import pyplot
from numpy.random import randint

from math import floor
from numpy import ones
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from numpy import exp
from numpy.random import shuffle
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets import cifar10
from skimage.transform import resize
from numpy import asarray
import cv2




class GAN():

	

	def __init__(self):
		self.g_model = 0
		self.d_model = 0
		self.gan_model = 0
		self.image_shape = 0
		self.testimage = ''
		self.generatedimage = ''
		self.testimagepath = ''
		self.out = 0
		from os import listdir
		from numpy import asarray
		from numpy import vstack
		from keras.preprocessing.image import img_to_array
		from keras.preprocessing.image import load_img
		from numpy import savez_compressed
		import PIL
		import numpy as np
		import matplotlib.pyplot as plt

		from numpy import load
		from numpy import zeros
		from numpy import ones
		from numpy.random import randint
		from keras.optimizers import Adam
		from keras.initializers import RandomNormal
		from keras.models import Model
		from keras.models import Input
		from keras.layers import Conv2D
		from keras.layers import Conv2DTranspose
		from keras.layers import LeakyReLU
		from keras.layers import Activation
		from keras.layers import Concatenate
		from keras.layers import Dropout
		from keras.layers import BatchNormalization
		from keras.layers import LeakyReLU
		from matplotlib import pyplot

		from keras.models import load_model
		from numpy import load
		from numpy import vstack
		from matplotlib import pyplot
		from numpy.random import randint

		from math import floor
		from numpy import ones
		from numpy import expand_dims
		from numpy import log
		from numpy import mean
		from numpy import std
		from numpy import exp
		from numpy.random import shuffle
		from keras.applications.inception_v3 import InceptionV3
		from keras.applications.inception_v3 import preprocess_input
		from keras.datasets import cifar10
		from skimage.transform import resize
		from numpy import asarray
		import cv2
		#import cv2_imshow

# load all images in a directory into memory
	

	def load_images(self,path, size=(256,512)):
		src_list, tar_list = list(), list()
		# enumerate filenames in directory, assume all are images
		for filename in listdir(path):
			# load and resize the image
			pixels = load_img(path + filename, target_size=size)
			# convert to numpy array
			pixels = img_to_array(pixels)
			# split into satellite and map
			sat_img, map_img = pixels[:, :256], pixels[:, 256:]
			src_list.append(sat_img)
			tar_list.append(map_img)
		return [asarray(src_list), asarray(tar_list)]


	def compress_images(self,path, filename): 
	    [src_images, tar_images] = self.load_images(path)
	    print('Loaded: ', src_images.shape, tar_images.shape)
	    savez_compressed(filename, src_images, tar_images)
	    print('Saved dataset: ', filename)

	"""Model Creation and Training"""

	# example of pix2pix gan for satellite to map image-to-image translation


	# define the discriminator model
	def define_discriminator(self):
		# weight initialization
		init = RandomNormal(stddev=0.02)
		# source image input
		in_src_image = Input(shape=self.image_shape)
		# target image input
		in_target_image = Input(shape=self.image_shape)
		# concatenate images channel-wise
		merged = Concatenate()([in_src_image, in_target_image])
		# C64
		d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
		d = LeakyReLU(alpha=0.2)(d)
		# C128
		d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
		d = BatchNormalization()(d)
		d = LeakyReLU(alpha=0.2)(d)
		# C256
		d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
		d = BatchNormalization()(d)
		d = LeakyReLU(alpha=0.2)(d)
		# C512
		d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
		d = BatchNormalization()(d)
		d = LeakyReLU(alpha=0.2)(d)
		# second last output layer
		d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
		d = BatchNormalization()(d)
		d = LeakyReLU(alpha=0.2)(d)
		# patch output
		d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
		patch_out = Activation('sigmoid')(d)
		# define model
		model = Model([in_src_image, in_target_image], patch_out)
		# compile model
		opt = Adam(lr=0.0002, beta_1=0.5)
		model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
		return model

	# define an encoder block
	def define_encoder_block(self,layer_in, n_filters, batchnorm=True):
		# weight initialization
		init = RandomNormal(stddev=0.02)
		# add downsampling layer
		g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
		# conditionally add batch normalization
		if batchnorm:
			g = BatchNormalization()(g, training=True)
		# leaky relu activation
		g = LeakyReLU(alpha=0.2)(g)
		return g

	# define a decoder block
	def decoder_block(self,layer_in, skip_in, n_filters, dropout=True):
		# weight initialization
		init = RandomNormal(stddev=0.02)
		# add upsampling layer
		g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
		# add batch normalization
		g = BatchNormalization()(g, training=True)
		# conditionally add dropout
		if dropout:
			g = Dropout(0.5)(g, training=True)
		# merge with skip connection
		g = Concatenate()([g, skip_in])
		# relu activation
		g = Activation('relu')(g)
		return g

	# define the standalone generator model
	def define_generator(self):
		# weight initialization
		init = RandomNormal(stddev=0.02)
		# image input
		in_image = Input(shape=self.image_shape)
		# encoder model
		e1 = self.define_encoder_block(in_image, 64, batchnorm=False)
		e2 = self.define_encoder_block(e1, 128)
		e3 = self.define_encoder_block(e2, 256)
		e4 = self.define_encoder_block(e3, 512)
		e5 = self.define_encoder_block(e4, 512)
		e6 = self.define_encoder_block(e5, 512)
		e7 = self.define_encoder_block(e6, 512)
		# bottleneck, no batch norm and relu
		b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
		b = Activation('relu')(b)
		# decoder model
		d1 = self.decoder_block(b, e7, 512)
		d2 = self.decoder_block(d1, e6, 512)
		d3 = self.decoder_block(d2, e5, 512)
		d4 = self.decoder_block(d3, e4, 512, dropout=False)
		d5 = self.decoder_block(d4, e3, 256, dropout=False)
		d6 = self.decoder_block(d5, e2, 128, dropout=False)
		d7 = self.decoder_block(d6, e1, 64, dropout=False)
		# output
		g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
		out_image = Activation('tanh')(g)
		# define model
		model = Model(in_image, out_image)
		return model

	# define the combined generator and discriminator model, for updating the generator
	def define_gan(self):
		# make weights in the discriminator not trainable
		for layer in self.d_model.layers:
			if not isinstance(layer, BatchNormalization):
				layer.trainable = False
		# define the source image
		in_src = Input(shape=self.image_shape)
		# connect the source image to the generator input
		gen_out = self.g_model(in_src)
		# connect the source input and generator output to the discriminator input
		dis_out = self.d_model([in_src, gen_out])
		# src image as input, generated image and classification output
		model = Model(in_src, [dis_out, gen_out])
		# compile model
		opt = Adam(lr=0.0002, beta_1=0.5)
		model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
		return model

	# load and prepare training images
	def load_real_samples(self,filename):
		# load compressed arrays
		data = load(filename)
		# unpack arrays
		X1, X2 = data['arr_0'], data['arr_1']
		# scale from [0,255] to [-1,1]
		X1 = (X1 - 127.5) / 127.5
		X2 = (X2 - 127.5) / 127.5
		return [X1, X2]

	# select a batch of random samples, returns images and target
	def generate_real_samples(self,dataset, n_samples, patch_shape):
		# unpack dataset
		trainA, trainB = dataset
		# choose random instances
		ix = randint(0, trainA.shape[0], n_samples)
		# retrieve selected images
		X1, X2 = trainA[ix], trainB[ix]
		# generate 'real' class labels (1)
		y = ones((n_samples, patch_shape, patch_shape, 1))
		return [X1, X2], y

	# generate a batch of images, returns images and targets
	def generate_fake_samples(self, samples, patch_shape):
		# generate fake instance
		X = self.g_model.predict(samples)
		# create 'fake' class labels (0)
		y = zeros((len(X), patch_shape, patch_shape, 1))
		return X, y

	# generate samples and save as a plot and save the model
	def summarize_performance(self,step, dataset, n_samples=3):
		# select a sample of input images
		[X_realA, X_realB], _ = self.generate_real_samples(dataset, n_samples, 1)
		# generate a batch of fake samples
		X_fakeB, _ = self.generate_fake_samples(X_realA, 1)
		# scale all pixels from [-1,1] to [0,1]
		X_realA = (X_realA + 1) / 2.0
		X_realB = (X_realB + 1) / 2.0
		X_fakeB = (X_fakeB + 1) / 2.0
		# plot real source images
		for i in range(n_samples):
			pyplot.subplot(3, n_samples, 1 + i)
			pyplot.axis('off')
			pyplot.imshow(X_realA[i])
		# plot generated target image
		for i in range(n_samples):
			pyplot.subplot(3, n_samples, 1 + n_samples + i)
			pyplot.axis('off')
			pyplot.imshow(X_fakeB[i])
		# plot real target image
		for i in range(n_samples):
			pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
			pyplot.axis('off')
			pyplot.imshow(X_realB[i])
		# save plot to file
		filename1 = 'plot_%06d.png' % (step+1)
		pyplot.savefig(filename1)
		pyplot.close()
		# save the generator model
		filename2 = 'g_model_%06d' % (step+1)
		self.g_model.save(filename2)
	#    filename3 = 'd_model_%06d.h5' % (step+1)
		filename3 = 'd_model_%06d' % (step+1)
		self.d_model.save(filename3)
		print('>Saved: %s and %s and %s' % (filename1, filename2, filename3))
	  

	# train pix2pix models
	def train(self, dataset, n_epochs=5, n_batch=1):
		# determine the output square shape of the discriminator
		n_patch = self.d_model.output_shape[1]
		# unpack dataset
		trainA, trainB = dataset
		# calculate the number of batches per training epoch
		bat_per_epo = int(len(trainA) / n_batch)
		# calculate the number of training iterations
		n_steps = bat_per_epo * n_epochs
		# manually enumerate epochs
		for i in range(n_steps):
			# select a batch of real samples
			[X_realA, X_realB], y_real = self.generate_real_samples(dataset, n_batch, n_patch)
			# generate a batch of fake samples
			X_fakeB, y_fake = self.generate_fake_samples(X_realA, n_patch)
			# update discriminator for real samples
			d_loss1 = self.d_model.train_on_batch([X_realA, X_realB], y_real)
			# update discriminator for generated samples
			d_loss2 = self.d_model.train_on_batch([X_realA, X_fakeB], y_fake)
			# update the generator
			g_loss, _, _ = self.gan_model.train_on_batch(X_realA, [y_real, X_realB])
			# summarize performance
			print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
			# summarize model performance
			if (i+1) % (bat_per_epo * 5) == 0:
				summarize_performance(i, dataset)



	# example of loading a pix2pix model and using it for image to image translation


	# load and prepare training images
	def load_real_samples(self,filename):
		# load compressed arrays
		data = load(filename)
		# unpack arrays
		X1, X2 = data['arr_0'], data['arr_1']
		# scale from [0,255] to [-1,1]
		X1 = (X1 - 127.5) / 127.5
		X2 = (X2 - 127.5) / 127.5
		return [X1, X2]

	# plot source, generated and target images
	def plot_images(self,src_img, gen_img, tar_img):
	    images = vstack((src_img, gen_img, tar_img))
	    # scale from [-1,1] to [0,1]
	    print(type(images))
	    images = (images + 1) / 2.0
	    titles = ['Source', 'Generated', 'Expected']
		# plot images row by row
	    for i in range(len(images)):
			# define subplot
	        pyplot.subplot(1, 3, 1 + i)
			# turn off axis
	        pyplot.axis('off')
			# plot raw pixel data
	        pyplot.imshow(images[i])
			# show title
	        pyplot.title(titles[i])
	    #pyplot.savefig('epoch-12')
	    pyplot.show(block = False)

	def load_image(self,size=(256,512)):
		# load image with the preferred size
		pixels = load_img(self.testimagepath, target_size=size,interpolation='nearest')
		# convert to numpy array
		pixels = img_to_array(pixels)
		# scale from [0,255] to [-1,1]
		pixels = (pixels - 127.5) / 127.5
		sat_img, map_img = pixels[:, :256], pixels[:, 256:]
		# reshape to 1 sample
		sat_img = expand_dims(sat_img, 0)
		return sat_img



	# calculate inception score for cifar-10 in Keras

	def detect_edifices(self):
		aerial_map = self.testimagepath.split('/')[-2] + '/' +self.testimagepath.split('/')[-1]
		img = cv2.imread("../Model/Dataset/aerial/"+aerial_map)
		imgGry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		ret , thresh = cv2.threshold(imgGry, 240 , 255, cv2.THRESH_BINARY_INV)    
		contours , hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

		for contour in contours:
		    approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
		    cv2.drawContours(img, [approx], 0, (0, 0, 0), 5)
		    if len(approx) == 4 :
		        x, y , w, h = cv2.boundingRect(approx)

		cv2.imshow("Detected Edifice",img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

		
# scale an array of images to a new size
	def scale_images(self,images, new_shape):
		images_list = list()
		for image in images:
			# resize with nearest neighbor interpolation
			new_image = resize(image, new_shape)
			# store
			images_list.append(new_image)
		return asarray(images_list)


	# assumes images have any shape and pixels in [0,255]
	def calculate_inception_score(self,images, n_split=10, eps=1E-16):
		# load inception v3 model
		model = InceptionV3()
		# enumerate splits of images/predictions
		scores = list()
		n_part = floor(images.shape[0] / n_split)
		for i in range(n_split):
			# retrieve images
			ix_start, ix_end = i * n_part, (i+1) * n_part
			subset = images[ix_start:ix_end]
			# convert from uint8 to float32
			subset = subset.astype('float32')
			# scale images to the required size
			subset = self.scale_images(subset, (299,299,3))
			# pre-process images, scale to [-1,1]
			subset = preprocess_input(subset)
			# predict p(y|x)
			p_yx = model.predict(subset)
			# calculate p(y)
			p_y = expand_dims(p_yx.mean(axis=0), 0)
			# calculate KL divergence using log probabilities
			kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
			# sum over classes
			sum_kl_d = kl_d.sum(axis=1)
			# average over images
			avg_kl_d = mean(sum_kl_d)
			# undo the log
			is_score = exp(avg_kl_d)
			# store
			scores.append(is_score)
		# average across images
		is_avg, is_std = mean(scores), std(scores)
		return is_avg, is_std




