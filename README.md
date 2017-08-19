# DCGAN
Repository creating [a deep convolutional generative adversarial network](https://arxiv.org/pdf/1511.06434.pdf) Radford et al. in tensorflow

### What does it do?
DCGANs generate images using CNNs

### How does it work?
There are two parts to most [Generative Adversarial Networks](https://arxiv.org/pdf/1406.2661.pdf) Goodfellow et al.
- a generator which generates images (using transposed CNN)
- a discriminator which decides if the images are real or fake (using normal CNN)   

Steps of a GAN:
1. Batches of images from a real-image dataset and images from the generator are inputed into the discriminator
	- discriminator learns real features from generated features by backpropogating to the inputed images
	- backpropogate through the discriminator only
	- use 0 as labels for generated images
	- use 1 as lables for real images
2. Backpropogate through whole discriminator-generator model with 1 as a label
	- This will allow the generator to learn features of real images

### Files:
Files in this repository:
- cifar10_input.py:
	- provides batches of unaltered cifar10 images to model
- model.py:
	- DCGAN model
- util.py:
	- functions to simplify model.py (these are not in model.py to keep it short)
- train.py:
	- trains data while saving sample images on a predefined epoch
### Improvements:
- on MNIST (and a little on cifar), Generator doesn't converge while discriminator converges completely to an accuracy of 99%
	- can be solved by crippling the discriminator

### References:
@article{DBLP:journals/corr/RadfordMC15,   
  author    = {Alec Radford and   
               Luke Metz and   
               Soumith Chintala},   
  title     = {Unsupervised Representation Learning with Deep Convolutional Generative   
               Adversarial Networks},  
  journal   = {CoRR},   
  volume    = {abs/1511.06434},  
  year      = {2015},   
  url       = {http://arxiv.org/abs/1511.06434},   
  timestamp = {Wed, 07 Jun 2017 14:40:30 +0200},   
  biburl    = {http://dblp.uni-trier.de/rec/bib/journals/corr/RadfordMC15},   
  bibsource = {dblp computer science bibliography, http://dblp.org}   
}   


@ARTICLE{2014arXiv1406.2661G,   
   author = {{Goodfellow}, I.~J. and {Pouget-Abadie}, J. and {Mirza}, M. and    
	{Xu}, B. and {Warde-Farley}, D. and {Ozair}, S. and {Courville}, A. and    
	{Bengio}, Y.},   
    title = "{Generative Adversarial Networks}",   
  journal = {ArXiv e-prints},   
archivePrefix = "arXiv",    
   eprint = {1406.2661},   
 primaryClass = "stat.ML",   
 keywords = {Statistics - Machine Learning, Computer Science - Learning},   
     year = 2014,   
    month = jun,   
   adsurl = {http://adsabs.harvard.edu/abs/2014arXiv1406.2661G},   
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}   
}   

