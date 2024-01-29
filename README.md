# Edifice Detection on Domain Translated Satellite Imagery
The Pix2Pix model which we are going to devise is a type of conditional GAN, where the
generation of the output image is conditional on an input, in this case, a source image. The
discriminator is provided both with a source image and the target image and must determine
whether the target is a plausible transformation of the source image. We attempt to introduce a new
strategy of training the discriminator network by using a different approach. The input source image
and the input target image are concatenated and then given as input to the first convolutional layer
of the discriminator network whose output is then split to retrieve the generated sample. This
(256*256*3) is then compared with the ground truth image and the final prediction
(16*16*1→{0,1}) is made. Because the generator is being via adversarial loss, which encourages
the generator to generate plausible images in the target domain, the training time substantially
increases. We handle this enormous train time by using batch training and batch normalization on
a Graphics Processing Unit (GPU) backend. This encourages the generator model to create
plausible translations of the source image. In particular, we attempt to translate images from their
satellite form to their aerial map form. In order to achieve this, we will make use of a compressed
NPZ Maps dataset. Nowadays, Pix2Pix GANs are further being applied on a range of image-toimage translation tasks such as converting black and white photographs to colour, sketches of
products to product photographs, etc. We evaluate the performance of our network using FréchetInception Distance, Manhattan Norm, Mean - Square Error and Structural Similarity in order to
check the quality and credibility of the generated image.

![image](https://github.com/Srihari123456/Edifice-Detection-of-Domain-translated-Satellite-Imagery-Using-Generative-Adversarial-Networks/assets/43612273/4a0a268c-e91f-4ab1-bf2d-0beaac139c2f)
