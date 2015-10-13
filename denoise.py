# coding: utf-8
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import os
import sys
import numpy as np

import pylab

import PIL.Image as Image
import PIL.ImageOps as ImageOps

class ImageNoise(object):
    def __init__(self, theano_rng):
        self.theano_rng = theano_rng

    def get_corrupted_input(self, input, corruption_level):
        return self.theano_rng.binomial(size=input.shape, n=1, p=1-corruption_level, dtype=theano.config.floatX)*input

if __name__ == '__main__':
    img = Image.open(open('image/cat.jpg'))
    width = img.size[0]
    height = img.size[1]
    img_gray = ImageOps.grayscale(img)
    
    print "original image shape: " + str(img)
    print "gray image shape: " + str(img_gray)

    #pylab.subplot(1,1,1)
    #pylab.imshow(img_gray, cmap="Greys_r")
    #pylab.show()

    img_gray = np.array(img_gray, dtype='float64')
    img_ = img_gray.reshape(1, 1, height, width)

    print img_.shape

    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    input = T.dtensor4(name='input')
    image_noise = ImageNoise(theano_rng=theano_rng)

    # can change corruption level 0.0 ~ 1.0
    output = image_noise.get_corrupted_input(input, corruption_level=0.7)
    
    corrputed_output = theano.function([input], output)    
    corrupted_image =  corrputed_output(img_)

    print "corrupted image shape: " + str(corrupted_image.shape)

    pylab.subplot(1, 2, 1)
    pylab.imshow(img_gray, cmap="Greys_r")

    pylab.subplot(1, 2, 2)
    pylab.imshow(corrupted_image[0,0,:,:], cmap="Greys_r")
    pylab.gray

    pylab.savefig('log/denoise_sample.png')
    pylab.show() 


