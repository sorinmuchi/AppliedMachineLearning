import os
import numpy
import scipy
import scipy.misc
from PIL import Image

import random, string

def randomword(length):
   return ''.join(random.choice(string.lowercase) for i in range(length))
   
OUTPUT_DIR = "data/label"
trainX = numpy.load('tinyX.npy') # this should have shape (26344, 3, 64, 64)
trainY = numpy.load('tinyY.npy') 
testX = numpy.load('tinyX_test.npy') # (6600, 3, 64, 64)

if not os.path.isdir("data"):
    os.makedirs("data")

labels = ["%d" % label for label in trainY]

for i in range(len(trainX)):
    label = labels[i]
    new_path = OUTPUT_DIR + label
    
    if not os.path.isdir(new_path):
        os.makedirs(new_path)

    im = Image.fromarray(trainX[i].transpose(2,1,0))
    im.save(OUTPUT_DIR + label + "/%d.jpeg" % i)