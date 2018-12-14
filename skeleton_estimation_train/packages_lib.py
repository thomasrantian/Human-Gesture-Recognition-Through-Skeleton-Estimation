from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2
from keras.initializers import random_normal,constant
from keras.models import Model
from keras.layers.merge import Multiply
import keras.backend as K
from keras.applications.vgg19 import VGG19
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger, TensorBoard
from keras.layers.merge import Concatenate
from keras.layers.pooling import MaxPooling2D
from keras.layers import Activation, Input, Lambda
import math
import os
import re
import sys
import pandas
from functools import partial
print ("Packages Loaded!")