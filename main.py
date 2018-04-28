import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from scipy.misc import imsave
import os
import shutil
from PIL import Image
import time
import random
import sys
from cyclegan import *
from layers import *


model = CycleGAN()
model.train()
