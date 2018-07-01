import numpy as np
#import tensorflow as tf
import pickle as pkl
import time
from random import shuffle
import pandas as pd 
import spectral
import matplotlib.pyplot as plt
import pylab as pl
import scipy
#import seaborn as sns
from collections import Counter
#import Spatial_dataset as input_data
#import patch_size
import os
import scipy.io as io
DATA_PATH = os.path.join(os.getcwd(),"Data")
input_image = scipy.io.loadmat( 'salinas_in.mat')
output_image = scipy.io.loadmat('salinas_gt.mat')

model_name = 'sample'
# input_image = np.rot90(input_image)
# output_image = np.rot90(output_image)
height = output_image.shape[0]
width = output_image.shape[1]
ground_truth=spectral.imshow(classes=output_image,figsize=(5,5))