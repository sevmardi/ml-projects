"""
Train and test our prediction model.
Note:
Correct batch-size for LSTM is one
that you can use to divde a number of
samples for both training and testing set by.
So, we have 200 training samples and 50
testing ones, we can use batch-size=1
because we can 200/1 and 50/1, we can
also use 2, can't use 3 because we can't
200/3 and 50/3 etc.
"""
# https://github.com/PacktPublishing/Real-World-Python-Deep-Learning-Projects/blob/master/Section%205%20Code/source/train.py
# Configure to get the same
# results every time.
import conf

from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM

import math
import os
import sys

from prep import get_data, prep_data
from matplotlib import pyplot

import numpy as np
