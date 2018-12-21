import pickle
import random
# from decimal import *
import json
import keras
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import os

class TrimmedModel():
    def __init__(self):
        pass

    '''
    1. Load original model graph
    2. Assign new weights to layers 
    3. Test Accuracy
    '''