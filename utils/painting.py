import os  
import matplotlib.pyplot as plt  
import numpy as np  
from .metrics import metric
import pickle

def draw_comparision(data_path):
    with open(data_path, 'rb') as file:
        data = pickle.load(file)