import os
import sys
import cv2
import scipy.io
import time
import random
import json
import imutils
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from skimage import exposure 
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data 
from torch.utils.data import DataLoader
from livelossplot import PlotLosses
from IPython.display import clear_output
import scipy.ndimage.filters as filters
from ipywidgets import IntProgress
from IPython.display import display
from tqdm import tqdm_notebook as tqdm
import warnings
torch.backends.cudnn.enable = True 
warnings.simplefilter('ignore')
from matplotlib.ticker import FormatStrFormatter
from pylab import *
