""" 

Usage:
    generate_datasets.py 
    

Options:
 

Improvements:

"""
import ssl
from docopt import docopt

import json
import time
import math
from datetime import datetime
import os
from tqdm import trange, tqdm
import re
from copy import deepcopy
import sys
from math import isnan
import csv

from transformers import pipeline

import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from numpy import ndarray 
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

from nptyping import NDArray, Int, Shape
from typing import Dict, List, Tuple, Union


from piazza_api import Piazza
from piazza_api.network import Network

from data_processing_utils import *
from data_transform_utils import *



DATASET_DIR = 'transform_csv/'

def main(args):
    fall2019_data = pd.read_csv(DATASET_DIR+'csc108_fall2019_aug.csv')
    fall2020_data = pd.read_csv(DATASET_DIR+'csc108_fall2020_aug.csv')
    fall2021_data = pd.read_csv(DATASET_DIR+'csc108_fall2021_aug.csv')
   
   
                



if __name__ == "__main__":
    arguments = docopt(__doc__)
    print(arguments)
    main(arguments)