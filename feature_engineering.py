import json
import time
import math
from datetime import datetime
import os
from tqdm import trange, tqdm
import re
from copy import deepcopy
import sys


import csv
import unicodedata
import html
from io import StringIO
from html.parser import HTMLParser


import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from numpy import ndarray 
import matplotlib.pyplot as plt
import seaborn as sns



from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import chi2




from nptyping import NDArray, Int, Shape
from typing import Dict, List, Tuple, Union


from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import nltk
import ssl
from nltk.tokenize import word_tokenize
from collections import namedtuple


from piazza_api import Piazza
from piazza_api.network import Network





def print_usage():
    usage = "Usage: python3.9 export_posts_json.py <cred-file-path> <save-json-filename>"
    print(usage)


def main():
    if len(sys.argv) != 3:
        print_usage()
        return
    cred_file_path = sys.argv[1]
    filename = sys.argv[2]
    with open(cred_file_path) as f:
        creds = json.load(f)
        email, password, courseid = creds['email'], creds['password'], creds['courseid']
        user_profile, course = login(email, password, courseid)
        export_posts_json(filename, course)


if __name__ == "__main__":
    main()