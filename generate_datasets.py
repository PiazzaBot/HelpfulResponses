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


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import seaborn as sns
from seaborn import FacetGrid
import nltk

from sklearn.model_selection import train_test_split

from nptyping import NDArray, Int, Shape
from typing import Dict, List, Tuple, Union


from piazza_api import Piazza
from piazza_api.network import Network

from data_processing_utils import *
from data_transform_utils import *



DATASET_DIR = 'transform_csv/'

'''
- train/val/test come from separate classes


- merge classes together and shuffle (shuffling not needed for df)
    - student poster id 
        - val and test should contain students from train
        
    - no student poster id
        - val and test should contain students from train
        (measures effect spid has on predictive performance)

1. combine and shuffle data
2. two train/val/test splits - 1 with same students in val/test and 1 w/o
3. Break into w/ and w/o student_poster_id

should have 4 train/val/test splits
can further use statistical tests to see if they improve performance

- baseline model


'''

class DataSet():
    """"""
    
    def __init__(self, train, val, test, continuous_features):
        self.train:DataFrame = train
        self.val:DataFrame = val
        self.test:DataFrame = test
        self.continuous_features:set = continuous_features
        self.discrete_features:set = set()

        self.features:list = train.keys()
        
        for f in self.features:
            if f not in self.continuous_features:
                self.discrete_features.add(f)


    def plot_discrete_distributions(self, set_type: str = 'train', hue=None, savePath=True) -> None:
        """
        :param hue: None, is_helpful, set where set indicates the type of dataset
        :param continuous_features: Set of features to exclude from the plot.
        :param hue: Add a third dimension to the plot. Can set to "is_helpful".
        """
        if set_type == 'train':
            dataset = self.train
        elif set_type == 'val':
            dataset = self.val
        else:
            dataset = self.test

        tot = len(self.discrete_features)
        cols = 3
        rows = tot // cols
        rows += tot % cols

        #g = FacetGrid(dataset, sharex=False, sharey=False)

        fig, axs = plt.subplots(nrows=rows, ncols=cols)
        fig.set_size_inches(15.5, 10.5)
        #fig, axs = plt.subplots(1)


        axs = axs.flatten()
        #sns.displot(data=dataset, x=dataset['is_private'])

        for idx, f in enumerate(self.discrete_features):
            if f != 'is_helpful':
                if hue:
                    sns.histplot(dataset, x=f"{f}",  discrete=True, binwidth=1, hue=hue, ax=axs[idx])
                    #g.map_dataframe(sns.histplot,  x=f"{f}",  discrete=True, binwidth=1, hue=hue)
                    #grid:FacetGrid = sns.displot(data=dataset, x=f"{f}",  kind="hist", discrete=True, binwidth=1, hue=hue)
                    #grid.savefig('tmp.png')
                else:
                    sns.histplot(dataset, x=f"{f}", discrete=True, binwidth=1, hue=hue, ax=axs[idx])
                    #g.map_dataframe(sns.histplot,  x=f"{f}",  discrete=True, binwidth=1)
             

        #g.savefig('tmp.png')


        # for idx, f in enumerate(self.discrete_features):
        #     if hue:
        #         sns.distplot(dataset[f], hue=hue, ax=axs[idx], kde=False)
        #     else:
        #         sns.distplot(dataset[f], ax=axs[idx], kde=False)

        if savePath:
            fig.savefig('save_as_a_png.png')

    def plot_continuous_distributions(self, set_type: str = 'train', hue=None, savePath=None) -> None:

        if set_type == 'train':
            dataset = self.train
        elif set_type == 'val':
            dataset = self.val
        else:
            dataset = self.test

        a = np.arange(0, 200, 5)
        b = np.arange(0, 200, 1)

        ax = sns.displot(data=dataset, x=f"response_time",  kind="hist", bins=a, hue=hue)
        ax.set_xlabels('response time (mins) with binwidth=5')

        ax=sns.displot(data=dataset, x=f"response_time",  kind="hist", bins=b,hue=hue)
        ax.set_xlabels('response time (mins) with binwidth=1')

        c = np.arange(0, 200, 1)
        c1 = np.arange(0, 200, 5)

        ax=sns.displot(data=dataset, x=f"question_length",  kind="hist", bins=c,hue=hue)
        ax.set_xlabels('question length with binwidth=1')
        ax=sns.displot(data=dataset, x=f"question_length",  kind="hist", bins=c1, hue=hue)
        ax.set_xlabels('question length with binwidth=5')


        ax=sns.displot(data=dataset, x=f"answer_length",  kind="hist", bins=c,hue=hue)
        ax.set_xlabels('answer length with binwidth=1')
        ax=sns.displot(data=dataset, x=f"answer_length",  kind="hist", bins=c1,hue=hue)
        ax.set_xlabels('answer length with binwidth=5')

        d= np.arange(0, 60, 1)
        e= np.arange(0, 50, 1) # take random sampling of 50
        ax=sns.displot(data=dataset, x=f"answerer_id",  kind="hist", bins=d, hue=hue)
        ax=sns.displot(data=dataset, x=f"student_poster_id",  kind="hist", bins=e, hue=hue)

    

    def print_stats(self):
        print(f'Printing split info:')
        print(f'# of Training examples: {len(self.train)}')
        print(f'# of Validation examples: {len(self.val)}')
        print(f'# of Test examples: {len(self.test)}')

        print(f'Total # of examples {len(self.train) + len(self.val) + len(self.test)}')

        print(f"Continuous features are {self.continuous_features}")
        print(f"Discrete features are {self.discrete_features}")

        print()
        print()

'''
1000 students in total
50% 25% 25% train val test split
'''
def split_dataset(dataset, continuous_features, student_biased=False):
    y_train = dataset['is_helpful']
    X_train = dataset.drop(labels=["is_helpful"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=0)

    #X_train['is_helpful'] = y_train
    #X_test['is_helpful'] = y_test

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=0)

    X_train['is_helpful'] = y_train
    X_test['is_helpful'] = y_test
    X_val['is_helpful'] = y_val

    return DataSet(X_train, X_val, X_test, continuous_features)



    


def main(args):
    fall2019_data = pd.read_csv(DATASET_DIR+'csc108_fall2019_aug.csv').drop(labels=["ID","post_id"], axis=1)
    fall2020_data = pd.read_csv(DATASET_DIR+'csc108_fall2020_aug.csv').drop(labels=["ID","post_id"], axis=1)
    fall2021_data = pd.read_csv(DATASET_DIR+'csc108_fall2021_aug.csv').drop(labels=["ID","post_id"], axis=1)

    continuous_features = {'question_length', 'answer_length', 'response_time', 'post_id', 'student_poster_id', 'answerer_id'}


    combined_data = pd.concat([fall2019_data, fall2020_data, fall2021_data], ignore_index=True)
    combined_data = combined_data.sample(frac=1, random_state=0)

    student_biased_dataset = split_dataset(combined_data, continuous_features)
    student_unbiased_dataset = DataSet(fall2020_data, fall2019_data, fall2021_data, continuous_features)

    student_biased_dataset.print_stats()
    student_unbiased_dataset.print_stats()

    #df[df.index.duplicated()]
    
    #df = student_biased_dataset.train
    

    #print(df[df.index.duplicated()])
    student_biased_dataset.plot_discrete_distributions(hue='is_helpful')




   
        


if __name__ == "__main__":
    arguments = docopt(__doc__)
    print(arguments)
    main(arguments)