""" 

Usage:
    generate_datasets.py 
    

Options:
 

Improvements:

"""
from functools import partial
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

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split

from nptyping import NDArray, Int, Shape
from typing import Dict, List, Tuple, Union


from piazza_api import Piazza
from piazza_api.network import Network

from data_processing_utils import *
from data_transform_utils import *



DATASET_DIR = 'transform_csv/'
IMG_DIR = 'imgs/'

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
    
    def __init__(self, train, val, test, continuous_features, name, ignore_features={'post_id', 'student_poster_id', 'answerer_id'}):

        self.name = name
        self.dataset_save_path = DATASET_DIR + self.name + "/"
        self.img_save_path = IMG_DIR + self.name + "/"

        self.train:DataFrame = train
        self.val:DataFrame = val
        self.test:DataFrame = test

        self.train_pruned = None
        self.val_pruned = None
        self.test_pruned = None

        self.continuous_features:set = continuous_features
        self.discrete_features:set = set()
        self.ignore_features = ignore_features
        self.target = 'is_helpful'


        self.features:list = train.keys()
        
        for f in self.features:
            if (f not in self.continuous_features) and (f not in self.ignore_features):
                self.discrete_features.add(f)

        #print(f'discrete features are {self.discrete_features}')


    def __plot_discrete_distributions(self, set_type: str = 'train', hue=None, save_path=None) -> None:
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

        fig, axs = plt.subplots(nrows=rows, ncols=cols)
        fig.set_size_inches(20, 20)
     
        axs = axs.flatten()
        

        for idx, f in enumerate(self.discrete_features):
            if hue:
                sns.histplot(dataset, x=f"{f}",  discrete=True, binwidth=1, hue=hue, ax=axs[idx])
            else:
                sns.histplot(dataset, x=f"{f}", discrete=True, binwidth=1, hue=hue, ax=axs[idx])

        if save_path:
            save_path += "_discrete"
            print(f'saving to {save_path}')
            fig.savefig(save_path)



    def __plot_continuous_distributions(self, set_type: str = 'train', hue=None, save_path=None) -> None:

        if set_type == 'train':
            dataset = self.train
        elif set_type == 'val':
            dataset = self.val
        else:
            dataset = self.test


        tot = 8
        cols = 3
        rows = tot // cols
        rows += tot % cols

        fig, axs = plt.subplots(nrows=rows, ncols=cols)
        fig.set_size_inches(20, 20)
     
        axs = axs.flatten()

        a = np.arange(0, 200, 5)
        b = np.arange(0, 200, 1)

        ax = sns.histplot(data=dataset, x=f"response_time", bins=a, hue=hue, ax=axs[0])
        #ax.set_xlabels('response time (mins) with binwidth=5')

        ax=sns.histplot(data=dataset, x=f"response_time", bins=b,hue=hue, ax=axs[1])
        #ax.set_xlabels('response time (mins) with binwidth=1')

        c = np.arange(0, 200, 1)
        c1 = np.arange(0, 200, 5)

        ax=sns.histplot(data=dataset, x=f"question_length", bins=c,hue=hue, ax=axs[2])
        #ax.set_xlabels('question length with binwidth=1')
        ax=sns.histplot(data=dataset, x=f"question_length", bins=c1, hue=hue, ax=axs[3])
        #ax.set_xlabels('question length with binwidth=5')


        ax=sns.histplot(data=dataset, x=f"answer_length", bins=c,hue=hue, ax=axs[4])
        #ax.set_xlabels('answer length with binwidth=1')
        ax=sns.histplot(data=dataset, x=f"answer_length", bins=c1,hue=hue, ax=axs[5])
        #ax.set_xlabels('answer length with binwidth=5')

        d= np.arange(0, 60, 1)
        e= np.arange(0, 50, 1) # take random sampling of 50
        ax=sns.histplot(data=dataset, x=f"answerer_id", bins=d, hue=hue, ax=axs[6])
        ax=sns.histplot(data=dataset, x=f"student_poster_id", bins=e, hue=hue, ax=axs[7])

        if save_path:
            save_path += "_continuous"
            print(f'saving to {save_path}')
            fig.savefig(save_path)

    
    def save_distributions(self, hue_name=None):

        print('SAVING DISCRETE AND CONTINUOUS DISTRIBUTIONS')

        if not os.path.exists(self.img_save_path):
            os.makedirs(self.img_save_path)      

        for s in ['train', 'val', 'test']:
            path = self.img_save_path + f'{s}' + f"_{hue_name}"
            self.__plot_discrete_distributions(s, save_path=path, hue=hue_name)
            self.__plot_continuous_distributions(s, save_path=path, hue=hue_name)

            print('\n\n')


    def prune_features(self, select_k_best, save_path=None):

        self.__prune_features(select_k_best, score_func=mutual_info_classif, save_path=True)
        self.__prune_features(select_k_best, score_func=chi2, save_path=True)
        self.__prune_features(select_k_best, score_func=f_classif, save_path=True)


    def __prune_features(self, select_k_best='all', seed=0, score_func=mutual_info_classif, save_path=None):


        features = self.features
        # ** Make deep copies if necessary **
        self.train_pruned = self.train.copy(deep=True)
        self.val_pruned = self.val.copy(deep=True)
        self.test_pruned = self.test.copy(deep=True)

        if score_func == chi2:
            print('Pruning based on chi2')
            self.train_pruned = self.train_pruned.drop(list(self.continuous_features) + list(self.ignore_features), axis=1, errors='ignore')

        elif score_func == f_classif:
            print('Pruning based on anova')
            y_train = self.train_pruned['is_helpful']
            self.train_pruned = self.train_pruned.drop(list(self.discrete_features) + list(self.ignore_features), axis=1, errors='ignore')
            self.train_pruned['is_helpful'] = y_train
    
        elif score_func == mutual_info_classif:
            print('Pruning based on mutual information')
        else:
            assert(1 == 0)

        y_train = self.train_pruned['is_helpful']
        X_train = self.train_pruned.drop(labels=["is_helpful"], axis=1)

        y_val = self.val_pruned['is_helpful']
        y_test = self.test_pruned['is_helpful']

        features = list(X_train.keys())


        print(f'features to be filtered are: {features}')

        if select_k_best == 'all' or select_k_best > len(features):
            select_k_best = len(features)

        print(f'k = {select_k_best}')

        discrete_indices = list(range(0, len(features)))
        for f in self.continuous_features:
            if f in features:
                discrete_indices.remove(features.index(f))

        
        if score_func == mutual_info_classif:
            score_func = partial(mutual_info_classif, discrete_features=discrete_indices, random_state=seed)


        select_k = SelectKBest(score_func, k=select_k_best).fit(X_train, y_train)
        scores = select_k.scores_
        support = select_k.get_support()

        scores = scores.tolist()


        if save_path:
            save_path = self.img_save_path

            plot_type = 'mi'
            if score_func == chi2:
                plot_type = 'chi2'

            if score_func == f_classif:
                plot_type = 'anova'


            fig, axs = plt.subplots(1)
            fig.set_size_inches(10, 10)
            sns.barplot(x=list(range(0, len(features))), y=scores, ax=axs)

            save_path += "train_" + f"{plot_type}" + "_scores"
            print(f'saving to {save_path}')
            fig.savefig(save_path)


        features = np.array(features)
        feature_indices = np.where(support)[0]
        feature_names:ndarray = features[feature_indices]

        print(f'chosen features are {feature_names}')
        print(f'feature indices are {feature_indices}')

        # augment self.train_pruned, self.val_pruned, self.test_pruned
        np.append(feature_names, 'is_helpful')  # append is_helpful so it gets added to train/val/test

        self.train_pruned = self.train_pruned[feature_names]
        self.val_pruned = self.val_pruned[feature_names]
        self.test_pruned = self.test_pruned[feature_names]

        print('\n\n\n')
    


    def print_stats(self):
        print(f'Printing split info:')
        print(f'# of Training examples: {len(self.train)}')
        print(f'# of Validation examples: {len(self.val)}')
        print(f'# of Test examples: {len(self.test)}')

        print(f'Total # of examples {len(self.train) + len(self.val) + len(self.test)}')

        print(f"Continuous features are {self.continuous_features}")
        print(f"Discrete features are {self.discrete_features}")
        print(f"Ignored features are {self.ignore_features}")

        # compute student overlap distribution

        print()
        print()




'''
1000 students in total
50% 25% 25% train val test split
'''
def split_dataset(dataset, continuous_features, dataset_name, student_biased=False):
    y_train = dataset['is_helpful']
    X_train = dataset.drop(labels=["is_helpful"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=0)

    #X_train['is_helpful'] = y_train
    #X_test['is_helpful'] = y_test

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=0)

    X_train['is_helpful'] = y_train
    X_test['is_helpful'] = y_test
    X_val['is_helpful'] = y_val

    return DataSet(X_train, X_val, X_test, continuous_features, dataset_name)



    


def main(args):
    fall2019_data = pd.read_csv(DATASET_DIR+'csc108_fall2019_aug.csv').drop(labels=["ID","post_id"], axis=1)
    fall2020_data = pd.read_csv(DATASET_DIR+'csc108_fall2020_aug.csv').drop(labels=["ID","post_id"], axis=1)
    fall2021_data = pd.read_csv(DATASET_DIR+'csc108_fall2021_aug.csv').drop(labels=["ID","post_id"], axis=1)

    # features to exclude from discrete features plot
    #continuous_features = {'question_length', 'answer_length', 'response_time', 'post_id', 'student_poster_id', 'answerer_id'}
    continuous_features = {'question_length', 'answer_length', 'response_time'}


    combined_data = pd.concat([fall2019_data, fall2020_data, fall2021_data], ignore_index=True)
    combined_data = combined_data.sample(frac=1, random_state=0)

    student_biased_dataset = split_dataset(combined_data, continuous_features, 'biased_dataset')
    student_unbiased_dataset = DataSet(fall2020_data, fall2019_data, fall2021_data, continuous_features, 'unbiased_dataset')

    student_biased_dataset.print_stats()
    student_unbiased_dataset.print_stats()
    student_unbiased_dataset.save_distributions(hue_name=None)

    student_unbiased_dataset.prune_features(select_k_best=6)

    



if __name__ == "__main__":
    arguments = docopt(__doc__)
    print(arguments)
    main(arguments)