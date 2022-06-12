from sys import argv
import sys

import pandas as pd
from pandas import DataFrame, Series

import numpy as np

from nptyping import NDArray, Int, Shape
from typing import Dict, List, Tuple, Union
from pandas import DataFrame, Series

from sklearn.tree import DecisionTreeClassifier 
from sklearn import metrics 

import tensorflow_decision_forests as tfdf
import tensorflow as tf

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from generate_datasets import DataSet

import os



train_path = 'datasets/train'
val_path = 'datasets/val'
test_path = 'datasets/test'


def plot_val_logs(logs, save_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(logs["score"], label="current trial")
    plt.plot(logs["score"].cummax(), label="best trial")
    plt.xlabel("Tuning step")
    plt.ylabel("Tuning score")
    plt.legend()
    
    plt.savefig(save_dir + 'val.png')


def plot_train_logs(logs, save_dir):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in logs])
    plt.xlabel("Number of trees")
    plt.ylabel("Accuracy (out-of-bag)")

    plt.subplot(1, 2, 2)
    plt.plot([log.num_trees for log in logs], [log.evaluation.loss for log in logs])
    plt.xlabel("Number of trees")
    plt.ylabel("Logloss (out-of-bag)")

    print(f'SAVING LOGS to {save_dir + "train.png"}')

    plt.savefig(save_dir + 'train.png')



def log_msg(filepath, msg, mode, summary=False):
    """Logs msg to both file and stdout

        :param msg: either a string or a model. If summary=True, then msg refers to a model and 
        we must do: model.summary() which I think prints to stdout within the function.
    """

    stdout = sys.stdout
    sys.stdout = open(filepath, mode)
    if summary:
        msg.summary()
    else:
        print(msg)
    sys.stdout.close()
    sys.stdout = stdout

    # if summary:
    #     msg.summary()
    # else:
    #     print(msg)


def classify(train, val, test, model, log_path=None, tuned=False, print_summary=False):

    model.fit(x=train, validation_data=val)
  
    print(f'number of training examples {model.num_training_examples}')
    print(f'number of validation examples {model.num_validation_examples}')

    if log_path:
        dirname = os.path.dirname(log_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)  

    score_filepath = ""
    summary_filepath = ""

    if not tuned:
        score_filepath = log_path + 'test_score_untuned.txt'
        summary_filepath = log_path + 'train_summary_untuned.txt'
    else:
        score_filepath = log_path + 'test_score_tuned.txt'
        summary_filepath = log_path + 'train_summary_tuned.txt'

    if print_summary:
        log_msg(summary_filepath, model, 'w', summary=True)


    model.compile(metrics=['accuracy'])
    evaluation = model.evaluate(test, return_dict=True)

   
    for name, value in evaluation.items():
        msg = f"{name}: {value:.4f}"
        log_msg(score_filepath, msg, 'a')
    

    train_logs = model.make_inspector().training_logs()

    if tuned:
        tuned_logs = model.make_inspector().tuning_logs()

        best_hyperparams = tuned_logs[tuned_logs.best].iloc[0]
        log_msg(score_filepath, best_hyperparams, 'a')

    if log_path:
        if tuned:
            plot_train_logs(train_logs, log_path)
            plot_val_logs(tuned_logs, log_path)
        else:
            plot_train_logs(train_logs, log_path)
    

    #tfdf.model_plotter.plot_model_in_colab(rf_model, tree_idx=0, max_depth=3)


def baseline(dataset:DataSet, log_path=""):
    """Pick most common class in train set and use that to make predictions in the test set"""

    log_path += 'baseline/test_scores.txt'

    if log_path != "":
        dirname = os.path.dirname(log_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)  

    

    train_pd, val_pd, test_pd = dataset.train, dataset.val, dataset.test

    #print(train_pd.head())

    num_helpful = len(train_pd[train_pd['is_helpful'] == 1])
    num_unhelpful  = len(train_pd[train_pd['is_helpful'] == 0])

    common_class = 1

    if num_unhelpful > num_helpful:
        common_class = 0

    accuracy =  len(test_pd[test_pd['is_helpful'] == common_class]) / len(test_pd)



    with open(log_path, 'w') as file:
        file.write(f'num helpful: {num_helpful}\n')
        file.write(f'num unhelpful: {num_unhelpful}\n')
        file.write(f'most common class is: {common_class}\n')

        file.write(f'accuracy is: {accuracy}\n')

 


def random_forest_classification(dataset:DataSet, print_summary=False, log_path="", tune=False):

    if tune:
        log_path += 'tuned/'
    else:
        log_path += 'not_tuned/'


    train_pd, val_pd, test_pd = dataset.train, dataset.val, dataset.test

    features:set = dataset.features
    continuous_features:set = dataset.continuous_features
    discrete_features:set = dataset.discrete_features
    ignored_features:set = dataset.ignore_features

    train_tf = tfdf.keras.pd_dataframe_to_tf_dataset(train_pd, label='is_helpful')
    val_tf = tfdf.keras.pd_dataframe_to_tf_dataset(val_pd, label='is_helpful')
    test_tf = tfdf.keras.pd_dataframe_to_tf_dataset(test_pd, label='is_helpful')

    categorical_features = []

    for f in discrete_features: #interprets ignored features as CATEGORICAL (correct semantics)
            tf_feature = tfdf.keras.FeatureUsage(name=f, semantic=tfdf.keras.FeatureSemantic.CATEGORICAL)
            categorical_features.append(tf_feature)


    if tune:
        # Create a Random Search tuner with 50 trials.
        tuner = tfdf.tuner.RandomSearch(num_trials=50)

        tuner.choice("min_examples", [2, 5, 7, 10])
        tuner.choice("categorical_algorithm", ["CART", "RANDOM"]) 

        local_search_space = tuner.choice("growing_strategy", ["LOCAL"])
        local_search_space.choice("max_depth", [3, 4, 5, 6, 8])  

        global_search_space = tuner.choice("growing_strategy", ["BEST_FIRST_GLOBAL"], merge=True)
        global_search_space.choice("max_num_nodes", [16, 32, 64, 128, 256])

        rf_model = tfdf.keras.RandomForestModel(features=categorical_features, verbose=1, tuner=tuner)
    else:
        rf_model = tfdf.keras.RandomForestModel(features=categorical_features, verbose=1)
    

    assert(rf_model._check_dataset == True)

    classify(train_tf, val_tf, test_tf, rf_model, log_path, tuned=tune, print_summary=print_summary)

       

    


        


