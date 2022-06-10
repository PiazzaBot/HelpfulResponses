from sys import argv

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



train_path = 'datasets/train'
val_path = 'datasets/val'
test_path = 'datasets/test'



def plot_logs(logs, save_file):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot([log.num_trees for log in logs], [log.evaluation.accuracy for log in logs])
    plt.xlabel("Number of trees")
    plt.ylabel("Accuracy (out-of-bag)")

    plt.subplot(1, 2, 2)
    plt.plot([log.num_trees for log in logs], [log.evaluation.loss for log in logs])
    plt.xlabel("Number of trees")
    plt.ylabel("Logloss (out-of-bag)")

    plt.savefig(save_file)

def classify(train, val, test, model, tuned=False):

    model.fit(x=train, validation_data=val)
    print(f'number of training examples {model.num_training_examples}')
    print(f'number of validation examples {model.num_validation_examples}')
    #print(rf_model.summary())

    model.compile(metrics=['accuracy'])
    evaluation = model.evaluate(test, return_dict=True)

    for name, value in evaluation.items():
        print(f"{name}: {value:.4f}")


    logs = model.make_inspector().training_logs()

    if tuned:
        tuned_logs = model.make_inspector().tuning_logs()
        #print(tuned_logs)

    #plot_logs(logs, 'train_logs.png')
    

    #tfdf.model_plotter.plot_model_in_colab(rf_model, tree_idx=0, max_depth=3)


def random_forest_classification(train, val, test, features, continuous_features):

    categorical_features = []

    for f in features:
        if f not in continuous_features:
            tf_feature = tfdf.keras.FeatureUsage(name=f, semantic=tfdf.keras.FeatureSemantic.CATEGORICAL)
            categorical_features.append(tf_feature)

    # Create a Random Search tuner with 50 trials.
    tuner = tfdf.tuner.RandomSearch(num_trials=50)

    tuner.choice("min_examples", [2, 5, 7, 10])
    tuner.choice("categorical_algorithm", ["CART", "RANDOM"]) 

    local_search_space = tuner.choice("growing_strategy", ["LOCAL"])
    local_search_space.choice("max_depth", [3, 4, 5, 6, 8])  

    global_search_space = tuner.choice("growing_strategy", ["BEST_FIRST_GLOBAL"], merge=True)
    global_search_space.choice("max_num_nodes", [16, 32, 64, 128, 256])


    rf_model = tfdf.keras.RandomForestModel(features=categorical_features, verbose=2)
    rf_tuned_model = tfdf.keras.RandomForestModel(features=categorical_features, verbose=2,tuner=tuner)

    assert(rf_model._check_dataset == True)

    classify(train, val, test, rf_model, tuned=False)
    classify(train, val, test, rf_tuned_model, tuned=True)

    




def print_usage():
    print('Usage: python3.9 model.py <dataset-type=default|mi|chi2|anova>')
        
def main():

    global train_path, val_path, test_path

    if len(argv) != 2:
        print_usage()
        return False

    dataset_type = argv[1]
    if dataset_type == 'default':
        train_path += '.csv'
        val_path += '.csv'
        test_path += '.csv'
    elif dataset_type == 'mi':
        train_path += '_mi.csv'
        val_path += '_mi.csv'
        test_path += '_mi.csv'
    elif dataset_type == 'chi2':
        train_path += '_chi2.csv'
        val_path += '_chi2.csv'
        test_path += '_chi2.csv'
    elif dataset_type == 'anova':
        train_path += '_anova.csv'
        val_path += '_anova.csv'
        test_path += '_anova.csv'
    else: # shouldn't get here
        assert(1 == 0)

    continuous_features = {'question_length', 'answer_length', 'response_time', 'post_id', 'student_poster_id', 'answerer_id', 'is_helpful'}

    train_pd = pd.read_csv(train_path)
    val_pd = pd.read_csv(val_path)
    test_pd = pd.read_csv(test_path)


    features = train_pd.keys()


    train_tf = tfdf.keras.pd_dataframe_to_tf_dataset(train_pd, label='is_helpful')
    val_tf = tfdf.keras.pd_dataframe_to_tf_dataset(val_pd, label='is_helpful')
    test_tf = tfdf.keras.pd_dataframe_to_tf_dataset(test_pd, label='is_helpful')

    #plt.plot([1, 2, 3, 4])
    #plt.ylabel('some numbers')
    #plt.savefig("mygraph.png")
    random_forest_classification(train_tf, val_tf, test_tf, features, continuous_features)

    


if __name__ == "__main__":
    main()
