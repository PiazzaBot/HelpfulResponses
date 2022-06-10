from sys import argv

import pandas as pd
from pandas import DataFrame, Series

from nptyping import NDArray, Int, Shape
from typing import Dict, List, Tuple, Union
from pandas import DataFrame, Series

from sklearn.tree import DecisionTreeClassifier 
from sklearn import metrics 

import tensorflow_decision_forests as tfdf


train_path = 'datasets/train'
val_path = 'datasets/val'
test_path = 'datasets/test'


def random_forest_classification(train, val):
    rf_model = tfdf.keras.RandomForestModel()
    rf_model.fit(x=train)


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

    train_pd = pd.read_csv(train_path)
    val_pd = pd.read_csv(val_path)
    test_pd = pd.read_csv(test_path)

    #print(train_path, val_path, test_path)

    train_tf = tfdf.keras.pd_dataframe_to_tf_dataset(train_pd, label='is_helpful')
    val_tf = tfdf.keras.pd_dataframe_to_tf_dataset(val_pd, label='is_helpful')

    random_forest_classification(train_tf, val_tf)

    


if __name__ == "__main__":
    main()
    pass
