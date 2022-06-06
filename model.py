import json
from sys import argv

from nptyping import NDArray, Int, Shape
from typing import Dict, List, Tuple, Union
from pandas import DataFrame, Series

from sklearn.tree import DecisionTreeClassifier 
from sklearn import metrics 


DATASET_PATH = "dataset_paths.json"

def decision_tree_clf(train: DataFrame, val: DataFrame, test: Union(DataFrame, None)) -> None:
    """Use a decision tree as a classifier"""


def main():
    with open(DATASET_PATH, 'r') as dataset_json:
        dataset_paths = json.load(dataset_json)
        train_path, val_path, test_path = dataset_paths['train'], dataset_paths['val'], dataset_paths['test'] 

        with open(train_path, 'r') as train:
            with open(val_path, 'r') as val:
                test = None
                if test_path:
                    with open(test_path, 'r') as test:
                        decision_tree_clf(train, val, test)
                else:
                    decision_tree_clf(train, val, test)

        

    


if __name__ == "__main__":
    main()
