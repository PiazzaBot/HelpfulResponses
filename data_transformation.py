""" A program which takes a raw Piazza CSV and transforms it for the good_answer classification task.

Usage:
    data_transformation.py  -v CSV-FILEPATH -s TRANSFORM-CSV-SAVEPATH  -c CLASS-YEAR
    

Options:
     CLASS-YEAR   Should be either 2018, 2019 or 2020.

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




def augment_data(data: DataFrame, save_path, folder_set: dict[str, int], add_nlp_features=True, sentiment_model=None, save_to_csv=True) -> DataFrame:

    augmented_data = []

    studentid_to_int = {}
    instructorid_to_int = {}
    num_users = 0
    #num_students, num_instructors = 0, 0

    num_iters = 0
    num_iters_thresh = 10  # len(data.index)
    #num_iters_thresh = len(data.index)

    #sentiment_pipeline = pipeline(model="cardiffnlp/twitter-roberta-base-sentiment")

    for r in tqdm(data.itertuples(), total=num_iters_thresh, colour='green', desc="Augment csv data"):

        if num_iters > num_iters_thresh:
            break

       
        new_row = [r.Index] 
        new_row.append(r.is_private)
        category = get_category(r.folders, folder_set)

        # ignore posts that are part of an unknown category
        if not category: 
            continue

        new_row.append(category)
       
        if r.student_poster_name not in studentid_to_int:
            studentid_to_int[r.student_poster_name] = hash(r.student_poster_name) 
            num_users += 1
            
        new_row.append(studentid_to_int[r.student_poster_name])
        stripped_q = ""
        raw_question = ""
        question_sentiment = NEUTRAL
        if isinstance(r.question, str): 
            stripped_q = strip_tags(r.question)
            raw_question = r.question
            if add_nlp_features:
                question_sentiment = get_sentiment(stripped_q, sentiment_model) # strip down to 512 tokens

        #new_row.append(stripped_q)
        new_row.append(get_length(stripped_q))
        new_row.append(is_references(stripped_q))
        new_row.append(level_of_detail(raw_question))

        if add_nlp_features:
            new_row.append(question_sentiment)


    
        is_followup = 1 if r.is_followup else 0
        new_row.append(is_followup)

        # add separate rows for student and instructor answer
        num_users += add_answer(augmented_data, deepcopy(new_row), r, studentid_to_int, num_users, STUDENT, add_nlp_features, sentiment_model)
        num_users+= add_answer(augmented_data, deepcopy(new_row), r, instructorid_to_int, num_users, INSTRUCTOR, add_nlp_features, sentiment_model)
        num_iters += 1
    
    augmented_data = np.array(augmented_data)
    #print(augmented_data.shape)
    if add_nlp_features:
        augmented_df = pd.DataFrame(augmented_data, columns=['post_id', 'is_private', 'category', 'student_poster_id', 
        'question_length', 'is_question_references', 'question_lod', 'question_sentiment', 'is_followup', 
        'answerer_id', 'answer_length', 'is_answer_references', 'answer_lod', 'answer_sentiment', 'response_time',  'reputation', 'is_helpful'])
    else:
        augmented_df = pd.DataFrame(augmented_data, columns=['post_id', 'is_private', 'category', 'student_poster_id', 
        'question_length', 'is_question_references', 'question_lod', 'is_followup', 
        'answerer_id', 'answer_length', 'is_answer_references', 'answer_lod', 'response_time',  'reputation', 'is_helpful'])

    if save_to_csv:
        augmented_df.index.name = 'ID'
        dir = os.path.dirname(save_path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        augmented_df.to_csv(save_path)


    return augmented_df




def run_ssl():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download('punkt')


def main(args):
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
    sentiment_pipeline = pipeline(model=MODEL, max_length=512, truncation=True)
    run_ssl()
    csv_path = args['CSV-FILEPATH']
    data = pd.read_csv(csv_path)
    year = args['CLASS-YEAR']

    folder_set =  CLASS_YEAR[year]

    augment_data(data, args['TRANSFORM-CSV-SAVEPATH'], folder_set, add_nlp_features=True, sentiment_model=sentiment_pipeline)
  
    


if __name__ == "__main__":
    arguments = docopt(__doc__)
    print(arguments)
    main(arguments)