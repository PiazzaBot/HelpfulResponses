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


from io import StringIO
from html.parser import HTMLParser

from nltk.tokenize import word_tokenize

from nptyping import NDArray, Int, Shape
from typing import Dict, List, Tuple, Union

from piazza_api import Piazza
from piazza_api.network import Network

from data_processing_utils import *



"""Custom Types"""
Answer = Dict[str,Dict[str,Union[str,int]]]
Post = Dict[str,Union[str, Union[str,int,List]]]

"""Macros"""
# who the answer is coming from
STUDENT, INSTRUCTOR, STUDENT_ENDORSED_ANSWERER = 0, 1, 2
EPSILON = 1e-05

# folder categories
GENERAL, LECTURES, ASSIGNMENTS, TESTS = 0, 1, 2, 3

# labels for sentiment
NEGATIVE, NEUTRAL, POSITIVE = 0, 1, 2


csc108_fall2021_categories = {'general': GENERAL, 'utm/life/other': GENERAL, 'lecture': LECTURES, 'lab': ASSIGNMENTS, 'tests/exam': TESTS}

csc108_fall2020_categories = {'general': GENERAL, 'administrative': GENERAL, 'post/uni-life': GENERAL, 'random': GENERAL, 'lecture': LECTURES, 
'minor_labs': ASSIGNMENTS, 'major_lab1': ASSIGNMENTS, 'major_lab2': ASSIGNMENTS, 'major_lab3': ASSIGNMENTS, 'major_lab4': ASSIGNMENTS, 
'major_lab5': ASSIGNMENTS, 'a1': ASSIGNMENTS, 'a2': ASSIGNMENTS, 'a3': ASSIGNMENTS, 'test1': TESTS, 'test2': TESTS, 'exam': TESTS}

csc108_fall2019_categories = {'general': GENERAL, 'misc': GENERAL, 'lecture': LECTURES, 'lab': ASSIGNMENTS, 'a0': ASSIGNMENTS, 
'a1': ASSIGNMENTS, 'a2': ASSIGNMENTS, 'a3': ASSIGNMENTS,  'midterm': TESTS, 'exam': TESTS}

CLASS_YEAR = {
    '2019' : csc108_fall2019_categories,
    '2020' : csc108_fall2020_categories,
    '2021' : csc108_fall2021_categories
}


def get_length(text:str) -> int:
    length = 0
    if isinstance(text, str):
        length = len(word_tokenize(text))
    else: # must be nan
        if not isnan(text):
            assert(1 == 0) # shouldn't get here
    
    return length


def is_references(text: str) -> bool:
    """ Check if answer contains a link to another post (i.e. @256) or a hyperlink
        :param text: question|answer with html stripping 
    """   
    return True if re.search(r'@+\d', text) or 'http' in text else False

def level_of_detail(text: str) -> bool:
    """ Detect imgs or code snippets
        :param text: raw question|answer without html stripping so can detect imgs/code-snippets
    """
    is_image = True if '<img' in text else False
    is_code_snippets = True if '<pre' in text else False

    return True if is_image or is_code_snippets else False


def answer_response_time(t1:str, t2:str) -> int:
    """:returns: Answer response time in mins, rounded up. Add option to use log scale?"""
    d1 = datetime.fromisoformat(t1[:-1])
    d2 = datetime.fromisoformat(t2[:-1])
    delta = d2-d1
    response_time = math.ceil(delta.total_seconds() // 60)
    if response_time == 0:
        response_time = EPSILON
    return response_time
    
def get_category(folder: str, folder_set: dict[str, int]) -> Union[str, bool]:
    """ For multi-categories just choose the 1st one.
        Varies b/w classes.

        :param folder: folder1,folder2, ...
        :param folder_set: set of all folders

        Precondition: folder should be part of folder_set

    """
    category = folder.split(',')[0]
    if category not in folder_set:
        return False

    return folder_set[category]


def get_sentiment(text: str, sentiment_model) -> int:
    senti = sentiment_model(text)[0]['label']
    if senti == 'LABEL_0':
        return NEGATIVE
    elif senti == 'LABEL_1':
        return NEUTRAL
    else:
        return POSITIVE
 

    

def add_answer(augmented_data:List[List], append_row:List, post_row:tuple, poster_dict: Dict[str, str], num_instances:int, answer_type:int,
              add_nlp_features=True, sentiment_model=None) -> int:
    """
    Append student or instructor answer fields to augmented_data.

    :param augmented_data: table to add append_row to
    :param append_row: partially filled row to be completed
    :param post_row: namedtuple containing information about the current Piazza post
    :param answer_type: INSTRUCTOR|STUDENT
    :returns: this is a description of what is returned
    :raises Nothing
    """
    poster =  'student' if answer_type == STUDENT  else 'instructor'
    fields = [f'{poster}_answer', f'{poster}_answer_name', f'is_{poster}_helpful', 'date_{poster}_answer_posted']
    increment_num_instances = False
    
    answer = getattr(post_row, f"{poster}_answer")
   
    if isinstance(answer, str): # if answer is nan, don't add an entry
        stripped_answer = strip_tags(answer)
        poster_id = getattr(post_row, f'{poster}_answer_name')
        if poster_id not in poster_dict:
            poster_dict[poster_id] = hash(poster_id)
            increment_num_instances = True

        is_helpful = 1 if getattr(post_row, f"is_{poster}_helpful") else 0

        
        #append_row.append(stripped_answer)
        append_row.append(poster_dict[poster_id])
        append_row.append(get_length(stripped_answer))
        append_row.append(is_references(stripped_answer))
        append_row.append(level_of_detail(answer))

        if add_nlp_features:
            append_row.append(get_sentiment(stripped_answer, sentiment_model))

        t1 = getattr(post_row, 'date_question_posted')
        t2 = getattr(post_row, f'date_{poster}_answer_posted')
        response_time = answer_response_time(t1, t2)


        append_row.append(response_time)
        
        append_row.append(answer_type)
        append_row.append(is_helpful)
        augmented_data.append(append_row)

    return increment_num_instances

    
#'general,lecture'.split(',')
    