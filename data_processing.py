""" A program to scrape Piazza posts (represented in JSON) and convert it to a csv file.

Usage:
    data_processing.py scrapejson -c CRED-FILE -j JSON-SAVE-FILEPATH 
    data_processing.py jsontocsv  -c CRED-FILE -j JSON-LOAD-FILEPATH  -v CSV-SAVE-FILEPATH [-o]

Options:
    -o      If the loaded course instance is from 2019 or earlier

Improvements:

"""

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


import csv

from nptyping import NDArray, Int, Shape
from typing import Dict, List, Tuple, Union


from piazza_api import Piazza
from piazza_api.network import Network


from data_processing_utils import *

def export_posts_json(path:str, course:Network, max_iters='all') -> None:
    """
    Create json of all Piazza posts 

    :param path: Path to save json file
    :param max_iters: How many posts to save due to PiazzaAPI throttling
    """

    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    posts = course.iter_all_posts()
    all_posts = []
    #text = json.dumps(post['children'][1], sort_keys=True, indent=4)
    try:
        iters = 0
        for p in tqdm(posts):
            time.sleep(2.5)
            all_posts.append(p)
            iters += 1
            if iters == max_iters and max_iters != 'all':
                break
    finally:
        print('------------------------------------')
        with open(path, 'w') as f:
            json.dump(all_posts, f)




def json_to_csv(json_file_path: str, csv_filename: str, course: Network, is_overwrite_csv: bool=False, is_old=False) -> None:
    """ 
    :param json_file_path: Path to json file to convert to csv
    :param csv_filename: Name of csv file to save to cur directory
    :param course: Used to extract student profile to determine whether they are endorsed. **Actually not a valid way of checking**
    :param is_old: Toggle to true if course instance is 2019 or earlier. These posts use the 'status' field to determine whether
    a post is private
    """

    schema = ("post_id,is_private,question_title,question,folders,student_poster_name,date_question_posted," 
    "student_answer,student_answer_name,date_student_answer_posted,is_student_endorsed,is_student_helpful,"
    "instructor_answer,instructor_answer_name,date_instructor_answer_posted,is_instructor_helpful," 
    "is_followup\n")

    parser = MyHTMLParser()

    endorsed_students = get_endorsed_students(course)[0]

    with open(json_file_path, 'r') as json_file:
        with open(csv_filename, 'w') as csv_file:
            csv_file.write(schema)
            posts = json.load(json_file)
            num_posts = 0
            for post in tqdm(posts):   
                row = [] 
                if post['type'] == 'question':
                    question = post['history'][0] # newest update of question. Change index to -1 for oldest
                    question_title = question['subject']
                    question_content = question['content']
                    folders = ','.join(post['folders'])
                    date_created = get_post_created(post)
                    answers = get_answers(post, endorsed_students)
                    student_answer = answers['s_answer']
                    instructor_answer = answers['i_answer']
                 

                    row = [post['nr'], is_private(post, is_old), question_title, question_content, folders, get_post_creator(post), date_created]
                    s_row, i_row = [], []
                    if student_answer:
                        s_row = [student_answer['text'], student_answer['poster'], student_answer['date'], str(student_answer['is_endorser']), str(student_answer['is_helpful'])] 
                    else:
                        s_row = [None, None, None, None, None]

                    if instructor_answer:
                        i_row = [instructor_answer['text'], instructor_answer['poster'], instructor_answer['date'], str(instructor_answer['is_helpful'])] 
                    else:
                        i_row = [None, None, None, None]
                    
                    row = row + s_row + i_row

                    is_followup = 'False'

                    for c in post['children']:
                        if c['type'] == 'followup':
                            is_followup = 'True'
                    
                    row += [is_followup]

                    post_writer = csv.writer(csv_file)
                    post_writer.writerow(row)
                    
                    csv_file.write('\n')

                    num_posts += 1



def main(args):
    cred_file_path = args['CRED-FILE']
    if args['scrapejson']:
        json_save_path = args['JSON-SAVE-FILEPATH']
        user_profile, course = login(cred_file_path)
        export_posts_json(json_save_path, course, max_iters='all')
     
    elif args['jsontocsv']:
        old = False
        if args['-o']:
            old = True
        user_profile, course = login(cred_file_path)
        json_loadpath = args['JSON-LOAD-FILEPATH']
        csv_savepath = args['CSV-SAVE-FILEPATH']
        json_to_csv(json_loadpath, csv_savepath, course, is_old=old)
    else:
        assert(1 == 0)


if __name__ == "__main__":
    arguments = docopt(__doc__)
    print(arguments)
    main(arguments)
   
