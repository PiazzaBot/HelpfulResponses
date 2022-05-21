import json
import time
from tqdm import trange, tqdm
import os
from typing import Dict, List, Tuple, Union


from piazza_api.network import Network

def get_post_creator(post):
    for entry in post['change_log']:
        if entry['type'] == 'create':
            return entry['uid']

def get_post_created(post):
    for entry in post['change_log']:
        if entry['type'] == 'create':
            return entry['when']


def get_posts_by_student(student_id):
    student_posts = []
    with open('csc108_fall2021_json.txt', 'r') as f:
        all_posts = json.load(f)
        for p in all_posts:
            if get_post_creator(p) == student_id:
                student_posts.append(p)
    return student_posts


def export_posts_json(filename:str, course:Network) -> None:

    if os.path.exists(filename):
        print(f"{filename} already exists!")
        return 
    posts = course.iter_all_posts()
    all_posts = []
    #text = json.dumps(post['children'][1], sort_keys=True, indent=4)
    try:
        for p in tqdm(posts):
            time.sleep(1)
            all_posts.append(p)
    
    finally:
        print('------------------------------------')
        with open(filename, 'w') as f:
            json.dump(all_posts, f)

"""Custom Types"""
Answer = Dict[str,Dict[str,Union[str,int]]]
Post = Dict[str,Union[str, Union[str,int,List]]]

def get_answers(post:Post) -> Answer:
    """ Get student and instructor answers
    """

    answers = {}
    answers['s_answer'] = {}
    answers['i_answer'] = {}

    for t in answers.keys():
        for ans in post['children']:
            if ans['type'] == t:
                vals = answers[t]
                vals['text'] = ans['history'][0]['content']
                vals['poster'] = ans['history'][0]['uid']
                vals['date'] = ans['history'][0]['created']
                vals['num_helpful'] = len(ans['tag_endorse_arr'])
                break
    
    return answers



def json_to_csv(json_file_path: str, csv_filename: str, is_overwrite_csv=False):

    schema = ("id,question_title,question,folders,student_poster_name,date_question_posted," 
    "student_answer,student_answer_name,date_student_answer_posted,num_student_helpful,"
    "instructor_answer,instructor_answer_name,date_instructor_answer_posted,num_instructor_helpful" 
    "is_followup,\n")

    print(schema)

    with open(json_file_path, 'r') as json_file:
        with open(csv_filename, 'w') as csv_file:
            csv_file.write(schema)
            posts = json.load(json_file)
            for post in tqdm(posts):   
                row = [] 
                if post['type'] == 'question':
                    question = post['history'][0] # newest update of question. Change index to -1 for oldest
                    question_title = question['subject']
                    question_content = question['content']
                    folders = ','.join(post['folders'])
                    date_created = get_post_created(post)
                    answers = get_answers(post)
                    student_answer = answers['s_answer']
                    instructor_answer = answers['i_answer']
                    #print(instructor_answer)


                    row = [post['id'], question_title, question_content, folders, get_post_creator(post), date_created]
                    s_row, i_row = [], []
                    if student_answer:
                        s_row = [student_answer['text'], student_answer['poster'], student_answer['date'], str(student_answer['num_helpful'])] 
                    else:
                        s_row = ['None', 'None', 'None', 'None']

                    if instructor_answer:
                        i_row = [instructor_answer['text'], instructor_answer['poster'], instructor_answer['date'], str(instructor_answer['num_helpful'])] 
                    else:
                        i_row = ['None', 'None', 'None', 'None']
                    
                    row = row + s_row + i_row

                    is_followup = 'False'

                    for c in post['children']:
                        if c['type'] == 'followup':
                            is_followup = 'True'
                    
                    row += [is_followup]
               
                    row_string = ','.join(row)
                    csv_file.write(row_string)
                    csv_file.write('\n')
               



                    





            


