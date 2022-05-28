#!/usr/local/bin/python3.9

import json
import time
from tqdm import trange, tqdm
import os
from typing import Dict, List, Tuple, Union

import csv
import unicodedata
import html
from io import StringIO
from html.parser import HTMLParser
from piazza_api import Piazza

from piazza_api.network import Network

CRED_FILE = "creds.json"


"""Custom Types"""
Answer = Dict[str,Dict[str,Union[str,int]]]
Post = Dict[str,Union[str, Union[str,int,List]]]


class MyHTMLParser(HTMLParser):
    """taken from: [1]"""
    
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.text = StringIO()
    def handle_data(self, d):
        self.text.write(d)
    def get_data(self):
        return self.text.getvalue()

def strip_tags(html):
    """strips html tags and substitutes html entities """
    #html = html.unescape(html)
    s =  MyHTMLParser()
    s.feed(html)
    return s.get_data()





def login() -> Tuple[dict, Network]:
    """logs user into Piazza"""

    email:str 
    password:str 
    courseid:str 

    with open(CRED_FILE) as f:
        creds = json.load(f)
        email, password, courseid = creds['email'], creds['password'], creds['courseid']


    print(f"email: {email} \npassword: {password} \ncourseid: {courseid}")


    p: Piazza = Piazza()
    p.user_login(email, password)
    user_profile: dict = p.get_user_profile()
    course: Network = p.network(courseid)
    return user_profile, course




def get_post_creator(post):
    for entry in post['change_log']:
        if entry['type'] == 'create':
            return entry['uid']


def get_post_created(post):
    """get time post was created"""
    for entry in post['change_log']:
        if entry['type'] == 'create':
            return entry['when']


def get_posts_by_student(filename:str, student_id):
    student_posts = []
    with open(filename, 'r') as f:
        all_posts = json.load(f)
        for p in all_posts:
            if get_post_creator(p) == student_id:
                student_posts.append(p)
    return student_posts


def export_posts_json(filename:str, course:Network) -> None:
    """Create json of all posts saved in current directory"""

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
                text = ans['history'][0]['content']
                text = strip_tags(text)
                vals['text'] = text
                vals['poster'] = ans['history'][0]['uid']
                vals['date'] = ans['history'][0]['created']
                vals['num_helpful'] = len(ans['tag_endorse_arr'])
                break
    
    return answers



def json_to_csv(json_file_path: str, csv_filename: str, is_overwrite_csv=False):

    schema = ("post_id,question_title,question,folders,student_poster_name,date_question_posted," 
    "student_answer,student_answer_name,date_student_answer_posted,num_student_helpful,"
    "instructor_answer,instructor_answer_name,date_instructor_answer_posted,num_instructor_helpful," 
    "is_followup\n")

    print(schema)

    parser = MyHTMLParser()

    with open(json_file_path, 'r') as json_file:
        with open(csv_filename, 'w') as csv_file:
            csv_file.write(schema)
            posts = json.load(json_file)
            for post in tqdm(posts):   
                row = [] 
                if post['type'] == 'question':
                    question = post['history'][0] # newest update of question. Change index to -1 for oldest
                    #question_title =  html.unescape(question['subject'])
                    question_title = strip_tags(question['subject'])
                    question_content = strip_tags(question['content'])
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
                        s_row = [None, None, None, None]

                    if instructor_answer:
                        i_row = [instructor_answer['text'], instructor_answer['poster'], instructor_answer['date'], str(instructor_answer['num_helpful'])] 
                    else:
                        i_row = [None, None, None, None]
                    
                    row = row + s_row + i_row

                    is_followup = 'False'

                    for c in post['children']:
                        if c['type'] == 'followup':
                            is_followup = 'True'
                    
                    row += [is_followup]
                    #print(row)
                    post_writer = csv.writer(csv_file)
                    post_writer.writerow(row)
                    
                    csv_file.write('\n')
                   
               




def main():
    """
    How to handle posts with imgs? Do we want the img tags stripped? Think about how it will affect textual features
        response length, sentiment, ...

        what elements do q&a contain?
            latex, code snippets, imgs/screenshots, links, lists, annotations to prev posts (i.e. @356)

        fields that can be added: num_answer_imgs, ...

        can remove posts with imgs or include a special field called "num_imgs" so can distinguish b/w posts that have imgs
    """
    #user_profile,course = login()
    #export_posts_json("csc108_fall2021.json", course)
    json_to_csv("./csc108_fall2021.json", "csc108_fall2021.csv")
    
    



if __name__ == "__main__":
    main()


"""
References

[1] https://stackoverflow.com/questions/753052/strip-html-from-strings-in-python
"""

            


