import json
import time
import math
from datetime import datetime
import os
from tqdm import trange, tqdm
import re
from copy import deepcopy
import sys


from io import StringIO
from html.parser import HTMLParser



from nptyping import NDArray, Int, Shape
from typing import Dict, List, Tuple, Union

from piazza_api import Piazza
from piazza_api.network import Network

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

def login(cred_filepath: str) -> Tuple[dict, Network]:
    """logs user into Piazza"""

    email:str 
    password:str 
    courseid:str 

    with open(cred_filepath) as f:
        creds = json.load(f)
        email, password, courseid = creds['email'], creds['password'], creds['courseid']


    #print(f"email: {email} \npassword: {password} \ncourseid: {courseid}")


    p: Piazza = Piazza()
    p.user_login(email, password)
    user_profile: dict = p.get_user_profile()
    course: Network = p.network(courseid)
    return user_profile, course


def get_post_creator(post: Post):
    for entry in post['change_log']:
        if entry['type'] == 'create':
            if 'uid' in entry:
                return entry['uid']
            else:
                return None


def get_post_created(post: Post):
    """get time post was created"""
    for entry in post['change_log']:
        if entry['type'] == 'create':
            return entry['when']


def get_posts_by_student(filename:str, student_id:str) -> List[Post]:
    student_posts = []
    with open(filename, 'r') as f:
        all_posts = json.load(f)
        for p in all_posts:
            if get_post_creator(p) == student_id:
                student_posts.append(p)
    return student_posts


def get_endorsed_students(course: Network) -> Tuple[Dict, Dict]:
    endorsed_users = {}
    non_endorsed_users = {}
    users = course.get_all_users()
    for u in users:
        if u['endorser']:
            endorsed_users[u['id']] = u['name']
        else:
            non_endorsed_users[u['id']] = u['name']


    return endorsed_users, non_endorsed_users


def is_private(post: Post, is_old=False) -> bool:
    """ Return true if post is private """
    if is_old: # use 'status' field of post to determine whether post is Private
        return True if post['status'] == 'private' else False

    #print(json.dumps(post['change_log'], indent=4, sort_keys=True))
    for entry in post['change_log']:
        print(entry)
        if entry['type'] == 'create':
            return True if entry['v'] == 'private' else False


def get_answers_simple(post:Post):
    answers = {}
    answers['s_answer'] = {}
    answers['i_answer'] = {}

    for t in answers.keys():
        for ans in post['children']:
            if ans['type'] == t:      
                vals = answers[t]
                text = ans['history'][0]['content']
                #text = strip_tags(text)
                vals['text'] = text
    
    return answers



def get_answers(post:Post, endorsed_students: Dict) -> List[Dict[str, Answer]]:
    """ Get student and instructor answers """

    answers = {}
    answers['s_answer'] = {}
    answers['i_answer'] = {}

    for t in answers.keys():
        for ans in post['children']:
            if ans['type'] == t:      
                vals = answers[t]
                text = ans['history'][0]['content']
                #text = strip_tags(text)
                vals['text'] = text
                if 'uid' in ans['history'][0]:
                    vals['poster'] = ans['history'][0]['uid']
                else:
                     vals['poster'] = None
                vals['date'] = ans['history'][0]['created']
                vals['num_helpful'] = len(ans['tag_endorse_arr'])
                # post creator is same student that liked response
                if get_post_creator(post) in ans['tag_endorse_arr']:
                    vals['is_helpful'] = True 
                else:
                    vals['is_helpful'] = False

                if ans['type'] == "s_answer":
                    
                    if 'uid' in ans['history'][0]:
                        student_poster_id = ans['history'][0]['uid'] # id of the most recent student answer editor
                    else:
                        student_poster_id = None

                     # check if student is endorsed (actually not a valid way of checking)
                    vals['is_endorser'] = False
                    if student_poster_id in endorsed_students:
                        vals['is_endorser'] = True
                   
                break
    
    return answers