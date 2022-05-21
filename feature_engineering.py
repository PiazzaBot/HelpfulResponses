#!/usr/local/bin/python3.9

import pandas as pd
from piazza_api import Piazza
from piazza_api.network import Network
import json
import pickle as pkl
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from time import sleep
import os
import sys
from io import StringIO

from typing import Dict, List, Tuple

from utils import *


CRED_FILE = "creds.json"

def login() -> Tuple[dict, Network]:
    """logs user into Piazza
    """

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


def main():
    #user_profile,course = login()
    json_to_csv("./csc108_fall2021.json", "csc108_fall2021.csv")
    



if __name__ == "__main__":
    main()


