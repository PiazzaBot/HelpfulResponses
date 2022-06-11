import json
import time
import math
from datetime import datetime
import os
from tqdm import trange, tqdm
import re
from copy import deepcopy
import sys


from nptyping import NDArray, Int, Shape
from typing import Dict, List, Tuple, Union


from piazza_api import Piazza
from piazza_api.network import Network



def login(email:str, password:str, courseid: str) -> Tuple[dict, Network]:
    """logs user into Piazza"""

    p: Piazza = Piazza()
    p.user_login(email, password)
    user_profile: dict = p.get_user_profile()
    course: Network = p.network(courseid)
    return user_profile, course



def export_posts_json(filename:str, course:Network, max_iters='all') -> None:
    """Create json of all posts saved in current directory"""

    if os.path.exists(filename):
        print(f"{filename} already exists!")
        return 
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
        with open(filename, 'w') as f:
            json.dump(all_posts, f)




def print_usage():
    usage = "Usage: python3.9 export_posts_json.py <cred-file-path> <save-json-filename>"
    print(usage)


def main():
    if len(sys.argv) != 3:
        print_usage()
        return
    cred_file_path = sys.argv[1]
    filename = sys.argv[2]
    with open(cred_file_path) as f:
        creds = json.load(f)
        email, password, courseid = creds['email'], creds['password'], creds['courseid']
        user_profile, course = login(email, password, courseid)
        export_posts_json(filename, course)


if __name__ == "__main__":
    main()
