""" 
Automate dataset generation and model predictions in a single script.

Usage:
    main.py 
    

Options:
 

Improvements:

"""

from generate_datasets import *
from model import *



def main(args):
    fall2019_data = pd.read_csv(DATASET_DIR+'csc108_fall2019_aug.csv').drop(labels=["ID","post_id"], axis=1)
    fall2020_data = pd.read_csv(DATASET_DIR+'csc108_fall2020_aug.csv').drop(labels=["ID","post_id"], axis=1)
    fall2021_data = pd.read_csv(DATASET_DIR+'csc108_fall2021_aug.csv').drop(labels=["ID","post_id"], axis=1)

    # features to exclude from discrete features plot
    #continuous_features = {'question_length', 'answer_length', 'response_time', 'post_id', 'student_poster_id', 'answerer_id'}
    continuous_features = {'question_length', 'answer_length', 'response_time'}


    combined_data = pd.concat([fall2019_data, fall2020_data, fall2021_data], ignore_index=True)
    combined_data = combined_data.sample(frac=1, random_state=0)

    student_unbiased_dataset = DataSet(fall2020_data, fall2019_data, fall2021_data, continuous_features, 'unbiased_dataset')
    student_biased_dataset = split_dataset(combined_data, continuous_features, 'biased_dataset')

    student_unbiased_dataset.print_stats()
    student_biased_dataset.print_stats()
   

    student_unbiased_dataset.save_distributions(hue_name=None)
    student_biased_dataset.save_distributions(hue_name=None)

    student_unbiased_dataset.prune_features(select_k_best=6)
    student_biased_dataset.prune_features(select_k_best=6)

    



if __name__ == "__main__":
    arguments = docopt(__doc__)
    print(arguments)
    main(arguments)


    '''
    calculate proportion of students in biased dataset
    '''