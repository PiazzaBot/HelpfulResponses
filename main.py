""" 
Automate dataset generation and model predictions in a single script.

Usage:
    main.py 
    

Options:
 

Improvements:

"""

from generate_datasets import *
from model import *



def prediction_pipeline(dataset: DataSet):

    # print(f'RUNNING PIPELINE ON: {dataset.name}')

    # dataset.print_stats()
    # dataset.save_distributions(hue_name=None)
    # dataset.save_distributions(hue_name='is_helpful')
    # dataset.prune_features(select_k_best=6)

    # print(f"PERFORMING MODELING ON {dataset.name}")

    # random_forest_classification(dataset, print_summary=True, tune=True, log_path=dataset.scores_save_path)
    # random_forest_classification(dataset, print_summary=True, tune=False, log_path=dataset.scores_save_path)

    baseline(dataset, log_path=dataset.scores_save_path)



def main(args):
    fall2019_data = pd.read_csv(DATASET_DIR+'csc108_fall2019_aug.csv').drop(labels=["ID","post_id"], axis=1)
    fall2020_data = pd.read_csv(DATASET_DIR+'csc108_fall2020_aug.csv').drop(labels=["ID","post_id"], axis=1)
    fall2021_data = pd.read_csv(DATASET_DIR+'csc108_fall2021_aug.csv').drop(labels=["ID","post_id"], axis=1)

    # features to exclude from discrete features plot
    # continuous_features = {'question_length', 'answer_length', 'response_time', 'post_id', 'student_poster_id', 'answerer_id'}
    continuous_features = {'question_length', 'answer_length', 'response_time'}


    combined_data = pd.concat([fall2019_data, fall2020_data, fall2021_data], ignore_index=True)
    combined_data = combined_data.sample(frac=1, random_state=0)

    combined_data_no_ids = combined_data.drop(labels=["student_poster_id", "answerer_id"], axis=1)


    student_unbiased_dataset = DataSet(fall2020_data, fall2019_data, fall2021_data, continuous_features, 'unbiased_dataset')
    student_biased_dataset = split_dataset(combined_data, continuous_features, 'biased_dataset')

    student_biased_dataset_no_ids = split_dataset(combined_data_no_ids, continuous_features, 'biased_dataset_no_ids')
    student_unbiased_dataset_no_ids = DataSet(fall2020_data.drop(labels=["student_poster_id", "answerer_id"], axis=1), 
    fall2019_data.drop(labels=["student_poster_id", "answerer_id"], axis=1), fall2021_data.drop(labels=["student_poster_id", "answerer_id"], axis=1), 
    continuous_features, 'unbiased_dataset_no_ids')


    prediction_pipeline(student_unbiased_dataset)
    #prediction_pipeline(student_biased_dataset)
    #prediction_pipeline(student_unbiased_dataset_no_ids)
    #prediction_pipeline(student_biased_dataset_no_ids)

    


    #student_unbiased_dataset.print_stats()
    #student_biased_dataset.print_stats()
    #student_unbiased_dataset_no_ids.print_stats()
    #student_biased_dataset_no_ids.print_stats()
   
   

    #student_unbiased_dataset.save_distributions(hue_name=None)
    # student_biased_dataset.save_distributions(hue_name=None)
    # student_unbiased_dataset_no_ids.save_distributions(hue_name=None)
    # student_biased_dataset_no_ids.save_distributions(hue_name=None)

    #student_unbiased_dataset.prune_features(select_k_best=6)
    # student_biased_dataset.prune_features(select_k_best=6)
    # student_unbiased_dataset_no_ids.prune_features(select_k_best=6)
    # student_biased_dataset_no_ids.prune_features(select_k_best=6)



    #print(f"PERFORMING MODELING ON {student_unbiased_dataset.name}")
    #random_forest_classification(student_unbiased_dataset)

    #print(f"PERFORMING MODELING ON {student_biased_dataset.name}")
    #random_forest_classification(student_biased_dataset)

    



if __name__ == "__main__":
    arguments = docopt(__doc__)
    print(arguments)
    main(arguments)


    '''
    calculate proportion of students in biased dataset

    incorporate statistical tests
    '''