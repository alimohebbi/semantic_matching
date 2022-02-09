import glob
import os
import re

import pandas as pd
from pandas import read_csv


def get_file_name(path):
    file_name = os.path.basename(path).split('.')[0]
    return file_name


def load_csv_dir(sim_score_dir):
    google_files = {}

    topic_files = {}
    for path in glob.glob(sim_score_dir + "*.csv"):
        file_name = get_file_name(path)
        if 'hierarchy' in file_name or 'adaptdroid' in file_name or not (
                'topics' in file_name or 'googleplay' in file_name):
            continue
        csv = read_csv(path, encoding='latin-1')
        if 'topics' in file_name:
            topic_files[file_name] = csv
        else:
            google_files[file_name] = csv
    return google_files, topic_files


def find_corresponding_topic_name(file_name, topic_results):
    return file_name.replace('googleplay', 'topics')
    # return topic_results[topic_equivalent_file_name]


def get_rank_of_correct_match(group, field):
    data = group.copy()
    data['rank'] = data[field].rank(ascending=False)
    correct_index = data.loc[data['target_label'] == 'correct'].index.values.astype(int)
    if correct_index.size == 0:
        return 0
    return data[data['target_label'] == 'correct'].reset_index()['rank'][0]


def evaluate_by_mrr(google_file, topic_file):
    group_by = ['src_app', 'target_app', 'src_event_index']
    google_groups = google_file.reset_index().groupby(group_by)
    topic_groups = topic_file.reset_index().groupby(group_by)
    last_column_name = google_file.columns[-1]
    sum_reveres_rank = 0
    for name, group in google_groups:
        google_rank = get_rank_of_correct_match(group, last_column_name)
        topic_rank = get_rank_of_correct_match(topic_groups.get_group(name), last_column_name)
        if min(google_rank, topic_rank) == 0:
            print('Warning: for an event there was not match!!!')
            continue
        sum_reveres_rank += 1.0 / min(google_rank, topic_rank)
    return sum_reveres_rank / len(google_groups)


def get_acceptable_targets(x):
    src_class = x['src_class']
    src_type = x['src_type']
    text = str(x['src_text'])
    tgt_classes = [src_class]
    if src_class in ['android.widget.ImageButton', 'android.widget.Button']:
        tgt_classes = ['android.widget.ImageButton', 'android.widget.Button', 'android.widget.TextView']
    elif src_class == 'android.widget.TextView':
        if src_type == 'clickable':
            tgt_classes += ['android.widget.ImageButton', 'android.widget.Button']
            if re.search(r'https://\w+\.\w+', text):  # e.g., a15-a1x-b12
                tgt_classes.append('android.widget.EditText')
    elif src_class == 'android.widget.EditText':
        tgt_classes.append('android.widget.MultiAutoCompleteTextView')  # a43-a41-b42

    elif src_class == 'android.widget.MultiAutoCompleteTextView':  # a41-a43-b42
        tgt_classes.append('android.widget.EditText')

    return tgt_classes


def get_potential_matches(data, key):
    last_column = data.columns[-1]
    algorithm = key.split('_')[0]
    if 'craftdroid' == algorithm:
        return data[data.apply(lambda x: x['target_class'] in get_acceptable_targets(x), axis=1)]
    if 'atm' == algorithm:
        data = data[data[last_column] >= 0]
    return data[data['src_type'] == data['target_type']]


def evalute_files(google_files: dict, topic_files):
    keys = list(google_files.keys())
    keys2 = list(google_files.keys())
    mrr_values = []
    top1_values = []
    for key in keys:
        google_file = google_files[key]
        topic_file_name = find_corresponding_topic_name(key, topic_files)
        google_file = get_potential_matches(google_file, key)
        if topic_file_name in topic_files:
            topic_file = topic_files[topic_file_name]
            topic_file = get_potential_matches(topic_file, key)
            mrr = evaluate_by_mrr(google_file, topic_file)
            mrr_values.append(mrr)
            top1 = evaluate_by_top_rank(google_file, topic_file)
            top1_values.append(top1)
        else:
            keys2.remove(key)
    data = {'config': keys2, 'mrr': mrr_values, 'top1': top1_values}
    df = pd.DataFrame(data)
    return df


def is_between_top_n(group, field, top_n):
    rank = get_rank_of_correct_match(group, field)
    if rank == 0:
        return 0
    return 1 if rank <= top_n else 0


def evaluate_by_top_rank(google_file, topic_file):
    group_by = ['src_app', 'target_app', 'src_event_index']
    google_groups = google_file.reset_index().groupby(group_by)
    topic_groups = topic_file.reset_index().groupby(group_by)
    last_column_name = google_file.columns[-1]
    top_n_number = 0
    for name, group in google_groups:
        google_rank = is_between_top_n(group, last_column_name, 1)
        topic_rank = is_between_top_n(topic_groups.get_group(name), last_column_name, 1)
        top_n_number += max(google_rank, topic_rank)
    return top_n_number / len(google_groups)


def modify_columns(mrr_result):
    mrr_result['algorithm'] = ''
    mrr_result['descriptors'] = ''
    mrr_result['training_set'] = ''
    mrr_result['word_embedding'] = ''
    for i, row in mrr_result.iterrows():
        config = row['config']
        mrr_result.at[i, 'algorithm'] = config.split('_')[0]
        mrr_result.at[i, 'descriptors'] = config.split('_')[1]
        mrr_result.at[i, 'training_set'] = config.split('_')[2]
        mrr_result.at[i, 'word_embedding'] = config.split('_')[3]
    mrr_result.drop(columns=['config'], inplace=True)
    mrr_result.rename(columns={'mrr': 'MRR'}, inplace=True)
    mrr_result['training_set'] = 'complementary'


def get_complement_result():
    print('Adding complementary model')
    if os.path.isfile('complementary_model.csv'):
        mrr_result = pd.read_csv('complementary_model.csv')
    else:
        google_files, topic_files = load_csv_dir('sim_scores/')
        mrr_result = evalute_files(google_files, topic_files)
        mrr_result.to_csv('complementary_model.csv', index=False)
    modify_columns(mrr_result)
    return mrr_result
