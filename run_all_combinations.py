import logging
import os
from statistics import median
import time
import pandas as pd

from complementary_models_exp import get_complement_result
from config import Config
from descriptor_processes.text_pre_process import download_nltk_packages
from evaluators.evaluator_builder import EvaluatorBuilder

config = Config()
logging.basicConfig(level=logging.WARNING)

metrics = ['top1', 'MRR', 'time', 'zeros']


def initial_result():
    result = {}
    for i in metrics:
        result[i] = []
    return result


def set_average_result(result):
    for i in metrics:
        semantic_config[i] = median(result[i])


def evaluate_config(df):
    result = initial_result()
    for j in range(config.eval_repeat):
        start_time = time.time()
        builder.set_semantic_config(semantic_config)
        evaluator = builder.build()
        set_top_one(evaluator, result)
        result['MRR'].append(evaluator.evaluate_by_mrr()[semantic_config['word_embedding']])
        result['time'].append(time.time() - start_time)
        if not result['zeros']:
            result['zeros'].append(evaluator.technique.zeros if evaluator.technique else -1)
    set_average_result(result)
    return df.append(semantic_config, ignore_index=True)


def set_top_one(evaluator, result):
    q_number = 337
    evaluator.config.top_n = 1
    result['top1'].append(evaluator.evaluate_by_top_rank()[semantic_config['word_embedding']]/q_number)


def save(df):
    header = False if os.path.exists(config.save_rank_results_path) else True
    columns_to_save = ['algorithm', 'descriptors', 'training_set', 'word_embedding']
    columns_to_save.extend(metrics)
    df = df[columns_to_save]
    df.to_csv(config.save_rank_results_path, index=False, mode='a', header=header)


def get_existing_config_list():
    if os.path.exists(config.save_rank_results_path):
        existing_data = pd.read_csv(config.save_rank_results_path, encoding='latin-1')
        columns_exist = list(existing_data.columns)
        dont_care = metrics
        config_col = [i for i in columns_exist if i not in dont_care]
        concated_columns = existing_data[config_col].agg('-'.join, axis=1)
        return [i for i in concated_columns]


def forbidden_config(semantic_config):
    if semantic_config['word_embedding'] in ['jaccard', 'edit_distance', 'random']:
        return semantic_config['training_set'] != 'empty'
    if semantic_config['word_embedding'] in ['use', 'nnlm', 'bert']:
        return semantic_config['training_set'] != 'standard'
    if semantic_config['word_embedding'] not in ['jaccard', 'edit_distance', 'random']:
        return semantic_config['training_set'] == 'empty'


def run():
    global df
    config_key = '-'.join([algorithm, descriptor, train_set, embedding])
    if not existing_configs or config_key not in existing_configs:
        if forbidden_config(semantic_config):
            return
        df = evaluate_config(df)
        save(df)
        df.drop(df.index, inplace=True)
        print(config_key + ' : ' + str(semantic_config['top1']) + ' MRR: ' + str(semantic_config['MRR']))
    else:
        print(config_key + ' already exist')


def add_complementary():
    results = pd.read_csv(config.save_rank_results_path)
    c_results = get_complement_result()
    final_results = pd.concat([results, c_results])
    final_results.to_csv('final.csv')
    final_results[['MRR','training_set']].groupby('training_set').describe()['MRR'].to_csv('table_mrr.csv')
    final_results[['top1','training_set']].groupby('training_set').describe()['top1'].to_csv('table_top1.csv')


if __name__ == "__main__":
    print('Check nltk')
    download_nltk_packages()
    print('Start metric calculation for configurations')
    builder = EvaluatorBuilder()
    columns = ["word_embedding", "training_set", "algorithm", "description"]
    df = pd.DataFrame(columns=columns)
    existing_configs = get_existing_config_list()
    for embedding in config.active_techniques:
        for train_set in config.train_sets:
            for algorithm in config.algorithm:
                for descriptor in config.descriptors:
                    semantic_config = {"algorithm": algorithm, "descriptors": descriptor, "training_set": train_set,
                                       'word_embedding': embedding}
                    run()
    add_complementary()
    print('Done!')
