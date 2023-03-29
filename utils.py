#  util functions for rl training

import pandas as pd
import numpy as np
import ast
import re
from gensim.models import Word2Vec
from gensim.models import FastText
from matplotlib import pyplot as plt


def process_test_data(df):
    """

    :param df: test data frame with student responses and ratings
    :return: processed
    """

    df = df.reset_index()
    df['dummy_score'] = df['1'].astype(str) + ', ' + df['2'].astype(str) + ', ' + df['3'].astype(str) + ', ' + df['4'].astype(str) + ', ' + df['5'].astype(str)
    df = df.drop(columns={'Unnamed: 0', '1', '2', '3', '4', '5', 'index'})
    df.dropna(inplace=True)
    df = df[~df['text'].apply(lambda x: len(str(x).replace('.', '').split()) < 2)]
    df['text_len'] = df['text'].apply(lambda x: len(x.split(' ')))
    df = df[df['text_len'] < 50]  # get rid of answers that are too short
    df['text'] = df['text'].apply(lambda x: re.sub("[^0-9a-zA-Z]+", " ", x))
    df['text'] = df['text'].apply(lambda x: str(x).replace('.', ' ').replace(',', ' ').replace('-', ' ').replace('6. ', '').replace('  ', ' '))
    df = df.drop(columns=['text_len'])
    df.reset_index(inplace=True, drop=True)
    df = df[df['dummy_score'] != '1, 0, 0, 0, 0']  # filter out if rated lowest score

    return df


def process_action_space(df):
    """

    :param df: action space df
    :return: list actions (strings)
    """

    df = pd.DataFrame(df['action_list'].dropna())
    df['action_list2'] = df['action_list'].apply(lambda x: ast.literal_eval(x))
    list_of_actions = [list(x) for x in [x for x in df['action_list2']][0]]
    action_sentences = [' '.join(x) for x in list_of_actions]

    return action_sentences


def train_w2v_model(processed_df, key_phrase_list, vector_size, model='w2v'):
    """

    :param processed_df: output of process_test_data function
    :param key_phrase_list: output of process_action_space function
    :return: word2vec model trained on our data specifically
    """
    sentences = [x.lower().split() for x in processed_df['text']]
    all_sentences = sentences + key_phrase_list
    if model == 'w2v':
        return_model = Word2Vec(all_sentences, min_count=1, vector_size=vector_size)
    elif model == 'fasttext':
        return_model = FastText(all_sentences, vector_size=vector_size, window=5, min_count=1)
    return return_model


def read_scores_df(path):
    scores_df = pd.read_csv(path)
    return [x for x in scores_df['0']]


def plot_rewards(scores, path, name, rolling_n=300):
    if not scores:
        scores = read_scores_df(path)

    plt.style.use('seaborn-whitegrid')
    rolling = pd.DataFrame(scores).rolling(rolling_n).mean()
    y = np.array(rolling[rolling_n:])
    x = np.array(list(range(len(y))))
    plt.plot(x, y)
    plt.savefig(name + '.png')
    return None
