import numpy as np
import json
import re
import pandas as pd
import os

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'


def replace_nan_with(df, columns, value):
    for column in columns:
        df[column].loc[df[column].isnull()] = df[column].loc[df[column].isnull()
                                                             ].apply(lambda x: value)


def load_data(file_path, format='tsv'):
    if format == 'tsv':
        df = pd.read_csv(file_path, delimiter='\t')
    elif format == 'json':
        data = json.load(file_path)
        if 'items' in data:
            df = pd.DataFrame(data['items'])
        else:
            df = pd.DataFrame(data)
    else:
        df = pd.read_csv(file_path)
    print("Reading a df with columns: ", df.columns.tolist())
    df.dropna(subset=['snippet'], inplace=True)
    df = df.sort_values(['snippet', 'classification'])
    df.drop_duplicates(subset=['snippet'], keep='first', inplace=True)
    df = df.sort_index().reindex()

    return df


def callback(score, epoch, steps):
    print(f"Score at epoch {epoch}, step {steps}: {score}")


def create_replace_no_tags_embeddings(df):
    def remove_hashtags(text):
        words = text.split()
        words = [word for word in words if not word.startswith('#')]
        return ' '.join(words)
    df['embedding_input'] = df['replace'].apply(remove_hashtags)
    df['embedding_input'] = df['embedding_input'].str.replace(r'\s+', ' ')
    return df


def replace_hashtags_in_named_entity_tags(df):
    def replace_hashtags(text):
        words = text.split()
        words = [word if not word.startswith(
            '#') else "<"+re.sub(r'[^A-Z]', '', word)+">" for word in words]
        return ' '.join(words)
    df['embedding_input'] = df['replace'].apply(replace_hashtags)
    return df


def str_to_numpy_converter(instr):
    return np.fromstring(instr[1:-1], sep=' ')
