import sys
from torch.autograd import Variable
import torch
# from skipthoughts import UniSkip, BiSkip
import sklearn
import sklearn.feature_extraction
import numpy as np
# import skipthoughts
import gensim.downloader as api
import numpy as np
from numpy import dot
from numpy.linalg import norm
import json
import pickle
from collections import Counter

import pandas as pd
from sentence_transformers import SentenceTransformer
import pycountry
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

sys.path.append('skip-thoughts.torch/pytorch')


def load_model(model="sentence-transformers/all-MiniLM-L12-v2"):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.has_mps else 'cpu'
    sbert_model = SentenceTransformer(model, device=device)
    return sbert_model


sbert_model = load_model()


def aggregate_embeddings(keys, embeddings_dict, to_lower=True):
    embeddings = []
    for key in keys:
        if to_lower:
            key = key.lower()
        embeddings.append(embeddings_dict[key])
    return np.mean(embeddings, axis=0)


def reverse_dictionary(dict):
    '''
    http://stackoverflow.com/questions/483666/python-reverse-inverse-a-mapping
    '''
    return {v: k for k, v in dict.items()}


# def encode_skipthoughts(snippets, model_name='uniskip'):
#     dir_st = 'data/skip-thoughts'
#     vectorizer = sklearn.feature_extraction.text.CountVectorizer(min_df=1)

#     corpus = list(snippets)

#     X = vectorizer.fit_transform(corpus).toarray()

#     tokenizer = vectorizer.build_tokenizer()
#     word2idx = vectorizer.vocabulary_
#     # print('word2idx: {0}'.format(word2idx))
#     max_value = max(value for value in word2idx.values())
#     empty_token = max_value+1
#     word2idx[''] = empty_token
#     output_corpus = []
#     max_length = max(len(line.split())
#                      for line in corpus)  # Find the length of the longest list
#     for line in corpus:
#         line = tokenizer(line.lower())
#         output_line = np.array(
#             [vectorizer.vocabulary_.get(token) for token in line])
#         padded_line = np.pad(
#             output_line, (0, max_length - len(output_line)), constant_values=empty_token)
#         output_corpus.append(padded_line)

#     vocab = list(word2idx.keys())
#     if model_name == 'uniskip':
#         model = UniSkip(dir_st, vocab)
#     elif model_name == 'biskip':
#         model = BiSkip(dir_st, vocab)
#     else:
#         raise ValueError('model_name must be either uniskip or biskip')

#     inp = Variable(torch.LongTensor(output_corpus))  # <eos> token is optional
#     print(inp.size())  # batch_size x seq_len
#     lengths = [len(lst) for lst in output_corpus]
#     output_seq2vec = model(inp, lengths=lengths)
#     return output_seq2vec


def encode_glove(snippets):
    # Load Pretrained Glove Embedding
    glove_dim = 300
    glove_wiki = api.load(f"glove-wiki-gigaword-{glove_dim}")

    def get_GloVe(text, size, vectors, aggregation='mean'):
        # create for size of glove embedding and assign all values 0
        vec = np.zeros(size).reshape((glove_dim,))
        count = 0
        for word in text.split():
            try:
                # update vector with new word
                vec += vectors[word].reshape((glove_dim,))
                count += 1  # counts every word in sentence
            except KeyError:
                continue
        if aggregation == 'mean':
            if count != 0:
                vec /= count  # get average of vector to create embedding for sentence
            return vec
        elif aggregation == 'sum':
            return vec

    embeddings = []
    for sent in snippets:
        embeddings.append(get_GloVe(sent, glove_dim, glove_wiki))

    return np.array(embeddings)


def encode_s3bert(snippets):
    model = SentenceTransformer(
        "S3BERT/src/s3bert_all-MiniLM-L12-v2", device=device)

    embeddings = model.encode(snippets)
    return embeddings


def country_code_map():
    device = 'mps'
    sbert_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L12-v2", device=device)
    code_to_country = {}
    organizations = {'EU': 'European Union'}
    for country in pycountry.countries:
        code_to_country[country.alpha_2] = sbert_model.encode(
            country.name)
    for org_abbreviation in organizations.keys():
        code_to_country[org_abbreviation] = sbert_model.encode(
            organizations[org_abbreviation])
    return code_to_country

# function to calculate cosine similarity


def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

# Function to check if 'snippet1' is a substring of 'snippet2' or vice versa


def check_substring(row):
    if row['snippet1'] in row['snippet2'] or row['snippet2'] in row['snippet1']:
        return 1.0
    else:
        # return the existing similarity if neither is a substring of the other
        return row['similarity']


def print_similarity_samples(df, similarity_column, sample_size=10):
    df = df.dropna(subset=[similarity_column])
    sorted_df = df.sort_values(by=similarity_column, ascending=False)
    display_df = sorted_df[['snippet1', 'snippet2', similarity_column]]

    intervals = np.arange(0, 1.1, 0.1)
    result_df = pd.DataFrame()
    for i in range(len(intervals) - 1):
        # Get start and end of interval
        start = intervals[i]
        end = intervals[i + 1]

        # Select rows where similarity is within the interval
        mask = (display_df[similarity_column] >= start) & (
            display_df[similarity_column] < end)
        selected_rows = display_df[mask]

        # If more than 10 rows are selected, randomly choose 10
        if selected_rows.shape[0] > sample_size:
            selected_rows = selected_rows.sample(n=sample_size)

        # Append the selected rows to the result DataFrame
        result_df = pd.concat([result_df, selected_rows])

    # Reset index of the result DataFrame
    result_df = result_df.reset_index(drop=True)

    # Iterate over the first 10 rows
    for index, row in result_df.iterrows():
        print(row['snippet1'])
        print(row['snippet2'])
        print(row[similarity_column])
        print("\n---\n")  # print a line for separation


def replace_nan_with(df, columns, value):
    for column in columns:
        df[column].loc[df[column].isnull()] = df[column].loc[df[column].isnull()
                                                             ].apply(lambda x: value)


def load_ai_news():
    prefix = '../glanos-data/datasets/'
    tsv_path = f"{prefix}ai_news.tsv"

    df = pd.read_csv(tsv_path, delimiter='\t')
    df.dropna(subset=['snippet'], inplace=True)
    df = df.sort_values(['snippet', 'classification'])
    df.drop_duplicates(subset=['snippet'], keep='first', inplace=True)
    df = df.sort_index().reindex()

    return df


def load_car_news():
    prefix = '../glanos-data/datasets/'
    tsv_path = f"{prefix}car_news.tsv"

    df = pd.read_csv(tsv_path, delimiter='\t')
    df.dropna(subset=['snippet'], inplace=True)
    df = df.sort_values(['snippet', 'classification'])
    df.drop_duplicates(subset=['snippet'], keep='first', inplace=True)
    df = df.sort_index().reindex()

    return df


def load_big_consulting_export():
    prefix = '../glanos-data/datasets/'
    with open(f'../glanos-data/embeddings/big_consulting_export-sbert.pickle', 'rb') as f:
        embeddings = pickle.load(f)

    json_file_path = f"{prefix}big_consulting_export.json"
    tsv_path = f"{prefix}big_consulting_export.tsv"

    with open(json_file_path, 'r') as j:
        contents = json.loads(j.read())
    df_details = pd.read_csv(tsv_path, delimiter='\t')
    df_details = df_details.drop(columns=['snippet', 'date'])

    # Get the list of item objects
    items = contents['items']
    df = pd.DataFrame(items)
    df["embedding"] = df["snippet"].map(embeddings)

    df = pd.concat([df, df_details], axis=1)
    df.dropna(subset=['snippet'], inplace=True)

    df['embedding'] = df['embedding'].apply(
        lambda x: np.array(x))  # should it be here?
    df = df.sort_values(['snippet', 'relationEntity'])

    df.drop_duplicates(subset=['snippet'], keep='first', inplace=True)
    df = df.sort_index().reindex()

    return df


def prepare_datasets(df):
    '''
    df is the dataframe containing big_consulting_export
    '''
    input_prefix = 'similarity-training-data/'
    output_prefix = 'dataset/'
    # Clean the train, test and val dataset to only include the original columns
    test_labeled_df = pd.read_csv(f'{input_prefix}test_labeled.tsv', sep='\t').drop(columns=[
        'country_similarity', 'company_similarity', 'similarity_range', 'companies1', 'companies2', 'country1', 'country2']).rename(columns={'snippet1': 'snippet-1', 'snippet2': 'snippet-2'})
    val_labeled_df = pd.read_csv(f'{input_prefix}val_labeled.tsv', sep='\t').drop(columns=[
        'country_similarity', 'company_similarity', 'similarity_range', 'companies1', 'companies2', 'country1', 'country2']).rename(columns={'snippet1': 'snippet-1', 'snippet2': 'snippet-2'})
    train_df = pd.read_csv(f'{input_prefix}train.tsv', sep='\t').drop(columns=[
        'country_similarity', 'company_similarity', 'similarity_range', 'companies1', 'companies2', 'country1', 'country2']).rename(columns={'snippet1': 'snippet-1', 'snippet2': 'snippet-2'})

    def clean_df(df_to_clean):
        df_to_clean = df_to_clean.merge(
            df, left_on='snippet-1', right_on='snippet', how='left')

        df_to_clean = df_to_clean.rename(
            columns={col: col + '1' for col in df.columns})

        df_to_clean = df_to_clean.merge(
            df, left_on='snippet-2', right_on='snippet', how='left')

        df_to_clean = df_to_clean.rename(
            columns={col: col + '2' for col in df.columns})
        df_to_clean = df_to_clean.drop('snippet1', axis=1).rename(
            columns={'snippet-1': 'snippet1', 'snippet-2': 'snippet2'})

        return df_to_clean

    test_labeled_df = clean_df(test_labeled_df)
    val_labeled_df = clean_df(val_labeled_df)
    train_df = clean_df(train_df)
    test_labeled_df.to_csv(f'{output_prefix}test.tsv', sep='\t')
    val_labeled_df.to_csv(f'{output_prefix}val.tsv', sep='\t')
    train_df.to_csv(f'{output_prefix}train.tsv', sep='\t')


def merge_into_pairs(df):
    # Combine rows into pairs
    df1 = df.iloc[::2].reset_index(drop=True)  # odd-indexed rows
    df2 = df.iloc[1::2].reset_index(drop=True)  # even-indexed rows

    # rename the columns
    df1.columns = [f"{col}1" for col in df1.columns]
    df2.columns = [f"{col}2" for col in df2.columns]

    # concatenate side by side
    merged_df = pd.concat([df1, df2], axis=1)
    return merged_df


def get_overall_similarity(df):
    df = df.dropna(subset=['snippet1', 'snippet2'])
    df['similarity'] = df.apply(
        lambda row: cosine_similarity(row['embedding1'], row['embedding2']), axis=1)
    df['similarity'] = df.apply(check_substring, axis=1)
    return df


# def get_train_val_test_split(df):
#     np.random.seed(42)  # Set the seed
#     perm = np.random.permutation(df.index)
#     perm_df = df.reindex(perm)
#     test = perm_df.iloc[:100]
#     validation = perm_df.iloc[100:200]
#     train = perm_df.iloc[200:]
#     return train, validation, test

def get_train_val_test_split(df):
    # Define the bins
    # This creates a list like [0.0, 0.1, 0.2, ..., 1.0]
    bins = [i / 10 for i in range(11)]

    # Cut the 'similarity' column into bins
    df['similarity_range'] = pd.cut(
        df['similarity'], bins=bins, include_lowest=True)

    # Count the number of rows in each bin
    counts = df['similarity_range'].value_counts().sort_index()
    print(counts)

    df['similarity_range'] = pd.cut(
        df['similarity'], bins=bins, include_lowest=True)
    np.random.seed(42)
    test_indices = df.groupby('similarity_range').apply(
        lambda x: x.sample(n=min(len(x), 10))).index.get_level_values(1)
    test_df = df.loc[test_indices]
    # frac=1 means return all rows (in random order).
    test_df = test_df.sample(frac=1)
    test_df = test_df.reset_index(drop=True)
    merged_df = df.drop(test_indices)
    validation_indices = merged_df.groupby('similarity_range').apply(
        lambda x: x.sample(n=min(len(x), 10))).index.get_level_values(1)
    validation_df = merged_df.loc[validation_indices]
    # frac=1 means return all rows (in random order).
    validation_df = validation_df.sample(frac=1)
    validation_df = validation_df.reset_index(drop=True)
    train_df = merged_df.drop(validation_indices)

    # Print the number of examples in each dataset
    print(f"Number of examples in training dataset: {len(train_df)}")
    print(f"Number of examples in validation dataset: {len(validation_df)}")
    print(f"Number of examples in test dataset: {len(test_df)}")

    return train_df, validation_df, test_df


def split_into_list(df, columns):
    if type(columns) == list:
        for column in columns:
            df[column] = df[column].str.split('|')
            replace_nan_with(df, [column], [])
            df[column] = df[column].apply(
                lambda x: [item.lower() for item in x])
    else:
        column = columns
        df[column] = df[column].str.split('|')
        replace_nan_with(df, [column], [])
        df[column] = df[column].apply(
            lambda x: [item.lower() for item in x])
    return df


def callback(score, epoch, steps):
    print(f"Score at epoch {epoch}, step {steps}: {score}")


def collect_column_values(df, column):
    df_copy = df.copy()
    df_copy[column] = df_copy[column].str.split('|')
    replace_nan_with(df_copy, [column], [])
    classification_labels = list(set().union(*df_copy[column]))
    return classification_labels

# def print_column_values_count(df, column):
#     if df[column].iloc[0]
#     all_keywords = df[column].str.split('|').explode().tolist()
#     word_count = Counter(all_keywords)

#     # Print the word frequencies
#     for word, count in word_count.items():
#         print(f'{word}: {count}')

#     if type(columns) == list:
#         for column in columns:
#             df[column] = df[column].str.split('|')


def create_replace_no_tags_embeddings(df, replace_no_tags_embeddings):
    def remove_hashtags(text):
        words = text.split()
        words = [word for word in words if not word.startswith('#')]
        return ' '.join(words)
    df['replace_no_tags'] = df['replace'].apply(remove_hashtags)
    df['replace_no_tags'] = df['replace_no_tags'].str.replace(r'\s+', ' ')
    df['embedding'] = df['replace_no_tags'].map(replace_no_tags_embeddings)
    return df
