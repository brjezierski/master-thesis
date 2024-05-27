from typing import Any, Dict, List, Tuple, Union
import numpy as np
import json
import re
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'



def remove_stopwords(sentence):
    stopwords_set = set(stopwords.words('english'))
    stopwords_set.add('%')
    words = sentence.split()
    filtered_words = [word for word in words if word.lower(
    ) not in stopwords_set and any(c.isalpha() for c in word)]
    return ' '.join(filtered_words)

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

def collect_classification_labels(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    '''
    Collects and processes unique classification labels from a DataFrame.
    
    Parameters:
    - df: pd.DataFrame - The DataFrame containing classification data.
    - verbose: bool - If True, prints the collected classification labels.

    Returns:
    - pd.DataFrame: The modified DataFrame with processed classification labels.
    '''
    if type(df['classification'].iloc[0]) == str:
        df['classification'] = df['classification'].str.split('|')
        replace_nan_with(df, ['classification'], [])
    classification_labels = list(set().union(*df['classification']))
    if verbose:
        print(classification_labels)
    return df


def calculate_cosine_similarity(embedding1: Union[np.ndarray, List[float]], embedding2: Union[np.ndarray, List[float]]) -> float:
    '''
    Calculates the cosine similarity between two embeddings.
    
    Parameters:
    - embedding1: Union[np.ndarray, List[float]] - The first embedding.
    - embedding2: Union[np.ndarray, List[float]] - The second embedding.

    Returns:
    - float: The cosine similarity between the two embeddings.
    '''
    if type(embedding1) == list:
        embedding1 = np.array(embedding1)
    if type(embedding2) == list:
        embedding2 = np.array(embedding2)
    return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]


def filter_top_values(row: pd.Series, labels_to_ignore: List[str]) -> str:
    '''
    Since a row can contain more than one label, it filters and returns 
    the first classification label from a row which is not in labels_to_ignore.

    Parameters:
    - row: pd.Series - A row from the DataFrame containing snippets with classification labels.
    - labels_to_ignore: List[str] - List of selected labels to filter from.

    Returns:
    - str: Selected label.
    '''
    if type(row['classification']) == str:
        classifications = row['classification'].split('|')
    else:
        classifications = row['classification']
    if len(classifications) == 1:
        return classifications[0].replace('\^', '')
    else:
        non_selected_labels = [
            label for label in classifications if label not in labels_to_ignore]
        if non_selected_labels:
            return non_selected_labels[0].replace('\^', '')
        else:
            return classifications[0].replace('\^', '')


def get_classification_counts(df: pd.DataFrame) -> Dict[str, int]:
    '''
    Counts the occurrences of each classification label in the DataFrame.

    Parameters:
    - df: pd.DataFrame - The DataFrame containing snippets with classification labels.

    Returns:
    - Dict[str, int]: A dictionary with classification labels as keys and their counts as values.
    '''
    classification_counts = {}
    for classification in sorted(df['classification'].str.split('|').explode()):
        normalized_classification = classification.replace('\^', '')
        if normalized_classification not in classification_counts:
            classification_counts[normalized_classification] = 1
        else:
            classification_counts[normalized_classification] += 1
    return classification_counts


def get_relevant_classifications(df: pd.DataFrame, params: Dict[str, Any], verbose: bool = False) -> Tuple[List[str], List[str]]:
    '''
    Identifies relevant classifications based on given parameters.

    Parameters:
    - df: pd.DataFrame - The DataFrame containing snippets with classification labels.
    - params: Dict[str, Any] - Hyperparameters containing occurence_cutoff.
    - verbose: bool - If True, prints detailed classification counts.

    Returns:
    - Tuple[List[str], List[str]]: A tuple containing a list of all classifications and a list of classifications occuring more than the occurence_cutoff.
    '''

    def get_value_count(df, col):
        counts = df[col].explode().value_counts()
        return counts

    if verbose:
        non_empty_count = df['top_classification'].apply(
            lambda x: len(x) > 0).sum()
        print(
            f"Share of rows with a non-empty list in 'classification': {non_empty_count}")
        len_2_count = df['top_classification'].apply(
            lambda x: len(x) == 2).sum()
        print(
            f"Share of rows with a list of len 2 in 'classification': {len_2_count}")

    value_col = 'top_classification'
    top_classification_counts = get_value_count(df, value_col)

    if verbose:
        non_empty_count = df[value_col].apply(lambda x: len(x) > 0).sum()
        print(
            f"# of rows with a non-empty list in {value_col}: {non_empty_count}")

    relevant_classifications = []
    all_classifications = []
    for i, v in top_classification_counts.items():
        if i != "":
            if verbose:
                print(i, v)
            all_classifications.append(i)
            if "occurrence_cutoff" in params and v < params["occurrence_cutoff"]:
                continue
            relevant_classifications.append(i)
    return all_classifications, relevant_classifications


def filter_relevant_classifications(df: pd.DataFrame, all_classifications: List[str], relevant_classifications: List[str]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    '''
    Filters the DataFrame to include only relevant classifications.

    Parameters:
    - df: pd.DataFrame - The DataFrame containing snippets with classification labels.
    - all_classifications: List[str] - List of all classifications.
    - relevant_classifications: List[str] - List of relevant classifications to filter.

    Returns:
    - Tuple[pd.DataFrame, Dict[str, int]]: A tuple containing the filtered DataFrame and a classification to index mapping.
    '''
    df = df[df['top_classification'] != '']
    df = df[df['top_classification'].isin(relevant_classifications)]

    classifications = {}
    i = 0
    for r_c in all_classifications:
        classifications[r_c] = i
        i += 1

    df.reset_index(drop=True, inplace=True)
    if 'id' not in df.columns:
        df['id'] = df.index
    return df, classifications
