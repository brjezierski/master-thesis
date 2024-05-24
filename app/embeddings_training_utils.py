from typing import Any, Dict, List, Tuple, Union
from sentence_transformers import losses
from GlanosSentenceTransformers import GlanosSentenceTransformer
import pandas as pd
from sentence_transformers.datasets import SentenceLabelDataset
from sentence_transformers.readers import InputExample
from sentence_transformers.evaluation import TripletEvaluator
from sklearn.metrics.pairwise import cosine_similarity
from utils import replace_nan_with, callback
import random
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')

tqdm.pandas()


def collect_classification_labels(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Collects and processes unique classification labels from a DataFrame.
    
    Parameters:
    - df: pd.DataFrame - The DataFrame containing classification data.
    - verbose: bool - If True, prints the collected classification labels.

    Returns:
    - pd.DataFrame: The modified DataFrame with processed classification labels.
    """
    if type(df['classification'].iloc[0]) == str:
        df['classification'] = df['classification'].str.split('|')
        replace_nan_with(df, ['classification'], [])
    classification_labels = list(set().union(*df['classification']))
    if verbose:
        print(classification_labels)
    return df


def calculate_cosine_similarity(embedding1: Union[np.ndarray, List[float]], embedding2: Union[np.ndarray, List[float]]) -> float:
    """
    Calculates the cosine similarity between two embeddings.
    
    Parameters:
    - embedding1: Union[np.ndarray, List[float]] - The first embedding.
    - embedding2: Union[np.ndarray, List[float]] - The second embedding.

    Returns:
    - float: The cosine similarity between the two embeddings.
    """
    if type(embedding1) == list:
        embedding1 = np.array(embedding1)
    if type(embedding2) == list:
        embedding2 = np.array(embedding2)
    return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]


def filter_top_values(row: pd.Series, labels_to_ignore: List[str]) -> str:
    """
    Since a row can contain more than one label, it filters and returns 
    the first classification label from a row which is not in labels_to_ignore.

    Parameters:
    - row: pd.Series - A row from the DataFrame containing snippets with classification labels.
    - labels_to_ignore: List[str] - List of selected labels to filter from.

    Returns:
    - str: Selected label.
    """
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
    """
    Counts the occurrences of each classification label in the DataFrame.

    Parameters:
    - df: pd.DataFrame - The DataFrame containing snippets with classification labels.

    Returns:
    - Dict[str, int]: A dictionary with classification labels as keys and their counts as values.
    """
    classification_counts = {}
    for classification in sorted(df['classification'].str.split('|').explode()):
        normalized_classification = classification.replace('\^', '')
        if normalized_classification not in classification_counts:
            classification_counts[normalized_classification] = 1
        else:
            classification_counts[normalized_classification] += 1
    return classification_counts


def get_relevant_classifications(df: pd.DataFrame, params: Dict[str, Any], verbose: bool = False) -> Tuple[List[str], List[str]]:
    """
    Identifies relevant classifications based on given parameters.

    Parameters:
    - df: pd.DataFrame - The DataFrame containing snippets with classification labels.
    - params: Dict[str, Any] - Hyperparameters containing occurence_cutoff.
    - verbose: bool - If True, prints detailed classification counts.

    Returns:
    - Tuple[List[str], List[str]]: A tuple containing a list of all classifications and a list of classifications occuring more than the occurence_cutoff.
    """

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
    """
    Filters the DataFrame to include only relevant classifications.

    Parameters:
    - df: pd.DataFrame - The DataFrame containing snippets with classification labels.
    - all_classifications: List[str] - List of all classifications.
    - relevant_classifications: List[str] - List of relevant classifications to filter.

    Returns:
    - Tuple[pd.DataFrame, Dict[str, int]]: A tuple containing the filtered DataFrame and a classification to index mapping.
    """
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


def get_train_dev_test_split(df: pd.DataFrame, val_dev_size: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the DataFrame into training, development, and test sets. 
    Creates the dev and test sets by sampling pairs of rows with the same classification,
    while the training set contains the remaining rows.

    Parameters:
    - df: pd.DataFrame - The DataFrame containing snippets with classification labels.
    - val_dev_size: int - The size of the development and test sets.

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the training, development, and test DataFrames.
    """

    def get_paired_dataset(df, size=val_dev_size):
        # Create an empty DataFrame to store the sampled rows
        new_df = pd.DataFrame(columns=df.columns)

        for _ in range(int(size / 2)):
            filtered_df = df[df['top_classification'].map(
                df['top_classification'].value_counts()) > 1]
            print('filtered_df', len(filtered_df))
            if len(filtered_df) < 2:
                raise ValueError(
                    f"Cannot create a dataset with {size} samples, only {len(filtered_df)} samples available.")
            random_top_classification = random.choice(
                filtered_df['top_classification'].unique())
            same_top_classification_rows = filtered_df[filtered_df['top_classification']
                                                       == random_top_classification]

            pair_of_rows = same_top_classification_rows.sample(
                n=2, random_state=42)
            new_df = pd.concat([new_df, pair_of_rows], ignore_index=True)
            df = df.drop(pair_of_rows.index)

        new_df = new_df.sample(frac=1, random_state=42).reset_index(drop=True)
        return new_df, df

    train_df = df.copy()
    dev_df, train_df = get_paired_dataset(train_df, size=val_dev_size)
    test_df, train_df = get_paired_dataset(train_df, size=val_dev_size)

    return train_df, dev_df, test_df


def triplets_from_labeled_dataset(input_examples: List[InputExample]) -> List[InputExample]:
    """
    Creates triplets from a [(label, sentence), (label, sentence)...] dataset 
    for triplet loss training by using each example as an anchor and selecting 
    randomly a positive instance with the same label and a negative instance
    with a different label.

    Parameters:
    - input_examples: List[InputExample] - A list of InputExample objects containing individual labeled snippets.

    Returns:
    - List[InputExample]: A list of triplet InputExample objects.
    """
    # Create triplets for 
    triplets = []
    label2sentence = defaultdict(list)
    for inp_example in input_examples:
        label2sentence[inp_example.label].append(inp_example)

    for inp_example in input_examples:
        anchor = inp_example

        # We need at least 2 examples per label to create a triplet
        if len(label2sentence[inp_example.label]) < 2:
            continue

        positive = None
        while positive is None or positive.guid == anchor.guid:
            positive = random.choice(label2sentence[inp_example.label])
            index_of_label = label2sentence[inp_example.label].index(
                positive)

        negative = None
        while negative is None or negative.label == anchor.label:
            negative = random.choice(input_examples)

        triplets.append(InputExample(
            texts=[anchor.texts[0], positive.texts[0], negative.texts[0]]))

    return triplets

# TODO create Dataset class
def split_data(df: pd.DataFrame, params: Dict[str, Any], label_map: Dict[str, int], val_dev_size: int = 100) -> Tuple[List[InputExample], List[InputExample], List[InputExample]]:
    """
    Prepares the data for training by extracting a top classification label for each snippet,
    converts snippets into InputExample format required for training,
    and creates triplets for the development and test sets.

    Parameters:
    - df: pd.DataFrame - The DataFrame containing snippets with classification labels.
    - params: Dict[str, Any] - Hyperparameters.
    - label_map: Dict[str, int] - Mapping from labels to their corresponding IDs.
    - val_dev_size: int - The size of the development and test sets.

    Returns:
    - Tuple[List[InputExample], List[InputExample], List[InputExample]]: A tuple containing lists of InputExample objects for train, dev, and test sets.
    """
    guid = 1
    train_df, dev_df, test_df = get_train_dev_test_split(df, val_dev_size)

    datasets = []
    for dataset in [train_df, dev_df, test_df]:
        examples = []
        for index, row in dataset.iterrows():
            classification = row['top_classification']
            if classification in label_map.keys():
                label_id = label_map[classification]
                examples.append(InputExample(guid=guid, texts=[
                                row[params['snippet_column_name'] if 'snippet_column_name' in params else 'snippet']], label=label_id))
                guid += 1
            else:
                print(classification)
        datasets.append(examples)

    dev_triplets = triplets_from_labeled_dataset(datasets[1]) # TODO why only dev and test are triplets 
    test_triplets = triplets_from_labeled_dataset(datasets[2])
    return datasets[0], dev_triplets, test_triplets

# TODO create Dataset class
def prepare_for_training(df: pd.DataFrame, classifications: Dict[str, int], params: Dict[str, Any], sbert_model: GlanosSentenceTransformer) -> Tuple[List[Tuple[DataLoader, Any]], TripletEvaluator, TripletEvaluator]:
    """
    Prepares the data and model for training.

    Parameters:
    - df: pd.DataFrame - The DataFrame containing snippets with classification labels.
    - classifications: Dict[str, int] - Mapping from labels to their corresponding IDs.
    - params: Dict[str, Any] - Parameters for training.
    - sbert_model: GlanosSentenceTransformer - The GlanosSentenceTransformer model to be trained.

    Returns:
    - Tuple[List[Tuple[DataLoader, Any]], TripletEvaluator, TripletEvaluator]: The training objective, development evaluator, and test evaluator.
    """
    num_epochs = params["epochs"]
    unfreeze_layers = params["unfreeze_layers"]
    inference_labels = params["inference_labels"] if "inference_labels" in params else []
    occurence_cutoff = params["occurrence_cutoff"]
    snippet_column_name = params["snippet_column_name"]

    if unfreeze_layers:
        sbert_model.freeze_except_last_layers(unfreeze_layers)
    else:
        sbert_model.freeze_all_layers()

    train_set, dev_set, test_set = split_data(
        df, params, classifications, val_dev_size=params["val_dev_size"])

    print(f"e={num_epochs}" +
          (f", embeddings from {snippet_column_name}") +
          (f", unfreezing {unfreeze_layers} last layers" if unfreeze_layers else "") +
          (f", excluding labels {inference_labels}" if inference_labels else "") +
          (f", only including words that occur at least {occurence_cutoff} times" if occurence_cutoff > 0 else "") +
          ", training data size " + str(len(train_set)) +
          "\n"
          )
    print('Training data size', len(train_set))
    print('Validation data size', len(dev_set))
    print('Test data size', len(test_set))

    # We create a special dataset "SentenceLabelDataset" to wrap out train_set
    # It will yield batches that contain at least two samples with the same label
    train_data_sampler = SentenceLabelDataset(train_set)
    train_dataloader = DataLoader(
        train_data_sampler, batch_size=params["batch_size"], drop_last=True)

    dev_evaluator = TripletEvaluator.from_input_examples(dev_set)
    test_evaluator = TripletEvaluator.from_input_examples(test_set)

    loss = losses.BatchAllTripletLoss(model=sbert_model)

    train_objective = [(train_dataloader, loss)]

    return train_objective, dev_evaluator, test_evaluator

# TODO create Trainer class
def train(training_data: List[pd.DataFrame], classifications: Union[Dict[str, int], List[Dict[str, int]]], params: Dict[str, Any], sbert_model: GlanosSentenceTransformer) -> Tuple[TripletEvaluator, GlanosSentenceTransformer]:
    """
    Trains the GlanosSentenceTransformer model on training_data with a batch all triplet loss function.

    Parameters:
    - training_data: Union[pd.DataFrame, List[pd.DataFrame]] - The list of dataframes containing training data.
    - classifications: Union[Dict[str, int], List[Dict[str, int]]] - Mapping from labels to their corresponding IDs.
    - params: Dict[str, Any] - Hyperparameters.
    - sbert_model: GlanosSentenceTransformer - The GlanosSentenceTransformer model to be trained.

    Returns:
    - Tuple[TripletEvaluator, GlanosSentenceTransformer]: The test evaluator, and the trained SentenceTransformer model.
    """
    # TODO why batch all triplet loss?
    print(f'Training with {len(training_data)} objectives')
    assert type(training_data) == list or type(training_data) == pd.DataFrame

    # if type(training_data) == list and len(training_data) == len(classifications):
    steps_per_epoch = min([len(df)
                              for df in training_data]) / params["batch_size"]
    train_objectives = []
    for sub_classified_df, sub_classifications in zip(training_data, classifications):
            train_objective, dev_evaluator, test_evaluator = prepare_for_training(
                sub_classified_df, sub_classifications, params, sbert_model)
            train_objectives.extend(train_objective)
    # else:
    #     print('Training with one objective')
    #     steps_per_epoch = len(training_data) / params["batch_size"]
    #     train_objectives, dev_evaluator, test_evaluator = prepare_for_training(
    #         training_data, classifications, params, sbert_model)

    print("Performance before fine-tuning:")
    dev_evaluator(sbert_model)

    sbert_model.fit(
        train_objectives=train_objectives,
        evaluator=dev_evaluator,
        epochs=params["epochs"],
        evaluation_steps=100 if steps_per_epoch <= 200 else steps_per_epoch / 4,
        callback=callback
    )
    return test_evaluator, sbert_model