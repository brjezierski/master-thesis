from sentence_transformers import losses
import model_freeze as freeze
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


def collect_classification_labels(df, verbose=False):
    if type(df['classification'].iloc[0]) == str:
        df['classification'] = df['classification'].str.split('|')
        replace_nan_with(df, ['classification'], [])
    classification_labels = list(set().union(*df['classification']))
    if verbose:
        print(classification_labels)
    return df


# Function to calculate cosine similarity between two embeddings
def calculate_cosine_similarity(embedding1, embedding2):
    if type(embedding1) == list:
        embedding1 = np.array(embedding1)
    if type(embedding2) == list:
        embedding2 = np.array(embedding2)
    return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]


def filter_top_values(row, selected_labels):
    if type(row['classification']) == str:
        classifications = row['classification'].split('|')
    else:
        classifications = row['classification']
    if len(classifications) == 1:
        return classifications[0].replace('\^', '')
    else:
        non_selected_labels = [
            label for label in classifications if label not in selected_labels]
        if non_selected_labels:
            return non_selected_labels[0].replace('\^', '')
        else:
            return classifications[0].replace('\^', '')


def get_joint_classification(df):
    # not currently used
    for index, row in df.iterrows():
        classification = row['classification']
        if len(classification) == 0:
            df.at[index, 'joint_classification'] = ""
            continue
        if len(classification) <= 1:
            df.at[index, 'joint_classification'] = classification[-1]
            continue
        df.at[index, 'joint_classification'] = "-".join(row['classification'])
    return df


def get_classification_counts(df):
    classification_counts = {}
    for classification in sorted(df['classification'].str.split('|').explode()):
        normalized_classification = classification.replace('\^', '')
        if normalized_classification not in classification_counts:
            classification_counts[normalized_classification] = 1
        else:
            classification_counts[normalized_classification] += 1
    return classification_counts


def get_relevant_classifications(df, params, verbose=False):

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


def filter_relevant_classifications(df, all_classifications, relevant_classifications):
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


def get_train_dev_test_split(df, val_dev_size=100):

    def get_paired_dataset(df, size=100):
        # Create an empty DataFrame to store the sampled rows
        new_df = pd.DataFrame(columns=df.columns)

        for _ in range(int(size / 2)):
            filtered_df = df[df['top_classification'].map(
                df['top_classification'].value_counts()) > 1]
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
    print('dev_df', len(dev_df))
    print('train_df', len(train_df))
    test_df, train_df = get_paired_dataset(train_df, size=val_dev_size)

    return train_df, dev_df, test_df


def triplets_from_labeled_dataset(input_examples):
    # Create triplets for a [(label, sentence), (label, sentence)...] dataset
    # by using each example as an anchor and selecting randomly a
    # positive instance with the same label and a negative instance with a different label
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


def get_dataset(df, params, label_map, val_dev_size=100):
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

    random.seed(42)

    dev_triplets = triplets_from_labeled_dataset(datasets[1])
    test_triplets = triplets_from_labeled_dataset(datasets[2])
    return datasets[0], dev_triplets, test_triplets


def prepare_for_training(classified_df, classifications, params, sbert_model):
    num_epochs = params["epochs"]
    unfreeze_layers = params["unfreeze_layers"]
    inference_labels = params["inference_labels"] if "inference_labels" in params else []
    occurence_cutoff = params["occurrence_cutoff"]
    snippet_column_name = params["snippet_column_name"]

    if unfreeze_layers:
        freeze.freeze_except_last_layers(sbert_model, unfreeze_layers)
    else:
        freeze.freeze_all_layers(sbert_model)

    train_set, dev_set, test_set = get_dataset(
        classified_df, params, classifications, val_dev_size=params["val_dev_size"])

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

    dev_evaluator = TripletEvaluator.from_input_examples(
        dev_set, name='Glanos')

    loss = losses.BatchAllTripletLoss(model=sbert_model)

    test_evaluator = TripletEvaluator.from_input_examples(
        test_set, name='Glanos')
    train_objective = [(train_dataloader, loss)]

    return train_objective, dev_evaluator, test_evaluator


def train(classified_df, classifications, params, sbert_model):
    """
    This script trains sentence transformers with a batch all triplet loss function.
    """
    if type(classified_df) == list and len(classified_df) == len(classifications):
        print('Training with multiple objectives')
        steps_per_epoch = min([len(df)
                              for df in classified_df]) / params["batch_size"]
        train_objectives = []
        for sub_classified_df, sub_classifications in zip(classified_df, classifications):
            train_objective, dev_evaluator, test_evaluator = prepare_for_training(
                sub_classified_df, sub_classifications, params, sbert_model)
            train_objectives.extend(train_objective)
    else:
        print('Training with one objective')
        steps_per_epoch = len(classified_df) / params["batch_size"]
        train_objectives, dev_evaluator, test_evaluator = prepare_for_training(
            classified_df, classifications, params, sbert_model)

    print("Performance before fine-tuning:")
    dev_evaluator(sbert_model)

    model_fit = sbert_model.fit(
        train_objectives=train_objectives,
        evaluator=dev_evaluator,
        epochs=params["epochs"],
        evaluation_steps=100 if steps_per_epoch <= 200 else steps_per_epoch / 4,
        warmup_steps=params["WARMUP_STEPS"],
        callback=callback
    )
    return model_fit, test_evaluator, sbert_model