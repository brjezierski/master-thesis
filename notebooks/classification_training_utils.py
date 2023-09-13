from sentence_transformers import losses
import model_freeze as freeze
import pandas as pd
from sentence_transformers.datasets import SentenceLabelDataset
from sentence_transformers.readers import InputExample
from sentence_transformers.evaluation import TripletEvaluator
from sklearn.metrics.pairwise import cosine_similarity
from utils import load_model, replace_nan_with, load_big_consulting_export, callback, load_ai_news, load_car_news
import random
from torch.utils.data import DataLoader
from datetime import datetime
import os
from collections import defaultdict
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('ggplot')

tqdm.pandas()


def get_big_consulting_df(params):
    replace_data_path = '../glanos-data/datasets/big_consulting_export_replace.tsv'
    original_data_path = '../glanos-data/embeddings/big_consulting_export_replace_with_embeddings.pkl'

    if params["USE_ORIGINAL_DATA"] and params["USE_REPLACE_DATA"] and os.path.exists(replace_data_path):
        big_consulting_df = load_big_consulting_export()
        with open(original_data_path, 'rb') as f:
            big_consulting_df_r = pickle.load(f)
        big_consulting_df = pd.concat(
            [big_consulting_df, big_consulting_df_r], axis=0).reset_index(drop=True)
        big_consulting_df.drop_duplicates(
            subset='snippet', keep='first', inplace=True)
    elif params["USE_ORIGINAL_DATA"]:
        # pd.read_pickle('big_consulting_export-df.pkl')
        big_consulting_df = load_big_consulting_export()
    elif params["USE_REPLACE_DATA"]:
        if os.path.exists(replace_data_path):
            with open(original_data_path, 'rb') as f:
                big_consulting_df = pickle.load(f)
        else:
            big_consulting_df = pd.read_csv(replace_data_path, sep='\t')
            sbert_model = load_model()

            def encode_w_sbert(snippet):
                return sbert_model.encode(snippet)
            big_consulting_df['embedding'] = big_consulting_df.progress_apply(
                lambda row: encode_w_sbert(row['replace']), axis=1)
            with open(original_data_path, 'wb') as f:
                pickle.dump(big_consulting_df, f)
    else:
        print("Not supported")
        exit()

    return big_consulting_df


def get_news_df(params, dataset_name):
    prefix = '../glanos-data/embeddings/'
    replace_data_path = f'{prefix}{dataset_name}_replace.pickle'
    original_data_path = f'{prefix}{dataset_name}_snippet.pickle'

    sbert_model = load_model()

    def encode_w_sbert(snippet):
        return sbert_model.encode(snippet)

    news_df = load_ai_news() if dataset_name == 'ai_news' else load_car_news()
    if params["USE_ORIGINAL_DATA"] and params["USE_REPLACE_DATA"] and os.path.exists(replace_data_path):
        print("Not supported")
        return None
        # with open(original_data_path, 'rb') as f:
        #     news_df_r = pickle.load(f)
        # news_df = pd.concat(
        #     [news_df, news_df_r], axis=0).reset_index(drop=True)
    elif params["USE_ORIGINAL_DATA"]:
        if os.path.exists(original_data_path):
            with open(original_data_path, 'rb') as f:
                original_dict = pickle.load(f)
            news_df['embedding'] = news_df['snippet'].map(original_dict)
        else:
            news_df['embedding'] = news_df.progress_apply(
                lambda row: encode_w_sbert(row['snippet']), axis=1)
    elif params["USE_REPLACE_DATA"]:
        if os.path.exists(replace_data_path):
            with open(replace_data_path, 'rb') as f:
                replace_dict = pickle.load(f)
            news_df['embedding'] = news_df['replace'].map(replace_dict)
        else:
            news_df['embedding'] = news_df.progress_apply(
                lambda row: encode_w_sbert(row['replace']), axis=1)
    else:
        print("Not supported")
        return None

    news_df.dropna(subset=['embedding'], inplace=True)
    news_df['id'] = dataset_name + news_df.index.astype(str)
    return news_df


def collect_classification_labels(df, verbose=True):
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


def get_top_values(df, params, column='classification'):
    sbert_model = load_model()
    # Iterate over the rows of the DataFrame
    for index, row in df.iterrows():
        value = row[column]
        snippet_embedding = row['embedding']

        if len(value) == 0:
            df.at[index, f'top_{column}'] = ""
            continue
        if len(value) <= 1:
            value = value[-1].lower().replace('-', '')
            df.at[index, f'top_{column}'] = "" if "EXCLUDE_ENTITY_OTHER" in params and params["EXCLUDE_ENTITY_OTHER"] and value in [
                "entity", "other"] else value
            continue

        # Calculate SBERT embeddings for each element
        embeddings = sbert_model.encode(value)

        # Find the element with the highest cosine similarity to the snippet embedding
        max_similarity = -1
        max_element = None
        for element, embedding in zip(value, embeddings):
            element = element.lower().replace('-', '')
            if "EXCLUDE_ENTITY_OTHER" in params and params["EXCLUDE_ENTITY_OTHER"] and element in ["entity", "other"]:
                continue
            similarity = calculate_cosine_similarity(
                embedding, snippet_embedding)
            if similarity > max_similarity:
                max_similarity = similarity
                max_element = element

        # Assign the selected element to a new column "max_element"
        df.at[index, f'top_{column}'] = max_element.lower().replace(
            '-', '') if max_element else ""

    return df


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


def get_relevant_classifications(df, params, verbose=True):

    def get_value_count(df, col):
        counts = df[col].explode().value_counts()
        return counts

    if verbose:
        non_empty_count = df['top_classification'].apply(
            lambda x: len(x) > 0).sum()
        # /len(merged_df)
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
        # /len(merged_df)
        print(
            f"# of rows with a non-empty list in {value_col}: {non_empty_count}")

    relevant_classifications = []
    all_classifications = []
    for i, v in top_classification_counts.items():
        if i != "":
            print(i, v)
            all_classifications.append(i)
            if "OCCURENCE_CUTOFF" in params and v < params["OCCURENCE_CUTOFF"]:
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

    df.reset_index(inplace=True)
    if 'id' not in df.columns:
        df['id'] = df.index
    return df, classifications


def get_train_dev_test_split(df, dataset_dir, val_dev_size=100):

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
    test_df, train_df = get_paired_dataset(train_df, size=val_dev_size)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    train_df.to_csv(f'{dataset_dir}train.tsv', sep='\t')
    dev_df.to_csv(f'{dataset_dir}dev.tsv', sep='\t')
    test_df.to_csv(f'{dataset_dir}test.tsv', sep='\t')

    return train_df, dev_df, test_df


def read_train_dev_test_files(df, dataset_dir):
    dev_df = pd.read_csv(f'{dataset_dir}dev.tsv', sep='\t')
    test_df = pd.read_csv(f'{dataset_dir}test.tsv', sep='\t')
    dev_and_test_df = pd.concat([dev_df, test_df], axis=0)
    train_df = df[~df['id'].isin(dev_and_test_df['id'])]
    return train_df, dev_df, test_df


def triplets_from_labeled_dataset(input_examples, output_filename):
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

    with open(output_filename, 'wb') as f:
        pickle.dump(triplets, f)
    return triplets


def get_dataset(df, label_map, create_new_split, dataset_dir, val_dev_size=100):
    guid = 1
    if create_new_split:
        train_df, dev_df, test_df = get_train_dev_test_split(
            df, dataset_dir, val_dev_size)
    else:
        train_df, dev_df, test_df = read_train_dev_test_files(
            df, dataset_dir)
    datasets = []

    for dataset in [train_df, dev_df, test_df]:
        examples = []
        for index, row in dataset.iterrows():
            classification = row['top_classification'].lower().replace(
                '-', ' ')
            if classification in label_map.keys():
                label_id = label_map[classification]
                examples.append(InputExample(guid=guid, texts=[
                                row['snippet']], label=label_id))
                guid += 1
        datasets.append(examples)

    random.seed(42)
    dev_set_path = f'{dataset_dir}dev_triplets.pkl'
    if not os.path.exists(dev_set_path):
        dev_triplets = triplets_from_labeled_dataset(
            datasets[1], dev_set_path)
    else:
        with open(dev_set_path, 'rb') as f:
            dev_triplets = pickle.load(f)

    test_set_path = f'{dataset_dir}test_triplets.pkl'
    if not os.path.exists(test_set_path):
        test_triplets = triplets_from_labeled_dataset(
            datasets[2], test_set_path)
    else:
        with open(test_set_path, 'rb') as f:
            test_triplets = pickle.load(f)

    return datasets[0], dev_triplets, test_triplets


def prepare_for_training(classified_df, classifications, params, dataset_dir, sbert_model):
    """
    This script trains sentence transformers with a batch hard loss function.

    The TREC dataset will be automatically downloaded and put in the datasets/ directory

    Usual triplet loss takes 3 inputs: anchor, positive, negative and optimizes the network such that
    the positive sentence is closer to the anchor than the negative sentence. However, a challenge here is
    to select good triplets. If the negative sentence is selected randomly, the training objective is often
    too easy and the network fails to learn good representations.

    Batch hard triplet loss (https://arxiv.org/abs/1703.07737) creates triplets on the fly. It requires that the
    data is labeled (e.g. labels 1, 2, 3) and we assume that samples with the same label are similar:

    In a batch, it checks for sent1 with label 1 what is the other sentence with label 1 that is the furthest (hard positive)
    which sentence with another label is the closest (hard negative example). It then tries to optimize this, i.e.
    all sentences with the same label should be close and sentences for different labels should be clearly seperated.
    """
    num_epochs = params["EPOCHS"]
    use_replace_data = params["USE_REPLACE_DATA"]
    use_original_data = params["USE_ORIGINAL_DATA"]
    unfreeze_layers = params["UNFREEZE_LAYERS"]
    exclude_entity_other = params["EXCLUDE_ENTITY_OTHER"]
    initialized_model = params["INITIALIZED_MODEL"]
    occurence_cutoff = params["OCCURENCE_CUTOFF"]
    create_new_split = params["CREATE_NEW_SPLIT"]

    if unfreeze_layers:
        freeze.freeze_except_last_layers(sbert_model, unfreeze_layers)

    train_set, dev_set, test_set = get_dataset(
        classified_df, classifications, create_new_split, dataset_dir)

    print(f"e={num_epochs} Using " + ("original and replacement" if use_original_data and use_replace_data else "original" if use_original_data else "replacement") + " data" +
          (", creating a new train-dev-test split" if create_new_split else "") +
          (f", unfreezing {unfreeze_layers} last layers" if unfreeze_layers else "") +
          (", excluding Entity and Other labels" if exclude_entity_other else "") +
          (f", starting with {initialized_model}" if initialized_model else "") +
          (f", only including words that occur at least {occurence_cutoff} times" if occurence_cutoff > 0 else "") +
          ", training data size" + str(len(train_set)) +
          "\n"
          )
    print('Training data size', len(train_set))
    print('Validation data size', len(dev_set))
    print('Test data size', len(test_set))

    # We create a special dataset "SentenceLabelDataset" to wrap out train_set
    # It will yield batches that contain at least two samples with the same label
    train_data_sampler = SentenceLabelDataset(train_set)
    train_dataloader = DataLoader(
        train_data_sampler, batch_size=params["BATCH_SIZE"], drop_last=True)

    dev_evaluator = TripletEvaluator.from_input_examples(
        dev_set, name='Glanos')

    loss = losses.BatchAllTripletLoss(model=sbert_model)

    test_evaluator = TripletEvaluator.from_input_examples(
        test_set, name='Glanos')
    train_objective = [(train_dataloader, loss)]

    return train_objective, dev_evaluator, test_evaluator


def train(classified_df, classifications, params, dataset_dir, sbert_model, save_model=False):
    """
    This script trains sentence transformers with a batch hard loss function.

    The TREC dataset will be automatically downloaded and put in the datasets/ directory

    Usual triplet loss takes 3 inputs: anchor, positive, negative and optimizes the network such that
    the positive sentence is closer to the anchor than the negative sentence. However, a challenge here is
    to select good triplets. If the negative sentence is selected randomly, the training objective is often
    too easy and the network fails to learn good representations.

    Batch hard triplet loss (https://arxiv.org/abs/1703.07737) creates triplets on the fly. It requires that the
    data is labeled (e.g. labels 1, 2, 3) and we assume that samples with the same label are similar:

    In a batch, it checks for sent1 with label 1 what is the other sentence with label 1 that is the furthest (hard positive)
    which sentence with another label is the closest (hard negative example). It then tries to optimize this, i.e.
    all sentences with the same label should be close and sentences for different labels should be clearly seperated.
    """
    if type(classified_df) == list and len(classified_df) == len(classifications):
        print('Training with multiple objectives')
        steps_per_epoch = min([len(df)
                              for df in classified_df]) / params["BATCH_SIZE"]
        train_objectives = []
        for sub_classified_df, sub_classifications in zip(classified_df, classifications):
            train_objective, dev_evaluator, test_evaluator = prepare_for_training(
                sub_classified_df, sub_classifications, params, dataset_dir, sbert_model)
            train_objectives.extend(train_objective)
    else:
        print('Training with one objective')
        steps_per_epoch = len(classified_df) / params["BATCH_SIZE"]
        train_objectives, dev_evaluator, test_evaluator = prepare_for_training(
            classified_df, classifications, params, dataset_dir, sbert_model)
    output_path = ("classification_train_output/sbert-" +
                   datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    print("Performance before fine-tuning:")
    dev_evaluator(sbert_model)

    model_fit = sbert_model.fit(
        train_objectives=train_objectives,
        evaluator=dev_evaluator,
        epochs=params["EPOCHS"],
        evaluation_steps=100 if steps_per_epoch <= 200 else steps_per_epoch / 4,
        warmup_steps=params["WARMUP_STEPS"],
        output_path=output_path if save_model else None,
        callback=callback
    )
    return model_fit, test_evaluator, sbert_model
