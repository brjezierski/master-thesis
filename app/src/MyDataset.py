from typing import Any, Dict, List, Tuple
import pandas as pd
from sentence_transformers.datasets import SentenceLabelDataset
from sentence_transformers.readers import InputExample
from sentence_transformers.evaluation import TripletEvaluator
import random
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

plt.style.use('ggplot')

tqdm.pandas()

class MyDataset:
    def __init__(self, df: pd.DataFrame, snippet_column_name: str, label_map: Dict[str, int], val_dev_size: int = 10):
        self.df = df
        self.snippet_column_name = snippet_column_name
        self.label_map = label_map
        self.val_dev_size = val_dev_size

    def _get_train_dev_test_split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        '''
        Splits the DataFrame into training, development, and test sets. 
        Creates the dev and test sets by sampling pairs of rows with the same classification,
        while the training set contains the remaining rows.

        Returns:
        - Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the training, development, and test DataFrames.
        '''

        def get_paired_dataset(df, size=self.val_dev_size):
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

        train_df = self.df.copy()
        dev_df, train_df = get_paired_dataset(train_df, size=self.val_dev_size)
        test_df, train_df = get_paired_dataset(train_df, size=self.val_dev_size)

        return train_df, dev_df, test_df

    @staticmethod
    def _triplets_from_labeled_dataset(input_examples: List[InputExample]) -> List[InputExample]:
        '''
        Creates triplets from a [(label, sentence), (label, sentence)...] dataset 
        for triplet loss training by using each example as an anchor and selecting 
        randomly a positive instance with the same label and a negative instance
        with a different label.

        Parameters:
        - input_examples: List[InputExample] - A list of InputExample objects containing individual labeled snippets.

        Returns:
        - List[InputExample]: A list of triplet InputExample objects.
        '''
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


    def _split_data(self) -> Tuple[List[InputExample], List[InputExample], List[InputExample]]:
        '''
        Prepares the data for training by extracting a top classification label for each snippet,
        converts snippets into InputExample format required for training,
        and creates triplets for the development and test sets.

        Returns:
        - Tuple[List[InputExample], List[InputExample], List[InputExample]]: A tuple containing lists of InputExample objects for train, dev, and test sets.
        '''
        guid = 1
        train_df, dev_df, test_df = self._get_train_dev_test_split()

        datasets = []
        for dataset in [train_df, dev_df, test_df]:
            examples = []
            for index, row in dataset.iterrows():
                classification = row['top_classification']
                if classification in self.label_map.keys():
                    label_id = self.label_map[classification]
                    examples.append(InputExample(guid=guid, texts=[
                                    row[self.snippet_column_name]], label=label_id))
                    guid += 1
                else:
                    print(classification)
            datasets.append(examples)

        dev_triplets = MyDataset._triplets_from_labeled_dataset(datasets[1]) 
        test_triplets = MyDataset._triplets_from_labeled_dataset(datasets[2])
        return datasets[0], dev_triplets, test_triplets

    def prepare_for_training(self) -> Tuple[List[Tuple[DataLoader, Any]], TripletEvaluator, TripletEvaluator]:
        '''
        Prepares the data and model for training.

        Returns:
        - Tuple[DataLoader, TripletEvaluator, TripletEvaluator]: The data loader for training, validation evaluator, and test evaluator.
        '''
        train_set, dev_set, test_set = self._split_data()

        print('Training data size', len(train_set))
        print('Validation data size', len(dev_set))
        print('Test data size', len(test_set))

        # We create a special dataset "SentenceLabelDataset" to wrap out train_set
        # It will yield batches that contain at least two samples with the same label
        train_data_sampler = SentenceLabelDataset(train_set)
        train_dataloader = DataLoader(
            train_data_sampler, batch_size=self.params["batch_size"], drop_last=True)

        dev_evaluator = TripletEvaluator.from_input_examples(dev_set)
        test_evaluator = TripletEvaluator.from_input_examples(test_set)

        return train_dataloader, dev_evaluator, test_evaluator


