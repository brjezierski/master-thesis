from typing import Any, Dict, List, Union
from MySentenceTransformer import MySentenceTransformer
import pandas as pd
from sentence_transformers import losses
from sentence_transformers.evaluation import TripletEvaluator
from MyDataset import MyDataset
from utils import  callback
from tqdm import tqdm
import matplotlib.pyplot as plt

plt.style.use('ggplot')

tqdm.pandas()

class MyTrainer:
    '''
    A class to train a SentenceTransformer model on a list of DataFrames.
    
    Attributes:
    - training_data (List[pd.DataFrame]): A list of DataFrames containing the training data.
    - classifications (Union[Dict[str, int], List[Dict[str, int]]): A dictionary or list of dictionaries containing the classification labels.
    - sentence_embedding_model (MySentenceTransformer): A SentenceTransformer model.
    - num_epochs (int): The number of epochs to train the model.
    - batch_size (int): The batch size for training.
    - unfreeze_layers (int): The number of layers to unfreeze for training. If None, all layers are frozen.
    '''
    def __init__(self, 
                 training_data: List[pd.DataFrame], 
                 classifications: Union[Dict[str, int], List[Dict[str, int]]], 
                 sentence_embedding_model: MySentenceTransformer,
                 num_epochs: int,
                 batch_size: int,
                 unfreeze_layers: int = None,
                 snippet_column_name: str = 'snippet'):
        self.training_data = training_data
        self.classifications = classifications
        self.sentence_embedding_model = sentence_embedding_model
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        if unfreeze_layers:
            sentence_embedding_model.freeze_except_last_layers(unfreeze_layers)
        else:
            sentence_embedding_model.freeze_all_layers()
        self.snippet_column_name = snippet_column_name  

    def fit(self) -> MySentenceTransformer:
        '''
        Trains the MySentenceTransformer model on training_data with a batch all triplet loss function.

        Returns:
        - MySentenceTransformer: The trained MySentenceTransformer model.
        '''

        print(f'Training with {len(self.training_data)} objectives')
        assert type(self.training_data) == list or type(self.training_data) == pd.DataFrame

        steps_per_epoch = min([len(df)
                                for df in self.training_data]) / self.batch_size
        train_objectives = []
        for sub_classified_df, sub_classifications in zip(self.training_data, self.classifications):
                dataset = MyDataset(sub_classified_df, self.snippet_column_name, sub_classifications)
                train_dataloader, dev_evaluator, test_evaluator = dataset.prepare_for_training(self.sentence_embedding_model)
                loss = losses.BatchAllTripletLoss(model=self.sentence_embedding_model)
                train_objective = [(train_dataloader, loss)]
                train_objectives.extend(train_objective)

        print("Performance before fine-tuning:")
        dev_evaluator(self.sentence_embedding_model)

        self.sentence_embedding_model.fit(
            train_objectives=train_objectives,
            evaluator=dev_evaluator,
            epochs=self.num_epochs,
            evaluation_steps=100 if steps_per_epoch <= 200 else steps_per_epoch / 4,
            callback=callback
        )
        return self.sentence_embedding_model