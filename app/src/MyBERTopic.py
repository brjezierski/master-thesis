from collections import Counter
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import umap
import hdbscan
import time

from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired

from MySentenceTransformer import MySentenceTransformer

class MyBERTopic(BERTopic):

    def __init__(self, params, df_training, df_inference, topic_model_name, embedding_model):
        # components
        umap_model = umap.UMAP(
            n_neighbors=params["n_neighbors"],
            n_components=params["n_components"],
            min_dist=0.0, 
            metric='cosine', 
            random_state=42
        )
        hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=params["min_cluster_size"], 
            metric='euclidean', 
            cluster_selection_method='eom', 
            prediction_data=True
        )
        vectorizer_model = CountVectorizer(
            stop_words="english", min_df=2, ngram_range=(1, 2)
        )
        keybert_model = KeyBERTInspired()
        representation_model = {
            "KeyBERT": keybert_model,
        }
        
        super().__init__(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            representation_model=representation_model,
            top_n_words=params["top_n_topic_words"],
            verbose=True
        )

        # Additional initialization code specific to MyBERTopic
        self.df_training = df_training
        self.df_inference = df_inference
        self.topic_model_name = topic_model_name
        self.params = params

    @classmethod
    def create_and_train(cls, params: Dict[str, Any], embedding_model: MySentenceTransformer,
                         topic_model_name: str, df_training: pd.DataFrame, df_inference: pd.DataFrame, 
                         reduce_outliers: bool, is_topic_model_trained: bool,
                         default_topic_model: str, model_dir: str):
        '''
        Create a new topic model or load an existing one and train it

        Args:
        - params: dict - Parameters for the model.
        - embedding_model: MySentenceTransformer - The embedding model to use.
        - topic_model_name: str - Name of the topic model.
        - df_training: pd.DataFrame - The training data.
        - df_inference: pd.DataFrame - The inference data.
        - reduce_outliers: bool - Whether to reduce outliers.
        - is_topic_model_trained: bool - Whether the topic model is already trained.
        - default_topic_model: str - Default topic model name.
        - model_dir: str - Directory of the model.

        Returns:
        - MyBERTopic: The created or loaded and trained topic model.
        '''
        if not topic_model_name or topic_model_name.strip() == default_topic_model:  # if no model selected
            topic_model = cls(params, df_training, df_inference, topic_model_name, embedding_model)
        else:
            topic_model = cls.load(model_dir)
            topic_model.df_training = df_training
            topic_model.df_inference = df_inference
            topic_model.topic_model_name = topic_model_name

        topic_model.generate_topics(is_topic_model_trained)

        if reduce_outliers:
            topic_model.reduce_outliers_and_update()
        topic_model.remove_topic_ids_from_topic_labels()
        topic_model.update_topics()
        
        return topic_model


    def generate_topics(self, is_topic_model_trained: bool, verbose: bool = True) -> None:
        '''
        Generate topics using the topic model. 
        If is_topic_model_trained, fit clustering and dimensionality reduction on labels not selected for inference and 
        Otherwise, predict using the topic model.

        Arguments:
        - is_topic_model_trained: bool - Whether the topic model should be trained.
        - verbose: bool - Whether to print verbose output.
        '''

        # get a map of all classes
        classes = self.df_training["top_classification"].to_list()
        labels_to_add = set(classes)
        label_to_int_mapping = {}
        counter = 1
        for string in labels_to_add:
            label_to_int_mapping[string] = counter
            counter += 1
        labels = [label_to_int_mapping[classification]
                if classification else -1 for classification in classes]

        if verbose:
            print("Selected topic model:", self.topic_model_name)
        if is_topic_model_trained:
            input_snippets = self.df_training["embedding_input"].to_list()
            input_embeddings = np.stack(
                    self.df_training["embedding"].to_numpy())

            tic = time.time()
            if verbose:
                print("Fitting a topic model...")
            self.fit(input_snippets,
                                embeddings=input_embeddings, y=labels)
            if verbose:
                print("Fitting a topic model:", time.time()-tic)

        if verbose:
            print("Predicting with a topic model...")
        tic = time.time()
        self.topics_, self.probabilities_ = self.transform(
                self.get_formatted_documents(), embeddings=self.get_embeddings())
        if verbose:
            print("Predicting with a topic model:", time.time()-tic)
        # else:
        #     self.topics_, self.probabilities_ = self.fit_transform(
        #         self.get_formatted_documents(), embeddings=self.get_embeddings())
        

    def get_documents_from_topic(self, topic_id: int, search_term: str = None) -> List[Dict[str, int]]:
        '''
        Get all documents from a specific topic filtering by a search term if provided.

        Arguments:
        - topic_id: int - The topic id.
        - search_term: str - The search term to filter the documents.

        Returns:
        - List[Dict[str, int]]: The documents from the topic with their ids.
        '''
        doc_df = self.get_doc_df()
        documents = doc_df[doc_df['Topic'] == topic_id]
        if search_term:
            documents = documents[documents['Document'].str.contains(
                search_term, case=False)]
        return [{"label": row['Document'], "value": row['ID']} for iter, row in documents.iterrows()]


    def change_topic(self, src_topic_id: int, dest_topic_id: int) -> None:
        '''
        Change the topic id of a document from src_topic_id to dest_topic_id.

        Arguments:
        - src_topic_id: int - The source topic id.
        - dest_topic_id: int - The destination topic id.
        '''
        for i, topic in enumerate(self.topics_):
            if topic == src_topic_id:
                self.topics_[i] = dest_topic_id


    def remove_topic_from_topics(self, src_topic_id: int) -> None:
        '''
        Remove a topic with the src_topic_id from the topics.

        Arguments:
        - src_topic_id: int - The source topic id.
        '''
        self.change_topic(src_topic_id, -1)
        for i, topic in enumerate(self.topics_):
            if topic > src_topic_id:
                self.topics_[i] = self.topics_[i] - 1


    def remove_topic_from_dict(self, topic_id: int, topic_dict: Dict[int, Any]) -> Dict[int, Any]:
        '''
        Remove a topic with the src_topic_id from the topics in such a way that the topics with higher ids are shifted down by 1.

        Arguments:
        - topic_id: int - The source topic id.
        - topic_dict: Dict[int, Any] - The dictionary of topics.

        Returns:
        - Dict[int, Any]: The dictionary of topics with the topic with the src_topic_id removed.
        '''
        topic_dict_copy = {}
        for key in topic_dict:
            if int(key) > int(topic_id):
                topic_dict_copy[key-1] = topic_dict[key]
            else:
                topic_dict_copy[key] = topic_dict[key]
        return topic_dict_copy


    def remove_topic_from_custom_labels(self, topic_id: int) -> None:
        '''
        Remove a topic with the src_topic_id from the custom labels.

        Arguments:
        - topic_id: int - The source topic id.
        '''
        custom_labels_copy = []
        for topic, topic_custom_label in zip(self.topic_labels_, self.custom_labels_):
            if topic != topic_id:
                custom_labels_copy.append(topic_custom_label)
        self.custom_labels_ = custom_labels_copy


    def remove_topic_from_c_tf_idf(self, topic_id: int) -> None:
        '''
        Remove a topic with the src_topic_id from the c_tf_idf.

        Arguments:
        - topic_id: int - The source topic id.
        '''
        indices = list([topic_id])
        mask = np.ones(self.c_tf_idf_.shape[0], dtype=bool)
        mask[indices] = False
        self.c_tf_idf_ = self.c_tf_idf_[mask]


    def recalculate_topic_sizes(self) -> None:
        '''
        Recalculate the topic sizes.
        '''
        self.topic_sizes_ = dict(Counter(self.topics_))


    def remove_topic(self, topic_to_remove):
        '''
        Remove a topic from the topic model by removing it from all the components of the model, 
        such as topics, topic_labels, topic_representations, topic_aspects, representative_docs, topic_sizes, custom_labels, topic_embeddings, and c_tf_idf.

        Args:
        - topic_to_remove: int - The topic to remove
        '''
        print('self.custom_labels_', self.custom_labels_)

        self.remove_topic_from_topics(topic_to_remove)
        # if self.custom_labels_:
        #     self.remove_topic_from_custom_labels(
        #         topic_to_remove)
        self.topic_labels_ = self.remove_topic_from_dict(
            topic_to_remove, self.topic_labels_)

        # change when removing a cluster: topic_representations_, topic_aspects_, representative_docs_, topic_sizes_, topic_labels_, custom_labels_
        self.topic_representations_ = self.remove_topic_from_dict(
            topic_to_remove, self.topic_representations_)
        for topic_aspect in self.topic_aspects_.keys():
            self.topic_aspects_[topic_aspect] = self.remove_topic_from_dict(
                topic_to_remove, self.topic_aspects_[topic_aspect])
        self.representative_docs_ = self.remove_topic_from_dict(
            topic_to_remove, self.representative_docs_)
        if -1 not in self.topic_sizes_:
            self.topic_sizes_[-1] = 0
        self.topic_sizes_[-1] += self.topic_sizes_[topic_to_remove]
        self.topic_sizes_ = self.remove_topic_from_dict(
            topic_to_remove, self.topic_sizes_)
        
        # get index of topic to remove by retrieving the index of the row of the topic_model.get_topic_info() df which contains the column topic_to_remove for column Topic
        # topic_to_remove_index = self.get_topic_info().index[self.get_topic_info()['Topic'] == topic_to_remove].tolist()[0]

        self.topic_embeddings_ = np.delete(self.topic_embeddings_, topic_to_remove, 0)
        self.remove_topic_from_c_tf_idf(topic_to_remove)

        self.recalculate_topic_sizes()


    def add_topic_to_topic_embeddings(self) -> None:
        '''
        Add a new row to the topic embeddings.
        '''
        last_row = self.topic_embeddings_[-1]
        self.topic_embeddings_ = np.vstack(
            (self.topic_embeddings_, last_row))


    def add_topic_to_c_tf_idf(self) -> None:
        '''
        Add a new row to the c_tf_idf.
        '''
        self.c_tf_idf_ = np.vstack([self.c_tf_idf_, self.c_tf_idf_[-1]])


    def add_topic(self) -> str:
        '''
        Add a new topic to the topic model by adding it to all the components of the model,
        such as topics, topic_labels, topic_representations, topic_aspects, representative_docs, topic_sizes, custom_labels, topic_embeddings, and c_tf_idf.

        Returns:
        - str: The new topic label
        '''

        new_topic = max(self.topic_labels_.keys()) + 1
        new_topic_label = f"topic-{new_topic}"
        self.topic_labels_[new_topic] = new_topic_label
        self.topic_representations_[new_topic] = [("", 0.0) for _ in range(
            len(self.topic_representations_[new_topic-1]))]
        for topic_aspect in self.topic_aspects_.keys():
            self.topic_aspects_[topic_aspect][new_topic] = [("", 0.0) for _ in range(
                len(self.topic_aspects_[topic_aspect][new_topic-1]))]
        self.representative_docs_[new_topic] = []
        self.topic_sizes_[new_topic] = 0

        self.add_topic_to_topic_embeddings()
        if self.custom_labels_:
            self.custom_labels_.append(new_topic_label)
        self.topic_sizes_[new_topic] = 0

        mappings = self.topic_mapper_.get_mappings()
        new_topic_ids = {topic: max(mappings.values()) + index + 1 for index, topic in enumerate([new_topic])}
        self.topic_mapper_.add_new_topics(new_topic_ids)

        self.update_topics()
        return new_topic_label


    def get_documents(self) -> List[str]:
        '''
        Returns a list of documents used for inference

        Returns:
        - List[str]: The documents
        '''
        return self.df_inference["snippet"].to_list()
    

    def get_formatted_documents(self) -> List[str]:
        '''
        Returns a list of documents used for inference in the format which was chosen for training of sentence embeddings

        Returns:
        - List[str]: The formatted documents
        '''
        return self.df_inference["embedding_input"].to_list()

    def get_embeddings(self) -> np.ndarray:
        '''
        Returns a list of embeddings used for inference

        Returns:
        - np.ndarray: The embeddings
        '''
        return np.stack(self.df_inference["embedding"].to_numpy())


    def get_doc_df(self) -> pd.DataFrame:
        '''
        Return a document dataframe in the format rquired by BERTopic.
        The dataframe has columns: Document, Topic, Probability, Image, ID

        Returns:
        - pd.DataFrame: The document dataframe
        '''
        doc_df = pd.DataFrame({"Document": self.get_documents(), 
                            "Topic": self.topics_,
                            "Probability": self.probabilities_
                            })
        doc_df['Image'] = None
        doc_df['ID'] = doc_df.index
        return doc_df


    def update_topics(self) -> None:
        '''
        Updates the topics, their probabilities, sizes, and representative docs.
        Useful after adding or removing a topic.
        '''
        self._outliers = 1 if -1 in set(self.topics_) else 0
        document_df = self.get_doc_df()

        old_topic_labels = self.topic_labels_ # to prevent _extract_topics from changing the topic labels
        self._extract_topics(document_df, embeddings=self.get_embeddings())
        self._update_topic_size(document_df)
        self._save_representative_docs(document_df)
        self.probabilities_ = self._map_probabilities(self.probabilities_)
        self.topic_labels_ = old_topic_labels


    def merge_topics(self, topics_to_merge: List[int]) -> None:
        '''
        Merge topics by adding the documents of the topics to merge to the first topic in the list of topics to merge and removing the other topics.

        Arguments:
        - topics_to_merge: List[int] - The list of topics to merge.
        '''
        return self.merge_topics(
            self.get_formatted_documents(), topics_to_merge)


    def remove_topic_ids_from_topic_labels(self) -> None:
        '''
        Remove redundant topic id from the topic labels which are prepended by BERTopic
        '''
        if self.topic_labels_ and self.topic_labels_[0][0].isdigit():
            self.topic_labels_ = {key: value.split('_', 1)[-1] for key, value in self.topic_labels_.items()}


    def reduce_dim_and_visualize_documents(self, reuse_reduced_embeddings: bool = True, verbose: bool = True) -> None:
        '''
        Reduce the dimensionality of the embeddings to 2D using UMAP and visualize the documents in 2D.

        Arguments:
        - reuse_reduced_embeddings: bool - Whether to reuse the reduced embeddings.
        - verbose: bool - Whether to print verbose output.
        '''
        tic = time.time()
        if verbose:
            print("Dim reduction to 2D...")
        if reuse_reduced_embeddings and hasattr(MyBERTopic, 'reduced_embeddings') and self.reduced_embeddings is not None:
            self.reduced_embeddings = self.reduced_embeddings
        else:
            self.reduced_embeddings = umap.UMAP(
                    n_neighbors=15, 
                    n_components=2, 
                    min_dist=0.0, 
                    metric='cosine',
                    random_state=42
                ).fit_transform(self.get_embeddings(), y=self.topics_)
        if verbose:
            print("Dim reduction to 2D:", time.time()-tic)
        tic = time.time()
        fig = self.visualize_documents(
                self.get_documents(), 
                reduced_embeddings=self.reduced_embeddings,
                custom_labels=True,
                hide_annotations=True
            )
        return fig
    
    def reduce_outliers_and_update(self) -> None:
        '''
        Reduce outliers and update the topics.
        '''
        old_topic_labels = self.topic_labels_
        if self.topics_.count(-1) > 0:
            self.topics_ = self.reduce_outliers(self.get_formatted_documents(), self.topics_)
        self.update_topics()
        self.topic_labels_ = old_topic_labels
 