import importlib
from GlanosBERTopic import GlanosBERTopic
import clustering_utils
importlib.reload(clustering_utils)

from clustering_utils import get_overlap_loss, get_diversity_loss, get_chat_intents_topics
from functools import partial

import numpy as np
import pandas as pd
import hdbscan
import umap
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, space_eval, Trials

import sys
sys.path.append('../notebooks')


def get_hyperparameter_string(hdbscan_umap_params):
    hyperparam_string = ""
    for key, value in hdbscan_umap_params.items():
        hyperparam_string += "{}: {}, ".format(key.capitalize(), value)
    return hyperparam_string[:-2]

class ChatIntents:
    def __init__(self, df_training, df_inference):
        self.df_training = df_training
        self.df_inference = df_inference
        self.best_params = None
        self.best_clusters = None
        self.trials = None
        self.bayesian_search_results = []
        """
        ChatIntents initialization

        Arguments:
            df_training
            df_inference
            best_params: UMAP + HDBSCAN hyperparameters associated with
                         the best performance after performing bayesian search
                         using 'bayesian_search' method
            best_clusters: HDBSCAN clusters and labels associated with
                           the best performance after performing baysiean
                           search using 'bayesian_search' method
            trials: hyperopt trials saved from bayesian search using
                    'bayesian_search' method
        """

    def get_training_embeddings(self):
        return np.stack(self.df_training["embedding"].to_numpy())

    def get_inference_embeddings(self):
        return np.stack(self.df_inference["embedding"].to_numpy())

    def generate_clusters(self,
                          n_neighbors,
                          n_components,
                          min_cluster_size,
                          min_samples=None,
                          random_state=None):
        """
        Generate HDBSCAN clusters from UMAP embeddings of instance training
        embeddings

        Arguments:
            n_neighbors: float, UMAP n_neighbors parameter representing the
                         size of local neighborhood (in terms of number of
                         neighboring sample points) used
            n_components: int, UMAP n_components parameter representing
                          dimension of the space to embed into
            min_cluster_size: int, HDBSCAN parameter minimum size of clusters
            min_samples: int, HDBSCAN parameter representing the number of
                         samples in a neighbourhood for a point to be
                         considered a core point
            random_state: int, random seed to use in UMAP process

        Returns:
            clusters: HDBSCAN clustering object storing results of fit
                      to instance training embeddings

        """
        umap_model = umap.UMAP(n_neighbors=n_neighbors,
                               n_components=n_components,
                               metric='cosine',
                               random_state=random_state).fit(self.get_training_embeddings())
        umap_embeddings = umap_model.transform(self.get_inference_embeddings())

        clusters = (hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                    min_samples=min_samples,
                                    metric='euclidean',
                                    gen_min_span_tree=True,
                                    cluster_selection_method='eom')
                    .fit(umap_embeddings))

        return clusters


    @staticmethod
    def score_clusters(probabilities, cluster_labels, prob_threshold=0.05):
        """
        Returns the label count and cost of a given clustering

        Arguments:
            clusters: HDBSCAN clustering object
            prob_threshold: float, probability threshold to use for deciding
                            what cluster labels are considered low confidence

        Returns:
            label_count: int, number of unique cluster labels, including noise
            cost: float, fraction of data points whose cluster assignment has
                  a probability below cutoff threshold
        """

        unique_cluster_labels = np.unique(cluster_labels)
        label_count = 0
        for value in unique_cluster_labels:
            if not pd.isna(value) and value >= 0:
                label_count += 1
        total_num = len(cluster_labels)
        # cost = (np.count_nonzero(probabilities < prob_threshold)
        #         / total_num)
        cost = (cluster_labels == -1).sum() / total_num

        return label_count, cost


    def _objective(self, space, params, label_lower, label_upper, metric='outliers'):
        """
        Objective function for hyperopt to minimize

        Arguments:
            space: dict, contains keys for 'n_neighbors', 'n_components',
                   'min_cluster_size', 'min_samples', 'random_state' and
                   their values to use for evaluation
            params:
            label_lower: int, lower end of range of number of expected clusters
            label_upper: int, upper end of range of number of expected clusters

        Returns:
            loss: cost function result incorporating penalties for falling
                  outside desired range for number of clusters
            label_count: int, number of unique cluster labels, including noise
            status: string, hypoeropt status

        """
        print("Testing hyperparameters:" + get_hyperparameter_string(space))
        try:
            if metric == 'outliers':
                loss = cost
            elif metric.startswith('topic_'):
                params.update(space)
                if 'id' not in self.df_inference.columns:
                    self.df_inference.insert(0, 'id', self.df_inference.index)
                if 'id' not in self.df_training.columns:
                    self.df_training.insert(0, 'id', self.df_training.index)
                topic_model = metric.split('+')[-1]
                if topic_model == "chat_intents":
                    clusters = self.generate_clusters(n_neighbors=space['n_neighbors'],
                                                    n_components=space['n_components'],
                                                    min_cluster_size=space['min_cluster_size'],
                                                    min_samples=space['min_samples'],
                                                    random_state=space['random_state'])

                    probabilities = clusters.probabilities_
                    cluster_labels = clusters.labels_
                    # TODO check or fix
                    label_count, cost = self.score_clusters(
                        probabilities, cluster_labels, prob_threshold=0.05)
                    self.df_inference['cluster_id'] = clusters.labels_
                    cluster_topics = get_chat_intents_topics(
                        self, self.df_inference)
                elif topic_model == "bert":
                    topic_model = GlanosBERTopic(params, self.df_training, self.df_inference, params["topic_model_name"], params["embedding_model"])
                    topic_model.train(params['is_topic_model_training'])
                    doc_df = topic_model.get_doc_df()

                    cluster_id_to_name_dict = topic_model.get_topic_info().set_index('Topic')[
                        'Representation'].to_dict()
                    cluster_id_to_name_dict = {
                        k: cluster_id_to_name_dict[k] for k in sorted(cluster_id_to_name_dict)
                    }
                    cluster_topics = [data for _, data in cluster_id_to_name_dict.items()]

                    probabilities = doc_df["Probability"].to_numpy()
                    cluster_labels = doc_df["Topic"].to_numpy()
                    label_count, cost = self.score_clusters(
                        probabilities, cluster_labels, prob_threshold=0.05)
                else:
                    raise ValueError(
                        f"topic_model {topic_model} not supported")

                if metric.startswith('topic_overlap'):
                    print('self.df_inference', len(self.df_inference), 'cluster_topics', len(cluster_topics), 'label_count', len(label_count))
                    loss = get_overlap_loss(
                        self.df_inference, cluster_topics, label_count)
                elif metric.startswith('topic_diversity'):
                    loss = get_diversity_loss(cluster_topics, params)

            # 15% penalty on the cost function if outside the desired range
            # for the number of clusters
            if 'penalty' in params and params['penalty'] == 'outside_range':
                if (label_count < label_lower) | (label_count > label_upper):
                    penalty = params['outside_range_penalty'] if 'outside_range_penalty' in params else 0.15
                else:
                    penalty = 0
            else:
                penalty = 2 / label_count

            loss = loss + penalty
            print(
                f"Loss: {loss} (penalty: {penalty}), Label_count: {label_count}, Metric: {metric}")
            self.bayesian_search_results.append((space, loss, label_count))
            return {'loss': loss, 'label_count': label_count, 'status': STATUS_OK}
        except ValueError as e:
            loss = float('inf')
            label_count = 0
            self.bayesian_search_results.append((space, loss, label_count))
            print(e)
            return {'loss': loss, 'label_count': label_count, 'status': STATUS_FAIL}


    def bayesian_search(self,
                        space,
                        label_lower,
                        label_upper,
                        params):
        """
        Perform bayesian search on hyperparameter space using hyperopt

        Arguments:
            space: dict, contains keys for 'n_neighbors', 'n_components',
                   'min_cluster_size', 'min_samples', and 'random_state' and
                   values that use built-in hyperopt functions to define
                   search spaces for each
            label_lower: int, lower end of range of number of expected clusters
            label_upper: int, upper end of range of number of expected clusters
            max_evals: int, maximum number of parameter combinations to try

        Saves the following to instance variables:
            best_params: dict, contains keys for 'n_neighbors', 'n_components',
                   'min_cluster_size', 'min_samples', and 'random_state' and
                   values associated with lowest cost scenario tested
            best_clusters: HDBSCAN object associated with lowest cost scenario
                           tested
            trials: hyperopt trials object for search

        """

        trials = Trials()
        fmin_objective = partial(self._objective,
                                 params=params,
                                 label_lower=label_lower,
                                 label_upper=label_upper,
                                 metric=params['bayesian_search_metric'])

        best = fmin(fmin_objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=params['max_evals'],
                    trials=trials)

        best_params = space_eval(space, best)
        print('best:')
        print(best_params)
        print(f"label count: {trials.best_trial['result']['label_count']}")

        best_clusters = self.generate_clusters(n_neighbors=best_params['n_neighbors'],
                                               n_components=best_params['n_components'],
                                               min_cluster_size=best_params['min_cluster_size'],
                                               min_samples=best_params['min_samples'],
                                               random_state=best_params['random_state'])

        self.best_params = best_params
        self.best_clusters = best_clusters
        self.trials = trials
        return best_params, best_clusters, trials
