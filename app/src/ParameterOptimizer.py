from MyBERTopic import MyBERTopic

from functools import partial

import numpy as np
import pandas as pd
import hdbscan
import umap
from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, space_eval, Trials


class ParameterOptimizer:
    def __init__(self, df_training: pd.DataFrame, df_inference: pd.DataFrame):
        '''
        Arguments:
            df_training: DataFrame used to train the UMAP + HDBSCAN model
            df_inference: DataFrame used to infer the UMAP + HDBSCAN model
            best_params: UMAP + HDBSCAN hyperparameters associated with
                         the best performance after performing bayesian search
                         using 'bayesian_search' method
            best_clusters: HDBSCAN clusters and labels associated with
                           the best performance after performing baysiean
                           search using 'bayesian_search' method
            trials: hyperopt trials saved from bayesian search using
                    'bayesian_search' method
        '''
        self.df_training = df_training
        self.df_inference = df_inference
        self.best_params = None
        self.best_clusters = None
        self.trials = None
        self.bayesian_search_results = []

    def _get_training_embeddings(self):
        '''
        Returns the training embeddings as a numpy array
        '''
        return np.stack(self.df_training["embedding"].to_numpy())

    def _get_inference_embeddings(self):
        '''
        Returns the inference embeddings as a numpy array
        '''
        return np.stack(self.df_inference["embedding"].to_numpy())

    def _generate_clusters(self,
                          n_neighbors,
                          n_components,
                          min_cluster_size,
                          min_samples=None,
                          random_state=42):
        '''
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

        '''
        umap_model = umap.UMAP(n_neighbors=n_neighbors,
                               n_components=n_components,
                               metric='cosine',
                               random_state=random_state).fit(self._get_training_embeddings())
        umap_embeddings = umap_model.transform(self._get_inference_embeddings())

        clusters = (hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                    min_samples=min_samples,
                                    metric='euclidean',
                                    gen_min_span_tree=True,
                                    cluster_selection_method='eom')
                    .fit(umap_embeddings))

        return clusters


    @staticmethod
    def _score_clusters(cluster_labels, prob_threshold=0.05):
        '''
        Returns the label count and cost of a given clustering

        Arguments:
            clusters: HDBSCAN clustering object
            prob_threshold: float, probability threshold to use for deciding
                            what cluster labels are considered low confidence

        Returns:
            label_count: int, number of unique cluster labels, including noise
            cost: float, fraction of data points whose cluster assignment has
                  a probability below cutoff threshold
        '''

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
        '''
        Objective function for hyperopt to minimize

        Arguments:
            space: Dict, contains keys for 'n_neighbors', 'n_components',
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

        '''
        print("Testing hyperparameters:" + self.get_hyperparameter_string(space))
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
                    clusters = self._generate_clusters(n_neighbors=space['n_neighbors'],
                                                    n_components=space['n_components'],
                                                    min_cluster_size=space['min_cluster_size'],
                                                    min_samples=space['min_samples'],
                                                    random_state=space['random_state'])

                    probabilities = clusters.probabilities_
                    cluster_labels = clusters.labels_
                    label_count, cost = self._score_clusters(
                        cluster_labels, prob_threshold=0.05)
                    self.df_inference['cluster_id'] = clusters.labels_
                    cluster_topics = ParameterOptimizer.get_chat_intents_topics(
                        self, self.df_inference)
                elif topic_model == "bert":
                    topic_model = MyBERTopic(params, self.df_training, self.df_inference, params["topic_model_name"], params["embedding_model"])
                    topic_model.generate_topics(params['is_topic_model_trained'])
                    doc_df = topic_model.get_doc_df()

                    cluster_id_to_name_dict = topic_model.get_topic_info().set_index('Topic')[
                        'Representation'].to_dict()
                    cluster_id_to_name_dict = {
                        k: cluster_id_to_name_dict[k] for k in sorted(cluster_id_to_name_dict)
                    }
                    cluster_topics = [data for _, data in cluster_id_to_name_dict.items()]

                    probabilities = doc_df["Probability"].to_numpy()
                    cluster_labels = doc_df["Topic"].to_numpy()
                    label_count, cost = self._score_clusters(
                        cluster_labels, prob_threshold=0.05)
                else:
                    raise ValueError(
                        f"topic_model {topic_model} not supported")

                if metric.startswith('topic_overlap'):
                    loss = ParameterOptimizer.get_overlap_loss(
                        self.df_inference, cluster_topics, label_count)
                elif metric.startswith('topic_diversity'):
                    loss = ParameterOptimizer.get_diversity_loss(cluster_topics, params)

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
        '''
        Perform bayesian search on hyperparameter space using hyperopt

        Arguments:
            space: Dict, contains keys for 'n_neighbors', 'n_components',
                   'min_cluster_size', 'min_samples', and 'random_state' and
                   values that use built-in hyperopt functions to define
                   search spaces for each
            label_lower: int, lower end of range of number of expected clusters
            label_upper: int, upper end of range of number of expected clusters
            max_evals: int, maximum number of parameter combinations to try

        Saves the following to instance variables:
            best_params: Dict, contains keys for 'n_neighbors', 'n_components',
                   'min_cluster_size', 'min_samples', and 'random_state' and
                   values associated with lowest cost scenario tested
            best_clusters: HDBSCAN object associated with lowest cost scenario
                           tested
            trials: hyperopt trials object for search

        '''

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

        best_clusters = self._generate_clusters(n_neighbors=best_params['n_neighbors'],
                                               n_components=best_params['n_components'],
                                               min_cluster_size=best_params['min_cluster_size'],
                                               min_samples=best_params['min_samples'],
                                               random_state=best_params['random_state'])

        self.best_params = best_params
        self.best_clusters = best_clusters
        self.trials = trials
        return best_params, best_clusters, trials

    @staticmethod
    def get_hyperparameter_string(hdbscan_umap_params):
        hyperparam_string = ""
        for key, value in hdbscan_umap_params.items():
            hyperparam_string += "{}: {}, ".format(key.capitalize(), value)
        return hyperparam_string[:-2]

    @staticmethod
    def get_chat_intents_topics(model, output_df):
        cluster_topics = []
        for cluster_name, group_df in output_df.groupby('cluster_id'):
            snippets = group_df['snippet'].tolist()
            topic = model._extract_labels(snippets)
            topic_list = topic.split('_')
            cluster_topics.append(topic_list)
        return cluster_topics


    @staticmethod
    def get_most_common_company(df, cluster_id_to_ids_map, cluster_id):
        # Find the company with the most counts and its count

        id_to_company_map = dict(
            zip(df['id'], df['company']))

        ids = cluster_id_to_ids_map[cluster_id]
        company_counts = {}
        for snippet_id in ids:
            company = id_to_company_map[snippet_id]
            if company in company_counts.keys():
                company_counts[company] += 1
            else:
                company_counts[company] = 1

        max_company_count = 0
        max_company_name = None
        second_max_company_name = None
        cluster_size = 0
        for company, count in company_counts.items():
            cluster_size += count
            if count > max_company_count:
                max_company_count = count
                second_max_company_name = max_company_name
                max_company_name = company

        max_company_occurence = "%.2f" % ((max_company_count / cluster_size) * 100)
        # print(
        #     f"Cluster {cluster_id}: top company {max_company_name} at {max_company_occurence}%")
        return cluster_size, (max_company_count / cluster_size) * 100


    @staticmethod
    def get_overlap_loss(df, topics_list, label_count, verbose=False):
        overlaps = {}
        overlap_count = [0] * 4
        cluster_id_to_ids_map = df.groupby(
            'cluster_id')['id'].apply(list).to_dict()

        for i, lst1 in enumerate(topics_list):
            if i - 1 in cluster_id_to_ids_map.keys():
                for j, lst2 in enumerate(topics_list):
                    if i == j:
                        continue  # Skip self-comparison
                    common_elements = set(lst1) & set(lst2)
                    if len(common_elements) >= 1:
                        overlap_count[len(common_elements) - 1 if len(common_elements)
                                    <= len(overlap_count) else len(overlap_count) - 1] += 1
                        overlaps[(i, j)] = len(common_elements)

        overlap_count = [int(x / 2) for x in overlap_count]
        loss = 0
        for index, count in enumerate(overlap_count):
            if index + 1 == 1:
                # overlap_count = index + 1
                # loss += count * 4^(overlap_count - 2)
                loss += count / 4  # 4^-1
            elif index + 1 == 2:
                loss += count  # 4^0
            elif index + 1 == 3:
                loss += count * 4  # 4^1
            elif index + 1 == 4:
                loss += count * 8  # 2^2
            if verbose:
                if index + 1 == 4:
                    print(f"Overlap count of length {index+1} or more: {count}")
                else:
                    print(f"Overlap count of length {index+1}: {count}")

        loss /= label_count
        return loss


    @staticmethod
    def get_diversity_loss(cluster_topics, params):
        topic_dict = {}
        topic_dict['topics'] = cluster_topics
        metric = TopicDiversity(topk=params['top_n_topic_words'])
        loss = 1 / metric.score(topic_dict)
        return loss


    @staticmethod
    def get_clustering_metrics(df, topics_list, params, verbose=True, input_file='../glanos-data/clustering/big_consulting_export.tsv'):
        cluster_id_to_ids_map = df.groupby(
            'cluster_id')['id'].apply(list).to_dict()

        cluster_sizes = []
        top_company_occurences = []
        for i, lst1 in enumerate(topics_list):
            if i - 1 in cluster_id_to_ids_map.keys():
                cluster_size, top_company_occurence = ParameterOptimizer.get_most_common_company(
                    df, cluster_id_to_ids_map, i - 1, input_file=input_file)
                cluster_sizes.append(cluster_size)
                top_company_occurences.append(top_company_occurence)

        if topics_list:
            overlap_loss = ParameterOptimizer.get_overlap_loss(
                df, topics_list, params['label_count'], verbose=verbose)
            diversity_loss = ParameterOptimizer.get_diversity_loss(topics_list, params)
            if verbose:
                print("Overlap loss: ", ("%.2f" % overlap_loss))
                print("Diversity loss: ", ("%.2f" % diversity_loss))

        avg_cluster_size = np.average(cluster_sizes)
        median_cluster_size = np.median(cluster_sizes)
        std_deviation_cluster_size = np.std(cluster_sizes)
        median_top_company_occurence = np.median(top_company_occurences)
        if verbose:
            print(f"Number of clusters: {params['label_count']}")
            print(f"Avg cluster size:", avg_cluster_size)
            print(f"Median cluster size:", median_cluster_size,
                ", standard deviation:", std_deviation_cluster_size)
            print(f"Median top company occurence:", median_top_company_occurence)
            print(f"Cost (outliers):", ("%.2f" % (params['cost'] * 100)) + "%\n")


class TopicDiversity():
    def __init__(self, topk=10):
        '''
        Initialize metric

        Parameters
        ----------
        topk: top k words on which the topic diversity will be computed
        '''
        self.topk = topk

    def info(self):
        return {
            "name": "Topic diversity"
        }

    def score(self, model_output):
        '''
        Retrieves the score of the metric

        Parameters
        ----------
        model_output : dictionary, output of the model
                       key 'topics' required.

        Returns
        -------
        td : score
        '''
        topics = model_output["topics"]
        if topics is None:
            return 0
        if self.topk > len(topics[0]):
            raise Exception('Words in topics are less than ' + str(self.topk))
        else:
            unique_words = set()
            for topic in topics:
                unique_words = unique_words.union(set(topic[:self.topk]))
            td = len(unique_words) / (self.topk * len(topics))
            return td
