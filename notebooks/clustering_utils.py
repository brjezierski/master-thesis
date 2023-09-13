import json
import numpy as np
import pandas as pd
import re
import numpy as np
import pandas as pd
import re
from bertopic import BERTopic
import classification_training_utils
import importlib
importlib.reload(classification_training_utils)

from classification_training_utils import collect_classification_labels, get_top_values
from utils import load_model, create_replace_no_tags_embeddings
import umap
from sentence_transformers import SentenceTransformer
import time
import plotly.express as px
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from octis.evaluation_metrics.diversity_metrics import TopicDiversity

# def get_topics(row):
#     cluster_id = row['cluster_id']
#     return sbert_cluster_topics[cluster_id + 1]


def remove_duplicates(df):
    numeric_pattern = r'(\$)?(\d{1,3}(,\d{3})*|(\d+))(\.\d+)?\s*%?'

    def replace_numeric(match):
        return "NUM"
    df['temp_snippet'] = df['snippet'].apply(lambda x: re.sub(
        numeric_pattern, replace_numeric, x, flags=re.IGNORECASE))
    df = df.drop_duplicates(subset=['temp_snippet'])
    return df


nltk.download('stopwords')


def remove_stopwords(sentence):
    stopwords_set = set(stopwords.words('english'))
    stopwords_set.add('%')
    words = sentence.split()
    filtered_words = [word for word in words if word.lower(
    ) not in stopwords_set and any(c.isalpha() for c in word)]
    return ' '.join(filtered_words)


def get_bert_topics(df, clustering, snippets, params):
    umap_model = umap.UMAP(n_neighbors=params['n_neighbors'],
                           n_components=params['n_components'],
                           min_dist=0.0, metric='cosine',
                           random_state=42)
    vectorizer_model = CountVectorizer(
        stop_words="english", min_df=2, ngram_range=(1, 2))
    topic_model = BERTopic(embedding_model=SentenceTransformer(params['HF_MODEL_NAME']),
                           umap_model=umap_model,
                           hdbscan_model=clustering,
                           vectorizer_model=vectorizer_model,
                           top_n_words=params['TOP_K_TOPICS'],
                           verbose=True
                           )  # , representation_model=representation_model)

    if params['REMOVE_STOPWORDS']:
        snippets = [remove_stopwords(snippet) for snippet in snippets]

    if params['SEMI_SUPERVISED'] and 'classification' in df.columns:
        df = collect_classification_labels(df)
        df = get_top_values(df, params)
        classes = df["top_classification"].to_list()
        labels_to_add = set(classes)
        label_to_int_mapping = {}
        counter = 1
        for string in labels_to_add:
            label_to_int_mapping[string] = counter
            counter += 1

        new_labels = [label_to_int_mapping[classification]
                      if classification else -1 for classification in classes]
        topics, probs = topic_model.fit_transform(snippets, y=new_labels)
    else:
        topics, probs = topic_model.fit_transform(snippets)

    document_info = topic_model.get_document_info(snippets)

    topic_df = document_info.rename(
        columns={'Document': params['SNIPPET_COLUMN_NAME']})
#     .drop(
#         columns=['Representation', 'Representative_Docs', 'Top_n_words', 'Probability', 'Representative_document'])
#     output_df = pd.merge(output_df, topic_df, on='snippet', how='left')
    df = pd.merge(df, topic_df, left_index=True,
                  right_index=True, how='left')
    cluster_id_to_name_dict = document_info.set_index('Topic')[
        'Representation'].to_dict()
    cluster_id_to_name_dict = {
        k: cluster_id_to_name_dict[k] for k in sorted(cluster_id_to_name_dict)}

    cluster_topics = [data for cluster_name,
                      data in cluster_id_to_name_dict.items()]

    # cluster_topics = [data.split(
    #     '_')[1:] for cluster_name, data in cluster_id_to_name_dict.items()]

    return cluster_topics, df, topic_model


def get_chat_intents_topics(model, output_df):
    cluster_topics = []
    for cluster_name, group_df in output_df.groupby('cluster_id'):
        #       data = group_df[['id', 'tooltip', 'x', 'y']].to_dict('records')
        topic = model._extract_labels(group_df['snippet'].tolist())
        topic_list = topic.split('_')
        cluster_topics.append(topic_list)
    return cluster_topics


# Import the tsv consulting file to get all the companies


def get_most_common_company(cluster_id_to_ids_map, cluster_id, input_file='../glanos-data/clustering/big_consulting_export.tsv'):
    # Find the company with the most counts and its count
    big_consulting_export = pd.read_csv(input_file, sep='\t')
    id_to_company_map = dict(
        zip(big_consulting_export['id'], big_consulting_export['company']))

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
    cluster_size = 0
    for company, count in company_counts.items():
        cluster_size += count
        if count > max_company_count:
            max_company_count = count
            max_company_name = company

    max_company_occurence = "%.2f" % ((max_company_count / cluster_size) * 100)
#     print(f"Cluster {cluster_id}: top company {max_company_name} at {max_company_occurence}%")
    return cluster_size, (max_company_count / cluster_size) * 100


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
                    overlap_count[len(common_elements) - 1] += 1
                    overlaps[(i, j)] = len(common_elements)

    overlap_count = [int(x / 2) for x in overlap_count]
    loss = 0
    for index, count in enumerate(overlap_count):
        if index + 1 == 1:
            loss += count / 4
        elif index + 1 == 2:
            loss += count
        elif index + 1 == 3:
            loss += count * 4
        elif index + 1 == 4:
            loss += count * 8
        if verbose:
            print(f"Overlap count of length {index+1}: {count}")

    loss /= label_count
    return loss


def get_diversity_loss(cluster_topics, params):
    topic_dict = {}
    topic_dict['topics'] = cluster_topics
    metric = TopicDiversity(topk=params['TOP_K_TOPICS'])
    loss = 1 / metric.score(topic_dict)
    return loss


def embed_sbert(snippets):
    model_sbert = SentenceTransformer('all-MiniLM-L12-v2')
    embeddings = model_sbert.encode(snippets)
    return embeddings


def reverse_dictionary(dict):
    '''
    http://stackoverflow.com/questions/483666/python-reverse-inverse-a-mapping
    '''
    return {v: k for k, v in dict.items()}


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


def perform_clustering(model, n_neighbors=None, n_components=None, min_cluster_size=None, min_samples=None):
    start_time_clustering_local = time.time()
    if model.best_params is not None:
        clustering = model.generate_clusters(n_neighbors=model.best_params["n_neighbors"],  # st1
                                             n_components=model.best_params["n_components"],
                                             min_cluster_size=model.best_params["min_cluster_size"],
                                             min_samples=model.best_params["min_samples"],
                                             random_state=42,
                                             )
    else:
        clustering = model.generate_clusters(n_neighbors=n_neighbors,
                                             n_components=n_components,
                                             min_cluster_size=min_cluster_size,
                                             min_samples=min_samples,
                                             random_state=42,
                                             )

    print(
        f"Elapsed total time for clustering alg: {time.time() - start_time_clustering_local} seconds")
    return clustering


def save_input_with_params(params, df, output_file_path):
    '''
    Save input files for different embedding models
    '''
    output_data = {}
    output_data["items"] = df.to_dict('records')
    output_data["parameters"] = params

    with open(output_file_path, 'w') as f:
        json.dump(output_data, f)


def get_clustering_metrics(df, topics_list, params, verbose=True, input_file='../glanos-data/clustering/big_consulting_export.tsv'):
    cluster_id_to_ids_map = df.groupby(
        'cluster_id')['id'].apply(list).to_dict()

    cluster_sizes = []
    top_company_occurences = []
    for i, lst1 in enumerate(topics_list):
        if i - 1 in cluster_id_to_ids_map.keys():
            cluster_size, top_company_occurence = get_most_common_company(
                cluster_id_to_ids_map, i - 1, input_file=input_file)
            cluster_sizes.append(cluster_size)
            top_company_occurences.append(top_company_occurence)

    overlap_loss = get_overlap_loss(
        df, topics_list, params['label_count'], verbose=verbose)
    diversity_loss = get_diversity_loss(topics_list, params)

    median_cluster_size = np.median(cluster_sizes)
    std_deviation_cluster_size = np.std(cluster_sizes)
    median_top_company_occurence = np.median(top_company_occurences)
    if verbose:
        print("Overlap loss: ", ("%.2f" % overlap_loss))
        print("Diversity loss: ", ("%.2f" % diversity_loss))
        print(f"Number of clusters: {params['label_count']}")
        print(f"Median cluster size:", median_cluster_size,
              ", standard deviation:", std_deviation_cluster_size)
        print(f"Median top company occurence:", median_top_company_occurence)
        print(f"Cost (outliers):", ("%.2f" % (params['cost'] * 100)) + "%\n")


def plot_best_clusters(model, df, labels, params, filename="", n_neighbors=15, min_dist=0.1):
    """
    Reduce dimensionality of best clusters and plot in 2D using instance
    variable result of running bayesian_search

    Arguments:
        n_neighbors: float, UMAP hyperparameter n_neighbors
        min_dist: float, UMAP hyperparameter min_dist for effective
                  minimum distance between embedded points

    """

    umap_reduce = (umap.UMAP(n_neighbors=n_neighbors,
                             n_components=2,
                             min_dist=min_dist,
                             # metric='cosine',
                             random_state=42)
                   .fit_transform(model.message_embeddings)
                   )

    result = pd.DataFrame(umap_reduce, columns=['x', 'y'])
    if 'cluster_id' in df.columns:
        df = df.drop(columns=['cluster_id'])
    result['cluster_id'] = labels  # model.best_clusters.labels_

    result_df = pd.concat(
        [result.reset_index(drop=True), df.reset_index(drop=True)], axis=1)

    df_wo_outliers = result_df[result_df.cluster_id != -1]
    hover_data = ['snippet']
    if 'tooltip' in df_wo_outliers.columns:
        hover_data.append('tooltip')
    fig = px.scatter(df_wo_outliers,
                     x="x", y="y",
                     hover_data=hover_data,
                     color="cluster_id",
                     width=1200, height=1200
                     )
    fig.show()
    fig.write_image(f"{params['OUTPUT_DIR']}{filename}.svg")
    fig.write_html(f"{params['OUTPUT_DIR']}{filename}.html",
                   include_plotlyjs=True, full_html=True)
    return result_df


def save_embeddings(model_name, input_file_path, output_file_path, snippet_column, params):
    '''
    Save input files for different embedding models
    '''
    print(f"Retrieving model {model_name}")
    if model_name == "":
        model = load_model()
    else:
        model = load_model(model=model_name)

    if not 'replace' in input_file_path:
        with open(input_file_path, 'r') as f:
            data = json.load(f)

        if 'items' in data:
            df = pd.json_normalize(data['items'])
            df = df.dropna()
            snippets = df[snippet_column].tolist()
            model = load_model(model_name)
            embeddings = model.encode(snippets)
            df['embedding'] = embeddings.tolist()
    else:
        df = pd.read_csv(input_file_path, sep='\t')
        if snippet_column == "replace_no_tags":
            print('Remove tags')
            with open(f'../glanos-data/embeddings/big_consulting_2_replace_no_tags.pickle', 'rb') as f:
                replace_no_tags_embeddings = pickle.load(f)
            df = create_replace_no_tags_embeddings(
                df, replace_no_tags_embeddings)
            snippet_column = 'replace_no_tags'
        if params['KEEP_ONLY_BAD_LABELS']:
            bad_labels = ['OTHER|ENTITY', 'OTHER', 'ENTITY', 'ENTITY|OTHER']
            df = df[df['classification'].isin(bad_labels)]
        snippets = df[snippet_column].tolist()
        embeddings = model.encode(snippets)
        df['embedding'] = embeddings.tolist()

    output_data = {}
    output_data["items"] = df.to_dict('records')

    with open(output_file_path, 'w') as f:
        json.dump(output_data, f)
