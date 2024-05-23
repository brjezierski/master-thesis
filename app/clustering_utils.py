from octis import TopicDiversity
import nltk
from nltk.corpus import stopwords
import numpy as np

nltk.download('stopwords')


def remove_stopwords(sentence):
    stopwords_set = set(stopwords.words('english'))
    stopwords_set.add('%')
    words = sentence.split()
    filtered_words = [word for word in words if word.lower(
    ) not in stopwords_set and any(c.isalpha() for c in word)]
    return ' '.join(filtered_words)


def is_supervised_training(df, params):
    return 'classification' in df.columns and len(params['inference_labels']) > 0


def print_number_of_outliers(predictions):
    outlier_count = predictions.count(-1)
    outlier_percentage = int((outlier_count / len(predictions)) * 100)
    print(
        f"Found {outlier_count} outliers constituting {outlier_percentage}% of all data")


def get_chat_intents_topics(model, output_df):
    cluster_topics = []
    for cluster_name, group_df in output_df.groupby('cluster_id'):
        snippets = group_df['snippet'].tolist()
        topic = model._extract_labels(snippets)
        topic_list = topic.split('_')
        cluster_topics.append(topic_list)
    return cluster_topics


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


def get_diversity_loss(cluster_topics, params):
    topic_dict = {}
    topic_dict['topics'] = cluster_topics
    metric = TopicDiversity(topk=params['top_n_topic_words'])
    loss = 1 / metric.score(topic_dict)
    return loss


def get_clustering_metrics(df, topics_list, params, verbose=True, input_file='../glanos-data/clustering/big_consulting_export.tsv'):
    cluster_id_to_ids_map = df.groupby(
        'cluster_id')['id'].apply(list).to_dict()

    cluster_sizes = []
    top_company_occurences = []
    for i, lst1 in enumerate(topics_list):
        if i - 1 in cluster_id_to_ids_map.keys():
            cluster_size, top_company_occurence = get_most_common_company(
                df, cluster_id_to_ids_map, i - 1, input_file=input_file)
            cluster_sizes.append(cluster_size)
            top_company_occurences.append(top_company_occurence)

    if topics_list:
        overlap_loss = get_overlap_loss(
            df, topics_list, params['label_count'], verbose=verbose)
        diversity_loss = get_diversity_loss(topics_list, params)
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