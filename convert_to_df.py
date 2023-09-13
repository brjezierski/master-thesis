import json
import pandas as pd
import argparse


def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def create_tsv(json_file1, json_file2, output_tsv):
    data1 = load_json(json_file1)['items']
    data2 = load_json(json_file2)['clusters']

    # Create a dictionary to map 'id' to 'cluster_id' from json_file2
    id_to_cluster = {item['id']: cluster['name']
                     for cluster in data2 for item in cluster['data']}

    # Prepare the data for the TSV
    tsv_data = []
    for item in data1:
        item_id = item['id']
        cluster_id = id_to_cluster.get(item_id, None)
        snippet = item['snippet']
        tsv_data.append({'id': item_id, 'snippet': snippet,
                        'cluster_id': cluster_id})

    # Create a DataFrame and save to TSV
    df = pd.DataFrame(tsv_data)
    df.to_csv(output_tsv, sep='\t', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-snippets', '--snippets', type=str, required=True,
                        help='JSON file with snippets')
    parser.add_argument('-clustering', '--clustering', type=str, required=True,
                        help='JSON file with clustering')
    parser.add_argument('-output_name', '--output_name', type=str, required=True,
                        help='Output file name')
    args = parser.parse_args()

    create_tsv(args.snippets, args.clustering, args.output_name)
