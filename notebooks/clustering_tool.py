# %% [markdown]
# # Module 1

# %%
import pickle
import os
import sys

import importlib
import classification_training_utils
importlib.reload(classification_training_utils)
import utils
importlib.reload(utils)

sys.path.append('../chat-intents/chatintents')
import chatintents
importlib.reload(chatintents)

from classification_training_utils import get_big_consulting_df, collect_classification_labels, get_relevant_classifications, train, filter_relevant_classifications, get_top_values, get_news_df
from clustering_utils import remove_stopwords
from utils import create_replace_no_tags_embeddings
import pandas as pd
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech
from bertopic import BERTopic
import numpy as np
from hyperopt import hp
from chatintents import ChatIntents
from collections import OrderedDict

# %%
def save_representative_docs(topic_model, doc_df: pd.DataFrame, nr_repr_docs=3):
    """ Save most representative docs per topic

    Arguments:
        doc_df: Dataframe with documents and their corresponding IDs

    Updates:
        self.representative_docs_: Populate each topic with most representative docs
    """
    documents = doc_df.rename(columns={'prediction': 'Topic', "title": "Document"})
    documents.drop(columns=['probability'], inplace=True)
    documents['Image'] = None
    documents['ID'] = documents.index
    repr_docs, _, _, _= topic_model._extract_representative_docs(topic_model.c_tf_idf_,
                                                           documents,
                                                           topic_model.topic_representations_,
                                                           nr_samples=500,
                                                           nr_repr_docs=nr_repr_docs*3)
    for cluster_id, docs in repr_docs.items():
        repr_docs[cluster_id] = list(OrderedDict.fromkeys(docs))[:(nr_repr_docs if nr_repr_docs < len(docs) else len(docs))]
    topic_model.representative_docs_ = repr_docs

def get_number_of_outliers(predictions):
    outlier_count = predictions.count(-1)
    outlier_percentage = int((outlier_count / len(predictions)) * 100)
    print(f"Found {outlier_count} outliers constituting {outlier_percentage}% of all data")

# %%
params = {
    # TODO add penalty, label_range to interface
    'PENALTY': 'label_count', # ['label_count', 'outside_range']
    'BAYESIAN_SEARCH_METRIC': 'topic_diversity+bert',
    'MAX_EVALS': 10,
    'HF_MODEL_NAME': "brjezierski/sentence-embeddings-classification-ai_car-sbert",
    'TOP_K_TOPICS': 10,
    'SNIPPET_COLUMN_NAME': 'replace_no_tags',
    'REMOVE_STOPWORDS': True,
    'DATASET': 'consulting', #['ai', 'consulting', 'car']
    "OCCURENCE_CUTOFF": 2,
    "READ_OLD_DATA": True,
    "READ_OLD_MODELS": True,
}


# %%
if params['DATASET'] == 'ai':
    replace_no_tags_embeddings_file = '../glanos-data/embeddings/ai_news_replace_no_tags.pickle'
    df = get_news_df(params, 'ai_news')
elif params['DATASET'] == 'car':
    replace_no_tags_embeddings_file = '../glanos-data/embeddings/car_news_replace_no_tags.pickle'
    df = get_news_df(params, 'car_news')
else:
    replace_no_tags_embeddings_file = '../glanos-data/embeddings/big_consulting_2_replace_no_tags.pickle'
    replace_data_path = '../glanos-data/datasets/big_consulting_export_replace.tsv'
    if os.path.exists(replace_data_path):
        df = pd.read_csv(replace_data_path, sep='\t')

with open(replace_no_tags_embeddings_file, 'rb') as f:
    replace_no_tags_embeddings = pickle.load(f)
    if params['SNIPPET_COLUMN_NAME'] == 'replace_no_tags':
        df = create_replace_no_tags_embeddings(df, replace_no_tags_embeddings)
    df['embedding'] = df[params['SNIPPET_COLUMN_NAME']].map(replace_no_tags_embeddings)

    
df = collect_classification_labels(df)
# # TODO: what to do with multiple categories
df = get_top_values(df, {})
all_classifications, relevant_classifications = get_relevant_classifications(df, params, verbose=False)
df, classifications = filter_relevant_classifications(df, all_classifications, relevant_classifications)
classifications = relevant_classifications
with open(f"plotly-temp/df_{params['DATASET']}.pkl", 'wb') as file:
    pickle.dump(df, file)

with open(f"plotly-temp/df_{params['DATASET']}.pkl", 'rb') as file:
    df = pickle.load(file) #[:3000]
df = df.drop(columns=['embedding'])


# %% [markdown]
# # Module 2

# %%
from sentence_transformers import SentenceTransformer
import time

df_good_labels_only, docs_good_labels_only, titles_good_labels_only, embeddings_good_labels_only, df_bad_labels_only, docs_bad_labels_only, titles_bad_labels_only, embeddings_bad_labels_only = [], [], [], [], [], [], [], []
topic_model, predictions, probabilities, reduced_embeddings_supervised = None, [], [], []

def get_data(selected_labels, read_old=True):
    global df_good_labels_only, docs_good_labels_only, titles_good_labels_only, embeddings_good_labels_only, df_bad_labels_only, docs_bad_labels_only, titles_bad_labels_only, embeddings_bad_labels_only
    
    df_good_labels_only = df[~df['top_classification'].isin(selected_labels)]
    docs_good_labels_only = df_good_labels_only["replace_no_tags"].to_list()
    titles_good_labels_only = df_good_labels_only["snippet"].to_list()

    df_bad_labels_only = df[df['top_classification'].isin(selected_labels)] #[:50]
    docs_bad_labels_only = df_bad_labels_only["replace_no_tags"].to_list()
    titles_bad_labels_only = df_bad_labels_only["snippet"].to_list()
    
    if params['REMOVE_STOPWORDS']:
        docs_bad_labels_only = [remove_stopwords(
                snippet) for snippet in docs_bad_labels_only]
        docs_good_labels_only = [remove_stopwords(
                snippet) for snippet in docs_good_labels_only]
    
    if 'embedding' in df.columns:
        print("Retrieving embeddings...")        
        embeddings_good_labels_only = df_good_labels_only["embedding"].to_list()
        embeddings_bad_labels_only = df_bad_labels_only["embedding"].to_list()
    elif read_old:
        with open(f"plotly-temp/embeddings_good_labels_only_{params['DATASET']}.pkl", 'rb') as file:
            embeddings_good_labels_only = pickle.load(file)
        with open(f"plotly-temp/embeddings_bad_labels_only_{params['DATASET']}.pkl", 'rb') as file:
            embeddings_bad_labels_only = pickle.load(file)
    else:
        tic = time.time()
        print("Pre-calculating embeddings...")        
        embedding_model = SentenceTransformer(params['HF_MODEL_NAME'])
        embeddings_good_labels_only = embedding_model.encode(docs_good_labels_only, show_progress_bar=True)
        embeddings_bad_labels_only = embedding_model.encode(docs_bad_labels_only, show_progress_bar=True)
        with open(f"plotly-temp/embeddings_good_labels_only_{params['DATASET']}.pkl", 'wb') as file:
            pickle.dump(embeddings_good_labels_only, file)
        with open(f"plotly-temp/embeddings_bad_labels_only_{params['DATASET']}.pkl", 'wb') as file:
            pickle.dump(embeddings_bad_labels_only, file)
        print("Pre-calculating embeddings:", time.time()-tic)        

    
def run_bayesian_search(min_cluster_size_range, min_samples_range, n_components_range, n_neighbors_range, read_old=True):
    
    hspace = {
        "n_neighbors": hp.choice('n_neighbors', n_neighbors_range),
        "n_components": hp.choice('n_components', n_components_range),
        "min_cluster_size": hp.choice('min_cluster_size', min_cluster_size_range),
        "min_samples": hp.choice('min_samples', min_samples_range), # the higher it is, the more strict will clustering be and have more outliers
        "random_state": 42
    }
    label_lower = 28
    label_upper = 50
    chat_intents_model = ChatIntents(embeddings_good_labels_only, embeddings_bad_labels_only, "model")
    best_params, best_clusters, trials = chat_intents_model.bayesian_search(space=hspace,
                          label_lower=label_lower,
                          label_upper=label_upper,
                          params=params,
                          df=df
                          )
    return best_params

    
def model(selected_labels, hdbscan_umap_params, read_old=True, clusters_to_merge=None, reduce_outliers=False, is_supervised=True, top_n_words=10):
    global topic_model, predictions, probabilities, reduced_embeddings_supervised
    
    if read_old:
        if not topic_model:
            with open(f"plotly-temp/topic_model_{params['DATASET']}.pkl", 'rb') as file:
                topic_model = pickle.load(file)
        if len(predictions) == 0:
            with open(f"plotly-temp/predictions_{params['DATASET']}.pkl", 'rb') as file:
                predictions = pickle.load(file)
            topic_model.topics_ = predictions
        if len(probabilities) == 0:
            with open(f"plotly-temp/probabilities_{params['DATASET']}.pkl", 'rb') as file:
                probabilities = pickle.load(file)
            topic_model.probabilities_ = probabilities
        if len(reduced_embeddings_supervised) == 0:
            with open(f"plotly-temp/reduced_embeddings_supervised_{params['DATASET']}.pkl", 'rb') as file:
                reduced_embeddings_supervised = pickle.load(file)

        if reduce_outliers:
            print("Reducing outliers...")
            predictions = topic_model.reduce_outliers(docs_bad_labels_only, predictions)
            topic_model.topics_ = predictions
        if clusters_to_merge:
            print("Merging...", clusters_to_merge)
            print(type(clusters_to_merge[0]))
            topic_model.merge_topics(docs_bad_labels_only, clusters_to_merge)
        fig = topic_model.visualize_documents(titles_bad_labels_only, predictions, reduced_embeddings=reduced_embeddings_supervised, custom_labels=True, hide_annotations=True)
#             with open(f"plotly-temp/fig_{params['DATASET']}.pkl", 'rb') as file:
#                 fig = pickle.load(file)

    # TODO: Remove duplicates?
    
    # Pre-calculate embeddings
    else:
        all_embeddings = np.vstack((embeddings_good_labels_only, embeddings_bad_labels_only))

        # components
        umap_model = UMAP(n_neighbors=hdbscan_umap_params["n_neighbors"], n_components=hdbscan_umap_params["n_components"], min_dist=0.0, metric='cosine', random_state=42)
        hdbscan_model = HDBSCAN(min_cluster_size=hdbscan_umap_params["min_cluster_size"], metric='euclidean', cluster_selection_method='eom', prediction_data=True)
        vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))
        keybert_model = KeyBERTInspired()
        representation_model = {
            "KeyBERT": keybert_model,
        }

        # training
        classes = df_good_labels_only["top_classification"].to_list()
        labels_to_add = set(classes)
        label_to_int_mapping = {}
        counter = 1
        for string in labels_to_add:
            label_to_int_mapping[string] = counter
            counter += 1

        labels = [label_to_int_mapping[classification]
                          if classification else -1 for classification in classes]

        tic = time.time()
        topic_model = BERTopic(
              embedding_model=SentenceTransformer(params['HF_MODEL_NAME']),
              umap_model=umap_model,
              hdbscan_model=hdbscan_model,
              vectorizer_model=vectorizer_model,
              representation_model=representation_model,
              top_n_words=top_n_words,
              verbose=True
            )

        if is_supervised:
            print("Fitting a model...")
            topic_model.fit(docs_good_labels_only, embeddings=embeddings_good_labels_only, y=labels)   
            with open(f"plotly-temp/topic_model_{params['DATASET']}.pkl", 'wb') as file:
                pickle.dump(topic_model, file)
            print("Fitting a model:", time.time()-tic)

            print("Transforming a model...")
            tic = time.time()
            predictions, probabilities = topic_model.transform(docs_bad_labels_only, embeddings=embeddings_bad_labels_only)
            topic_model.topics_ = predictions
            topic_model.probabilities_ = probabilities
            with open(f"plotly-temp/predictions_{params['DATASET']}.pkl", 'wb') as file:
                pickle.dump(predictions, file)
            with open(f"plotly-temp/probabilities_{params['DATASET']}.pkl", 'wb') as file:
                pickle.dump(probabilities, file)
            print("Transforming:", time.time()-tic)
        else:
            predictions, probabilities = topic_model.fit_transform(docs_bad_labels_only, embeddings=embeddings_bad_labels_only)


        tic = time.time()
#         umap_model_supervised = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine', random_state=42).fit(embeddings_good_labels_only)
        reduced_embeddings_supervised = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings_bad_labels_only, y=predictions) # bad or good?
#         reduced_embeddings_supervised = umap_model_supervised.transform(embeddings_bad_labels_only)
#         reduced_embeddings_supervised = reduced_embeddings_supervised[len(embeddings_good_labels_only):] # TODO check?
        with open(f"plotly-temp/reduced_embeddings_supervised_{params['DATASET']}.pkl", 'wb') as file:
            pickle.dump(reduced_embeddings_supervised, file)
        print("Dim reduction to 2D:", time.time()-tic)
        tic = time.time()
        fig = topic_model.visualize_documents(titles_bad_labels_only, reduced_embeddings=reduced_embeddings_supervised, custom_labels=True, hide_annotations=True)
            
    get_number_of_outliers(predictions)
    print(len(titles_bad_labels_only), len(topic_model.topics_), len(topic_model.probabilities_))
    doc_df = pd.DataFrame({"title": titles_bad_labels_only, "prediction": topic_model.topics_, "probability": topic_model.probabilities_})
    save_representative_docs(topic_model, doc_df)

    return fig, doc_df, topic_model

# %%
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash import ctx, ALL
import dash_bootstrap_components as dbc

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

doc_df = pd.DataFrame({"title": [], "prediction": [], "probability": []})
reduce_outliers = False
topic_model = None
topic_name_dict = {}

# Create a modal dialog for entering the filename
filename_modal = html.Div(id='filename-modal', style={'display': 'none'}, children=[
        html.Div([
            html.Label('Enter the filename:'),
            dcc.Input(id='filename-input', type='text', placeholder='Filename'),
            html.Button('Save', id='filename-confirm'),
            html.Button('Cancel', id='filename-cancel'),
        ], className='modal-content')
    ])

    
app.layout = html.Div([
    dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle(f"Snippets for Cluster")),
                dbc.ModalFooter(dbc.Button("Close", id="snippets-modal-close")),
                dbc.ModalBody(id="snippets-modal-content", children='Snippets listed'),
            ],
            id="snippets-modal",
            is_open=False,
            scrollable=True,
    ),
    html.Div(children='Select bad classification labels'),    
    dcc.Dropdown(
        id='label-dropdown',
        options=[{'label': label, 'value': label} for label in classifications],
        multi=True
    ),
    html.Div(id='selected-labels'),
    dcc.Checklist(
        id='param-search-checklist',
        options=[{'label': 'Run parameter search', 'value': 'parameter_search'}]
    ),
    # TODO determine which to use and how to describe them
    html.Div([
        html.Div([
            html.Label('Min cluster size'),
            dcc.RangeSlider(
                id='min-cluster-size-range',
                min=10,
                max=80,
                step=1,
                marks={i: str(i) for i in range(10, 81, 10)},
                value=[25, 60],
            ),
        ]),
        html.Div([
            html.Label('Min samples'),
            dcc.RangeSlider(
                id='min-samples-range',
                min=1,
                max=10,
                step=1,
                marks={i: str(i) for i in range(1, 11)},
                value=[1, 4],
            ),
        ]),
        
        html.Div([
            html.Label('Number of components'),
            dcc.RangeSlider(
                id='n-components-range',
                min=2,
                max=20,
                step=1,
                marks={i: str(i) for i in range(2, 21, 2)},
                value=[4, 15],
            ),
        ]),
        html.Div([
            html.Label('Number of neighbors'),
            dcc.RangeSlider(
                id='n-neighbors-range',
                min=0,
                max=100,
                step=1,
                marks={i: str(i) for i in range(0, 101, 10)},
                value=[50, 80],
            ),
        ]),
    ], id='params-range-div', style={'display': 'none'}),  # Hidden by default

    html.Div([
        html.Label('Minimum cluster size', id='min_cluster_size'),
        dcc.Input(id='min-cluster-size', type='number', value=38),
        
        html.Label('Min samples', id='min_samples'),
        dcc.Input(id='min-samples', type='number', value=3),
        
        html.Label('Number of components', id='n_components'),
        dcc.Input(id='n-components', type='number', value=5),
        
        html.Label('Number of neighbors', id='n_neighbors'),
        dcc.Input(id='n-neighbors', type='number', value=51),
        
        html.Label('Top n keywords', id='top_n'),
        dcc.Input(id='top-n', type='number', value=10),
    ], id='params-div', style={'display': 'block'}),
    # Checkbox for removing outliers
    dcc.Checklist(
        id='outlier-checklist',
        options=[{'label': 'Remove Outliers', 'value': 'remove_outliers'}],
        value=[],  # Initialize as empty (no removal by default)
    ),
    # Merge cluster dropdown initially hidden
    html.Div([
        html.Div(children='Select clusters to merge'),    
        html.Div(
            id='merge-dropdown-parent',
            children=[
                dcc.Dropdown(
                    id='merge-dropdown',
                    options=[{'label': pred, 'value': pred} for pred in sorted(list(set(doc_df["prediction"].to_list())))],
                    multi=True
                )
            ]
        )
    ], id='merge-div', style={'display': 'none'}),

    html.Button('Run', id='run-button', n_clicks=0),
    # Button to open Cluster Management dialog
    html.Button('Cluster Management', id='cluster-button', n_clicks=0, disabled=True),
    html.Button('Save File', id='save-file-button', n_clicks=0, disabled=True),
    filename_modal,

    # Cluster Management dialog (initially hidden)
    html.Div(id='cluster-dialog', style={'display': 'none'}, children=[
        html.H3('Cluster Management'),
        html.Div(id='topic-list', children=[]),
    ]),

    html.Div(id='output-container'),
#     html.Div([
    dcc.Graph(figure={}, id='controls-and-graph'),
#     ], id='controls-and-graph-div', style={'display': 'none'}),

])

@app.callback(
    Output('selected-labels', 'children'),
    [Input('label-dropdown', 'value')]
)
def display_selected_labels(selected_labels):
    if not selected_labels:
        return "No labels selected."
    
    selected_labels_text = ", ".join(selected_labels)
    return f"Selected labels: {selected_labels_text}"

@app.callback(
    Output('run-button', 'disabled'),
    [Input('label-dropdown', 'value')],
    [Input('merge-dropdown', 'value')]
)
def enable_run_button(selected_labels, selected_clusters):
    # Enable the "Run" button if bad_labels is not empty and either no clusters selected to merge or more than two 
    if not selected_clusters:
        return not bool(selected_labels)
    else:
        if len(selected_clusters) >= 2:
            return False
        print("You need to select at least two clusters to merge")
        return True

    
@app.callback(
    Output('params-div', 'style'),  # Show/hide the parameter input boxes
    Output('params-range-div', 'style'),
    [Input('param-search-checklist', 'value')]
)
def enable_params_div(param_search_value):
    if not param_search_value or 'parameter_search' not in param_search_value:
        return {'display': 'block'}, {'display': 'none'} 
    return {'display': 'none'}, {'display': 'block'}


@app.callback(
    Output('merge-div', 'style'),
    [Input('run-button', 'n_clicks')],
)
def show_merge_div(n_clicks):
    if n_clicks > 0:
        return {'display': 'block'}
    else:
        return {'display': 'none'}

    
@app.callback(
    Output(component_id='controls-and-graph', component_property='figure'),
    Output('cluster-button', 'disabled'),
    Output('save-file-button', 'disabled'),
    [Input('run-button', 'n_clicks')],
    [Input('outlier-checklist', 'value')],
    [State('label-dropdown', 'value')],
    [State('merge-dropdown', 'value')],
    [State('param-search-checklist', 'value')], 
    [State('min-cluster-size', 'value')],  
    [State('min-samples', 'value')],  
    [State('n-components', 'value')],  
    [State('n-neighbors', 'value')],
    [State('top-n', 'value')],
    [State('min-cluster-size-range', 'value')],  
    [State('min-samples-range', 'value')],  
    [State('n-components-range', 'value')],  
    [State('n-neighbors-range', 'value')],
)
def button_click(n_clicks, remove_outliers, selected_labels, clusters_to_merge, 
                 param_search, 
                 min_cluster_size, min_samples, n_components, n_neighbors, top_n,
                 min_cluster_size_range, min_samples_range, n_components_range, n_neighbors_range):
    global topic_model, doc_df, topic_name_dict
    triggered_id = ctx.triggered_id
    if triggered_id == 'run-button':
        params['INFERENCE_LABELS'] = selected_labels
        get_data(selected_labels, read_old=params["READ_OLD_DATA"])
        if param_search:
            hdbscan_umap_params = run_bayesian_search(min_cluster_size_range, min_samples_range, n_components_range, n_neighbors_range)
        else:
            hdbscan_umap_params = {
                'min_cluster_size': min_cluster_size,
                'min_samples': min_samples,
                'n_components': n_components,
                'n_neighbors': n_neighbors,
            }
        
        fig, doc_df, topic_model = model(selected_labels, hdbscan_umap_params, clusters_to_merge=clusters_to_merge, reduce_outliers=remove_outliers, read_old=params["READ_OLD_MODELS"])
        topic_name_dict = dict(zip(topic_model.get_topic_info()['Topic'], topic_model.get_topic_info()['Name']))
        return fig, False, False
    else:
        raise dash.exceptions.PreventUpdate("cancel the callback")
    

@app.callback(
    Output('merge-dropdown', 'options'),  # Update the options of the merge-dropdown
    Output('merge-dropdown', 'value'),    # Clear the selected values of the merge-dropdown
    [Input('merge-dropdown-parent', 'n_clicks')],
    [State('merge-dropdown', 'value')]
)
def update_merge_dropdown_options(n_clicks, selected_clusters):
    global doc_df
    if n_clicks:
        if len(doc_df) > 0:
            options = [{'label': pred, 'value': pred} for pred in sorted(list(set(doc_df["prediction"].to_list())))]
            return options, selected_clusters
        else:
            # https://community.plotly.com/t/callback-that-waits-until-other-callbacks-are-executed/71811/11
            return dash.no_update
    return [], []


@app.callback(
    Output('cluster-dialog', 'style'),
    Output('topic-list', 'children'),
    [Input('cluster-button', 'n_clicks')],
)
def show_cluster_dialog(n_clicks):
        
    if n_clicks % 2 == 1:  # Toggle the dialog on every odd click
        # Create a list of topic details
        topic_details = []
        topic_info_df = topic_model.get_topic_info()
        for _, row in topic_info_df.iterrows():
            topic_id = row["Topic"]
            topic_details.append( 
                html.Div([
                    html.H4(f'Topic {topic_id}'),
                    html.Label('Name:'),
                    dcc.Input(id={"type": "topic-name", "index": topic_id}, type='text', value=topic_name_dict[topic_id]),
                    html.Br(),
                    html.Label(f'Count: {row["Count"]}'),
                    html.Br(),        
                    html.Label(f'Representation: {row["Representation"]}'),
                    html.Br(),        
                    html.Label(f'KeyBERT: {row["KeyBERT"]}'),
                    html.Br(),        
                    html.Label(f'Representative Docs: {row["Representative_Docs"]}'),
                    html.Br(),
                    html.Button(f"Show Snippets", id={"type": "btn-show-snippets", "index": topic_id}, n_clicks=0),
                ])
            )
        
        return {'display': 'block'}, topic_details
    else:
        return {'display': 'none'}, []


def save_output(filename):
    pred_dict = {}
    outlier_count = 0
    for doc, pred in zip(titles_bad_labels_only, doc_df["prediction"].to_list()):
        if pred == -1:
            outlier_count += 1
        pred_dict[doc] = pred

    def set_new_classifications(row):
        if row['snippet'] in pred_dict:
            return topic_name_dict[pred_dict.get(row['snippet'])]
        else: 
            return row['top_classification']
        
    df['new_classification'] = df.apply(lambda row : set_new_classifications(row), axis=1)
    df.to_csv(f'plotly_output/{filename}.tsv', sep='\t')

    
@app.callback(
    Output('filename-modal', 'style'),
    [Input('save-file-button', 'n_clicks'),
     Input('filename-confirm', 'n_clicks'),
     Input('filename-cancel', 'n_clicks')],
    [State('filename-input', 'value')],
    prevent_initial_call=True
)
def handle_filename_modal(save_file_clicks, confirm_clicks, cancel_clicks, filename):
    ctx = dash.callback_context

    if ctx.triggered:
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if triggered_id == 'save-file-button':
            return {'display': 'block'}
        if triggered_id == 'filename-confirm':
            if filename:
                save_output(filename)
                return {'display': 'none'}
            else:
                return {'display': 'block'}
        if triggered_id == 'filename-cancel':
            return {'display': 'none'}

    return {'display': 'none'}


@app.callback(
    Output('output-container', 'style'),
    Input({"type": "topic-name", "index": ALL}, "value"),
)
def update_cluster_names(topic_names):
    global topic_model
    
    if topic_model is None:
        # Handle the case when topic_model is not yet assigned
        return {'display': 'none'}
    # Update the 'Name' column in topic_info_df with the new cluster names
    for i, new_name in enumerate(topic_names):
        topic_name_dict[i-1] = new_name
    
    # You might want to return an empty list or some other placeholder
    return {'display': 'none'}


# Callback to update the modal content when the button is clicked
@app.callback(
    Output("snippets-modal", "is_open"),
    Output("snippets-modal-close", "n_clicks"),
    Output({"type": "btn-show-snippets", "index": ALL}, "n_clicks"),
    Output("snippets-modal-content", "children"),
    Input({"type": "btn-show-snippets", "index": ALL}, "n_clicks"),
    Input("snippets-modal-close", "n_clicks"),
    prevent_initial_call=True
)
def update_modal_content(btn_clicks, close_clicks):
    global doc_df
    modal_contents = []
    is_open = False

    if close_clicks and close_clicks > 0:
        return is_open, 0, [0] * len(btn_clicks), modal_contents


    for topic_id, btn_click in enumerate(btn_clicks):
        if btn_click > 0:
            # Filter snippets for the selected cluster
            cluster_snippets = doc_df[doc_df['prediction'] == topic_id]['title'].tolist()

            # Create a list of snippets
            # snippets_list = [html.P(snippet) for snippet in cluster_snippets]
            # take just up to 10 snippets
            for snippet_ind in range(10 if len(cluster_snippets) > 10 else len(cluster_snippets)):
                modal_contents.append(html.Label(cluster_snippets[snippet_ind]))
                modal_contents.append(html.Br())
            is_open = True
    return is_open, 0, [0] * len(btn_clicks), modal_contents

# cluster management : remove certain clusters 
# save a model and reuse it 

# TODO: ensure hyperparam search is as fast as possible
# TODO: remove carrot
# sentence embedding models 
# remove examples from clusters 
# TODO: use seb's model with <REPLACE TAGS>
# TODO: fix replace when running on the read_old=False mode
# consider using LLAMA
# Should we remove duplicates because of what we get for most representative docs?
if __name__ == '__main__':
    app.run_server(debug=True)
    
# Calculating embeddings is very long
# Fitting a model: 80.08043503761292

# embeddings_good_labels_only 38834 - car
# 67514 - ai

# %%



