import base64
import io
import itertools
import pickle
from GlanosBERTopic import GlanosBERTopic
from bertopic import BERTopic
from chatintents import ChatIntents, get_hyperparameter_string
import dash_bootstrap_components as dbc
from dash import ctx, ALL
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import dash
import time
from hyperopt import hp
import pandas as pd
from utils import create_replace_no_tags_embeddings, load_data, replace_hashtags_in_named_entity_tags, str_to_numpy_converter
from clustering_utils import remove_stopwords
from GlanosSentenceTransformers import GlanosSentenceTransformer
from embeddings_training_utils import train, filter_top_values, get_classification_counts
import os
import sys
from flask_caching import Cache

from tqdm.notebook import tqdm
tqdm.pandas()
sys.path.append('chat-intents/chatintents')

OUTPUT_DIR = 'output'
DATA_DIR = 'data'
EMBEDDING_MODEL_DIR = 'embedding_models'
REPLACE_HASHTAG_IN_NAMED_ENTITIES_DIR = 'replacing_#_in_NE'
REMOVE_NAMED_ENTITY_TAGS = 'removing_NE'

TOPIC_MODEL_DIR = 'topic_models'

DEFAULT_TOPIC_MODEL = ''
DEFAULT_EMBEDDING_MODEL = 'all-MiniLM-L12-v2'
DEFAULT_MIN_CLUSTER_SIZE = 38
DEFAULT_MIN_SAMPLES = 3
DEFAULT_N_COMPONENTS = 5
DEFAULT_N_NEIGHBORS = 51
DEFAULT_TOP_N_TOPIC_WORDS = 10

# 'remove_named_entity_tags', 'replace_hashtag_in_named_entities'
DEFAULT_SNIPPET_PROCESSING_STRATEGY = 'replace_hashtag_in_named_entities'
params = {
    "penalty": 'label_count',
    "occurrence_cutoff": 2,
    "snippet_column_name": 'embedding_input',
    "max_evals": 10,
    "bayesian_search_metric": 'topic_diversity+bert',
    "epochs": 1,
    "unfreeze_layers": 2,
    "batch_size": 32,
    "val_dev_size": 100
}

app = dash.Dash(external_stylesheets=[
                dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
cache = Cache(app.server, config={
    # 'CACHE_TYPE': 'redis',
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory',
    'CACHE_DEFAULT_TIMEOUT': 0,
    'CACHE_THRESHOLD': 200
})


def train_classification_model(pre_trained_model, df, classifications, params):
    '''
    Train the sentence embedding model using classification labels as classes to create triplets for training
    '''
    params["WARMUP_STEPS"] = int(len(df) * params["epochs"] * 0.1)
    training_datasets = [df]
    classification_lists = [classifications]
    model_fit, test_evaluator, model = train(
        training_datasets, classification_lists, params, pre_trained_model)
    # print('Score', model.evaluate(test_evaluator))

    # frozen_model = load_model() if not params["INITIALIZED_MODEL"] else load_model(
    #     model=params["INITIALIZED_MODEL"])
    # print('Baseline', frozen_model.evaluate(test_evaluator))
    return model


def get_data(selected_labels, classifications, snippet_processing_strategy, train_embeddings=False):
    '''
    Modify the snipptes according to the selected processing strategy, train the sentence embedding model if needed, encode the snippets, and split into inference and training sets
    '''
    df = cache.get('df')
    embedding_model = cache.get('embedding_model')

    # Pre-process the input
    if snippet_processing_strategy == 'remove_named_entity_tags':
        print("Removing named entity tags...")
        df = create_replace_no_tags_embeddings(df)
    elif snippet_processing_strategy == 'replace_hashtag_in_named_entities':
        print("Replacing hashtag in named entity tags...")
        df = replace_hashtags_in_named_entity_tags(df)
    else:
        raise Exception(f"This snippet preprocessing strategy is not valid: {snippet_processing_strategy}")

    df['embedding_input'] = df.apply(
            lambda row: remove_stopwords(row['embedding_input']), axis=1)

    df_training = df[~df['top_classification'].isin(selected_labels)]

    if 'embedding' not in df.columns:
        # Train the sentence embedding model
        if train_embeddings:
            tic = time.time()
            print("Training sentence embeddings...")
            embedding_model = train_classification_model(
                embedding_model, df_training, classifications, params)
            cache.set('embedding_model', embedding_model)
            print("Training sentence embeddings:", time.time()-tic)

        # Encode with the trained model
        tic = time.time()
        print("Encoding snippets...")

        embeddings = embedding_model.encode(
            df["embedding_input"].to_list(), show_progress_bar=True)

        embeddings = [row for row in embeddings]
        df['embedding'] = embeddings

        print("Encoding snippets:", time.time()-tic)
    else:
        print("Reading embeddings")

        if isinstance(df['embedding'].iloc[0], str):
            print("Converting string to numpy")
            df['embedding'] = df['embedding'].apply(str_to_numpy_converter)

    # Split the data again to include generated embeddings
    df_training = df[~df['top_classification'].isin(selected_labels)]
    df_inference = df[df['top_classification'].isin(
        selected_labels)]
    print('Sample of the snippet encoding format',
          df_training["embedding_input"].to_list()[:2])
    cache.set('df', df)
    return df_training, df_inference


def run_bayesian_search(df_training, df_inference, is_topic_model_training, params, min_cluster_size_range, min_samples_range, n_components_range, n_neighbors_range):
    '''
    Run the Bayesian search for the best hyperparameters
    '''
    hspace = {
        "n_neighbors": hp.choice('n_neighbors', n_neighbors_range),
        "n_components": hp.choice('n_components', n_components_range),
        "min_cluster_size": hp.choice('min_cluster_size', min_cluster_size_range),
        "min_samples": hp.choice('min_samples', min_samples_range),
        "is_topic_model_training": is_topic_model_training,
        "random_state": 42
    }
    label_lower = 28
    label_upper = 50
    chat_intents_model = ChatIntents(df_training, df_inference)
    best_params, best_clusters, trials = chat_intents_model.bayesian_search(space=hspace,
                                                                            label_lower=label_lower,
                                                                            label_upper=label_upper,
                                                                            params=params)
    return best_params, chat_intents_model.bayesian_search_results


def create_and_train_topic_model(embedding_model, topic_model, topic_model_name, df_training, df_inference, params, reduce_outliers, is_topic_model_training):
    '''
    Create a new topic model or load an existing one and train it
    '''
    if topic_model_name.strip() == DEFAULT_TOPIC_MODEL: # if no model selected
        topic_model = GlanosBERTopic(params, df_training, df_inference, topic_model_name, embedding_model)
    else:
        topic_model = BERTopic.load(f'{OUTPUT_DIR}/{TOPIC_MODEL_DIR}/{topic_model_name}')
        topic_model.df_training = df_training
        topic_model.df_inference = df_inference
        topic_model.topic_model_name = topic_model_name

    topic_model.train(is_topic_model_training)

    if reduce_outliers:
        topic_model.reduce_outliers_and_update()
    topic_model.remove_topic_ids_from_topic_labels()
    topic_model.update_topics()

    return topic_model


@app.callback(
    Output('run-btn', 'disabled'),
    Input('label-dropdown', 'value'),
    Input('topic-merge-dropdown', 'value')
)
def enable_run_button(selected_labels, selected_clusters):
    '''
    Enable the "Run" button if inference is not empty and either no clusters selected to merge or more than two
    '''
    if not selected_clusters:
        return not bool(selected_labels)
    else:
        if len(selected_clusters) >= 2:
            return False
        print("You need to select at least two clusters to merge")
        return True


@app.callback(
    Output('params-div', 'style'),
    Output('params-range-div', 'style'),
    Input('param-search-checklist', 'value')
)
def enable_params_div(param_search_value):
    if not param_search_value or 'parameter_search' not in param_search_value:
        return {'display': 'block'}, {'display': 'none'}
    return {'display': 'none'}, {'display': 'block'}


@app.callback(
    Output('snippet-processing-strategy', 'data'),
    Input('embedding-model-dropdown', 'value'),
    Input('topic-model-dropdown', 'value'),
    State('snippet-processing-strategy', 'data'),
)
def assign_selected_models(embedding_model_value, topic_model_value, snippet_processing_strategy):
    '''
    Assign the selected embedding and topic models to the cache
    '''
    if type(embedding_model_value) == list:
        embedding_model_value = embedding_model_value[0]
    if type(topic_model_value) == list:
        topic_model_name = topic_model_value[0]
    else:
        topic_model_name = topic_model_value
    cache.set('topic_model_name', topic_model_name)

    # Read from huggingface
    if DEFAULT_EMBEDDING_MODEL in embedding_model_value:
        embedding_model = GlanosSentenceTransformer(DEFAULT_EMBEDDING_MODEL)
    # Read locally with a path
    else:
        if REPLACE_HASHTAG_IN_NAMED_ENTITIES_DIR in embedding_model_value:
            snippet_processing_strategy = 'replace_hashtag_in_named_entities'
        elif REMOVE_NAMED_ENTITY_TAGS in embedding_model_value:
            snippet_processing_strategy = 'remove_named_entity_tags'
        embedding_model = GlanosSentenceTransformer(
                f'{OUTPUT_DIR}/{EMBEDDING_MODEL_DIR}/{embedding_model_value}')
    cache.set('embedding_model', embedding_model)
    return snippet_processing_strategy


def get_all_model_options(dir):
    '''
    Get all the models in the specified directory
    '''
    if dir == EMBEDDING_MODEL_DIR:
        subdir_list = [subdir for subdir in os.listdir(
            f'{OUTPUT_DIR}/{dir}/') if "." not in subdir]
        file_list = []
        for subdir in subdir_list:
            file_list += [f'{subdir}/{file}' for file in os.listdir(
                f'{OUTPUT_DIR}/{dir}/{subdir}') if not file.startswith(".")]

        pretrained_embedding_models = [
            {'label': f'{REPLACE_HASHTAG_IN_NAMED_ENTITIES_DIR}/{DEFAULT_EMBEDDING_MODEL}',
                'value': f'{REPLACE_HASHTAG_IN_NAMED_ENTITIES_DIR}/{DEFAULT_EMBEDDING_MODEL}'},
            {'label': f'{REMOVE_NAMED_ENTITY_TAGS}/{DEFAULT_EMBEDDING_MODEL}', 'value': f'{REMOVE_NAMED_ENTITY_TAGS}/{DEFAULT_EMBEDDING_MODEL}'}]

        return [{'label': file, 'value': file} for file in file_list] + pretrained_embedding_models
    elif dir == TOPIC_MODEL_DIR:
        file_list = [file for file in os.listdir(
            f'{OUTPUT_DIR}/{dir}') if not file.startswith(".")]
        pretrained_topic_models = [
            {'label': DEFAULT_TOPIC_MODEL, 'value': DEFAULT_TOPIC_MODEL}]
        return [{'label': file, 'value': file} for file in file_list] + pretrained_topic_models
    else:
        return []

def save_state(app_layout, figure, to_cache=True):
    '''
    Save the current state of the app to the cache. Currently not necessary since the state gets updated with every change
    '''
    def set_figure_at_id(d, target_id, fig):
        """
        Recursively set the 'figure' property at the specified 'id' in a nested dictionary.

        Parameters:
        - d: The input dictionary
        - target_id: The 'id' at which to set the 'figure'
        - fig: The value to set for the 'figure' property
        """
        if isinstance(d, dict):
            if 'id' in d and d['id'] == target_id:
                d['figure'] = fig
            else:
                for key, value in d.items():
                    set_figure_at_id(value, target_id, fig)
        elif isinstance(d, list):
            for item in d:
                set_figure_at_id(item, target_id, fig)

    set_figure_at_id(app_layout, 'controls-and-graph', figure)
    if to_cache:
        cache.set('app_layout', app_layout)
    else:
        with open('file.pkl', 'wb') as file:
            pickle.dump(app_layout, file) 
    print("State saved.")

@app.callback(
    Output('state-confirm-dialog', 'displayed'),
    Output('state-confirm-dialog', 'message'),
    Output('app-layout-div', 'children'),
    Input('save-state-btn', 'n_clicks'),
    Input('load-state-btn', 'n_clicks'),
    State('app-layout-div', 'children'),
    State('controls-and-graph', 'figure'),
    prevent_initial_call=True
)
def save_load_state_callback(save_state_n_clicks, load_state_n_clicks, app_layout, fig):
    if ctx.triggered_id == 'save-state-btn':
        save_state(app_layout, fig)
        return True, 'State saved.', app_layout
    elif ctx.triggered_id == 'load-state-btn':
        return False, '', cache.get('app_layout')
    raise dash.exceptions.PreventUpdate


@app.callback(
    Output('embedding-model-dropdown', 'options'),
    Output('topic-model-dropdown', 'options'),
    Input('embedding-model-dropdown-div', 'n_clicks'),
    Input('topic-model-dropdown-div', 'n_clicks'),
)
def enable_training_divs(embedding_model_n_clicks, topic_model_n_clicks):
    embedding_options = get_all_model_options(
        EMBEDDING_MODEL_DIR)
    topic_model_options = get_all_model_options(
        TOPIC_MODEL_DIR)
    return embedding_options, topic_model_options


@app.callback(
    Output('controls-and-graph', 'figure'),
    Output('hyperparameters-display', 'children'),
    Output('first-run-flag', 'data'),
    Input('run-btn', 'n_clicks'),
    Input('cluster-dialog', 'is_open'),
    Input('label-dropdown', 'value'),
    State('label-dropdown', 'options'),    
    State('embedding-training-checklist', 'value'),
    State('topic-model-training-checklist', 'value'),
    State('outlier-checklist', 'value'),
    State('param-search-checklist', 'value'),
    State('min-cluster-size', 'value'),
    State('min-samples', 'value'),
    State('n-components', 'value'),
    State('n-neighbors', 'value'),
    State('top-n-topics', 'value'),
    State('min-cluster-size-range', 'value'),
    State('min-samples-range', 'value'),
    State('n-components-range', 'value'),
    State('n-neighbors-range', 'value'),
    State('bayesian-iterations', 'value'),
    State('hyperparameters-display', 'children'),
    State('controls-and-graph', 'figure'),
    State('first-run-flag', 'data'),
    State('snippet-processing-strategy', 'data'),
    State('app-layout-div', 'children'),
    prevent_initial_call=True
)
def run_button_click(n_clicks, is_cluster_dialog_open, selected_labels, 
                 classification_labels, embedding_training_checklist, topic_model_training_checklist, reduce_outliers, param_search,
                 min_cluster_size_value, min_samples_value, n_components_value, n_neighbors_value, top_n_topic_words_value,
                 min_cluster_size_range, min_samples_range, n_components_range, n_neighbors_range, bayesian_iterations, hyperparam_string,
                 fig, first_run_flag, snippet_processing_strategy, app_layout):
    embedding_model = cache.get('embedding_model')
    topic_model = cache.get('topic_model')
    topic_model_name = cache.get('topic_model_name')
    df = cache.get('df')

    if ctx.triggered_id == 'label-dropdown':
        save_state(app_layout, fig)
        return fig, hyperparam_string, True

    classifications = {classification_label['value']: i for i, classification_label in enumerate(classification_labels)}

    is_topic_model_training = 'topic_model_training' in topic_model_training_checklist or topic_model_name == "DEFAULT_TOPIC_MODEL"

    # On closing the cluster dialog, update the figure since the topic assignment might have changed
    if ctx.triggered_id == 'cluster-dialog' and not is_cluster_dialog_open:
        fig = topic_model.reduce_dim_and_visualize_documents(reuse_reduced_embeddings=True) # reuse reduced_embeddings to not have to recompute them and save time 
        save_state(app_layout, fig)
        return fig, hyperparam_string, first_run_flag

    if ctx.triggered_id == 'run-btn':
        if 'top_classification' not in df.columns:
            df['top_classification'] = df.apply(
                lambda row: filter_top_values(row, selected_labels), axis=1)
            cache.set('df', df)

        if first_run_flag or param_search:
            params['inference_labels'] = selected_labels
            df_training, df_inference = get_data(selected_labels, classifications, snippet_processing_strategy,
                    train_embeddings='embedding_training' in embedding_training_checklist)
        if param_search:
            params["max_evals"] = bayesian_iterations
            hdbscan_umap_params, bayesian_search_results = run_bayesian_search(
                    df_training,
                    df_inference,
                    is_topic_model_training, 
                    {**params, **{'topic_model_name': topic_model_name, 'embedding_model': embedding_model, 'top_n_topic_words': top_n_topic_words_value, 'is_topic_model_training': is_topic_model_training}},
                    min_cluster_size_range, 
                    min_samples_range, 
                    n_components_range, 
                    n_neighbors_range
                )

            hyperparameters_display_list = [html.Div("Trials:"), html.Br()]
            for epoch, result in enumerate(bayesian_search_results):
                space, loss, label_count = result
                hyperparameters_display_list += [html.Div(f"Epoch {epoch+1}: loss {loss:.2f}, label count {label_count} for {get_hyperparameter_string(space)}"), html.Br()]
            hyperparameters_display_list += [html.Div(f"Best hyperparameters {get_hyperparameter_string(hdbscan_umap_params)}"), html.Br()]
        else:
            hdbscan_umap_params = {
                "min_cluster_size": min_cluster_size_value,
                "min_samples": min_samples_value,
                "n_components": n_components_value,
                "n_neighbors": n_neighbors_value,
                "top_n_topic_words": top_n_topic_words_value}
            hyperparameters_display_list = [
                html.Div(f"Hyperparameters: {get_hyperparameter_string(hdbscan_umap_params)}"), 
                html.Br()
                ]

        if first_run_flag:
            topic_model = create_and_train_topic_model(embedding_model, topic_model, topic_model_name, df_training, df_inference, {**params, **hdbscan_umap_params}, reduce_outliers, is_topic_model_training)
            first_run_flag = False
        elif reduce_outliers:
            topic_model.reduce_outliers_and_update()

        fig = topic_model.reduce_dim_and_visualize_documents()
        topic_model.update_topics()
        save_state(app_layout, fig)
        cache.set('topic_model', topic_model)
        return fig, hyperparameters_display_list, first_run_flag
    else:
        raise dash.exceptions.PreventUpdate


@app.callback(
    Output('cluster-dialog', 'is_open'),
    Output('topic-list', 'children'),
    Output('add-new-cluster-confirm-dialog', 'displayed'),
    Output('add-new-cluster-confirm-dialog', 'message'),
    Input('cluster-managment-btn', 'n_clicks'),
    Input("snippets-modal", "is_open"),
    Input('add-new-cluster-btn', 'n_clicks'),
    Input('snippets-modal-add-new-cluster-btn', 'n_clicks'),
    Input('recalculate-cluster-info-btn', 'n_clicks'),
    prevent_initial_call=True
    )
def show_cluster_dialog(cluster_managment_n_clicks, is_open, add_new_cluster_n_clicks, snippets_modal_add_new_cluster_n_clicks, recalculate_cluster_info_n_clicks):
    topic_model = cache.get('topic_model')

    topic_model.recalculate_topic_sizes()

    if ctx.triggered_id == 'add-new-cluster-btn' or ctx.triggered_id == 'snippets-modal-add-new-cluster-btn':
            new_topic_label = topic_model.add_topic()
            is_confirmation_displayed = True
    else:
            new_topic_label = ''
            is_confirmation_displayed = False

    if ctx.triggered_id == 'recalculate-cluster-info-btn':
        topic_model.update_topics()

    # Create a list of topic details
    topic_details = []
    topic_info_df = topic_model.get_topic_info()

    for _, row in topic_info_df.iterrows():
            topic_id = row["Topic"]
            topic_size = topic_model.topic_sizes_[topic_id]
            if topic_size > 0:
                topic_details.append(
                    html.Div([
                        html.H4(f'Topic {topic_id}' if topic_id != -1 else 'Outliers'),
                        html.Label('Name:'),
                        dcc.Input(id={"type": "topic-name", "index": topic_id},
                                  type='text', value=topic_model.topic_labels_[topic_id]),
                        html.Br(),
                        html.Label(f'Size: {topic_size}', id={
                                   "type": "topic-count", "index": topic_id}),
                        html.Br(),
                        html.Label(f'Keywords: {row["Representation"]}'),
                        html.Br(),
                        html.Label(f'KeyBERT: {row["KeyBERT"]}'),
                        html.Br(),
                        html.Label(
                            f'Representative Docs: {row["Representative_Docs"]}'),
                        html.Br(),
                        dbc.Button(f"Show Snippets", id={
                                    "type": "btn-show-snippets", "index": topic_id}, n_clicks=0),
                        html.Br(),
                    ])
                )
            else:
                topic_details.append(
                    html.Div([
                        html.H4(f'Topic {topic_id}' if topic_id != -1 else 'Outliers'),
                        html.Label('Name:'),
                        dcc.Input(id={"type": "topic-name", "index": topic_id},
                                  type='text', value=topic_model.topic_labels_[topic_id]),
                        html.Br(),
                        html.Label(f'Size: {topic_size}', id={
                                   "type": "topic-count", "index": topic_id}),
                        html.Br(),
                    ])
                )

    cache.set('topic_model', topic_model)
    return True, topic_details, is_confirmation_displayed, f'New topic {new_topic_label} was just added.'


def save_output(filename, is_embedding_included, last_opened_save_modal, snippet_processing_strategy):
    topic_model = cache.get('topic_model')
    embedding_model = cache.get('embedding_model')

    # Save file
    if last_opened_save_modal == 'save-file-btn':
        pred_dict = {}
        outlier_count = 0
        for doc, pred in zip(topic_model.get_documents(), topic_model.topics_):
            if pred == -1:
                outlier_count += 1
            pred_dict[doc] = pred

        def set_new_classifications(row):
            if row['snippet'] in pred_dict:
                return topic_model.topic_labels_[pred_dict.get(row['snippet'])]
            else:
                return row['top_classification']
            
        entire_df = pd.concat([topic_model.df_inference, topic_model.df_training], ignore_index=True)

        entire_df['new_classification'] = entire_df.apply(
            lambda row: set_new_classifications(row), axis=1)

        entire_df = entire_df if is_embedding_included else entire_df.drop(
            columns=['embedding'])
        entire_df["old_classification"] = entire_df["classification"]
        entire_df["classification"] = entire_df["new_classification"]
        entire_df = entire_df.drop(
            columns=['embedding_input', 'new_classification', 'top_classification'])
        entire_df.to_csv(f'{OUTPUT_DIR}/{DATA_DIR}/{filename}.tsv', sep='\t', index=False)

    # Save embedding model
    elif last_opened_save_modal == 'save-embedding-model-btn':
        if snippet_processing_strategy == 'replace_hashtag_in_named_entities':
            filename = f'{REPLACE_HASHTAG_IN_NAMED_ENTITIES_DIR}/{filename}'
        elif snippet_processing_strategy == 'remove_named_entity_tags':
            filename = f'{REMOVE_NAMED_ENTITY_TAGS}/{filename}'
        embedding_model.save(
            f'{OUTPUT_DIR}/{EMBEDDING_MODEL_DIR}/{filename}')

    # Save topic model
    elif last_opened_save_modal == 'save-topic-model-btn':
        topic_model.save(f'{OUTPUT_DIR}/{TOPIC_MODEL_DIR}/{filename}')
    cache.set('topic_model', topic_model)


@app.callback(
    Output('filename-modal', 'style'),
    Output('filename-embeddings-checklist-div', 'style'),
    Output('last-opened-save-modal', 'data'),
    Input('save-file-btn', 'n_clicks'),
    Input('save-embedding-model-btn', 'n_clicks'),
    Input('save-topic-model-btn', 'n_clicks'),
    Input('filename-confirm', 'n_clicks'),
    Input('filename-cancel', 'n_clicks'),
    State('filename-input', 'value'),
    State('filename-embeddings-checklist', 'value'),
    State('last-opened-save-modal', 'data'),
    State('snippet-processing-strategy', 'data'),
    prevent_initial_call=True
)
def handle_saving_modal(save_file_clicks, save_embedding_model_clicks, save_topic_model_clicks,
                        confirm_clicks, cancel_clicks, filename, embeddings_checklist_value, last_opened_save_modal, snippet_processing_strategy):

    embedding_checklist_style = {'display': 'block'} if last_opened_save_modal == 'save-file-btn' else {
        'display': 'none'}
    ctx = dash.callback_context

    if ctx.triggered:
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if triggered_id.startswith('save'):
            last_opened_save_modal = triggered_id
            embedding_checklist_style = {'display': 'block'} if last_opened_save_modal == 'save-file-btn' else {
                'display': 'none'}
            return {'display': 'block'}, embedding_checklist_style, last_opened_save_modal
        if triggered_id == 'filename-confirm':
            if filename:
                save_output(
                    filename, 'embeddings' in embeddings_checklist_value, last_opened_save_modal, snippet_processing_strategy)
                return {'display': 'none'}, {'display': 'none'}, last_opened_save_modal
            else:
                return {'display': 'block'}, embedding_checklist_style, last_opened_save_modal
        if triggered_id == 'filename-cancel':
            return {'display': 'none'}, {'display': 'none'}, last_opened_save_modal

    return {'display': 'none'}, {'display': 'none'}, last_opened_save_modal


@app.callback(
    Output("dropdown-assign", "style"),
    Input({"type": "topic-name", "index": ALL}, "value"),
    State({"type": "topic-name", "index": ALL}, "id"),
)
def update_cluster_names(topic_names, topic_ids):
    topic_model = cache.get('topic_model')

    if ctx.triggered_id and 'index' in ctx.triggered_id:
        triggered_topic_id = ctx.triggered_id.get('index')

        if topic_model is None:
            # Handle the case when topic_model is not yet assigned
            return {'display': 'block'}
        
        dropdown_options = []
        custom_labels = {}
        for topic_id_dict, topic_name in zip(topic_ids, topic_names):
            topic_id = topic_id_dict.get('index')
            if triggered_topic_id == topic_id:
                topic_model.topic_labels_[triggered_topic_id] = topic_name
            dropdown_options.append({'label': topic_name, 'value': topic_id})
            custom_labels[topic_id] = topic_name
        topic_model.set_topic_labels(custom_labels)

    cache.set('topic_model', topic_model)
    return {'display': 'block'}


@app.callback(
    Output("output-container", "style"),
    Input("snippets-checklist", "value"),
    prevent_initial_call=True,
)
def handle_selected_snippets(selected_snippets):
    cache.set("selected_snippets", selected_snippets)
    return {'display': 'block'}


@app.callback(
    Output("snippets-modal", "is_open"),
    Output("snippets-modal-content", "children"),
    Output("dropdown-assign", "options"),
    Output("snippets-modal-title", "children"),
    Output("current-topic-id", 'data'),
    Input({"type": "btn-show-snippets", "index": ALL}, "n_clicks"),
    Input('snippet-search', 'value'),
    Input("snippets-modal", "is_open"),
    Input("btn-remove-cluster", "n_clicks"),
    Input("btn-select-all", "n_clicks"),
    State("current-topic-id", 'data'),
    prevent_initial_call=True
)
def show_cluster_dialog(btn_show_clicks, search_term, is_open, btn_remove_clicks, btn_select_all_clicks, current_topic_id):
    topic_model = cache.get('topic_model')
    selected_snippets = cache.get("selected_snippets")

    if ctx.triggered_id == "btn-remove-cluster":
        topic_model.remove_topic(current_topic_id)

    # When closing the dialog, reset the current_topic_id and empty the selected_snippets
    if ctx.triggered_id == "snippets-modal" and not is_open:
        current_topic_id = -2
        selected_snippets = []

    if isinstance(ctx.triggered_id, dict) and ctx.triggered_id.get('type') == 'btn-show-snippets':
        current_topic_id = ctx.triggered_id.get('index')

    if current_topic_id != -2 and not all(element == 0 for element in btn_show_clicks): # the second condition because when clicking cluster management button for whatever reason it triggers the click callback but is not registered in btn_show_clicks
        is_open = True

    displayed_snippets = topic_model.get_documents_from_topic(
                current_topic_id, search_term)
        
    if ctx.triggered_id == 'btn-select-all':
        selected_snippets = [snippet['value'] for snippet in displayed_snippets]

    checklist = dbc.Checklist(
            options=displayed_snippets,
            value=selected_snippets,
            id="snippets-checklist",
        )

    if ctx.triggered_id == 'snippet-search':
        is_open = True

    # Trigger closing after removing a cluster
    if ctx.triggered_id == "btn-remove-cluster":
        is_open = False

    topic_labels_excluding_current = {
        key: value for key, value in topic_model.topic_labels_.items() if key != current_topic_id}

    cache.set('topic_model', topic_model)

    return is_open, checklist, [{'label': cluster_name, 'value': cluster_id} for cluster_id, cluster_name in topic_labels_excluding_current.items()], f"Snippets for Cluster {current_topic_id}" if current_topic_id != -1 else "Outliers", current_topic_id


# Callback to handle moving snippets to a different cluster
@app.callback(
    Output("snippets-checklist", "value"),
    Output("snippets-checklist", "options"),
    Input("btn-assign", "n_clicks"),
    State("snippets-checklist", "value"),
    State("dropdown-assign", "value"),
    State("current-topic-id", 'data'),
    prevent_initial_call=True,
)
def handle_assign_snippets(assign_clicks, selected_snippets, selected_topic, current_topic_id):
    topic_model = cache.get('topic_model')

    if not assign_clicks or not selected_snippets:
        raise dash.exceptions.PreventUpdate

    # Move the selected snippets to the target cluster
    for snippet_ind in selected_snippets:
        topic_model.topics_[snippet_ind] = selected_topic

    topic_model.recalculate_topic_sizes()
    
    cache.set('topic_model', topic_model)

    if current_topic_id > -2:
        return [], topic_model.get_documents_from_topic(current_topic_id)
    else:
        raise dash.exceptions.PreventUpdate


@app.callback(
    Output('upload-modal', 'is_open'),
    Input('upload-btn', 'n_clicks'),
    State('upload-modal', 'is_open'),
)
def toggle_modal(n_clicks, is_open):
    if n_clicks or is_open:
        return not is_open
    return is_open


@app.callback(
    Output('selected-file-output', 'children'),
    Output('label-dropdown', 'options'),
    Output('embedding-model-dropdown-div', 'style'),
    State('upload-data', 'filename'),
    Input('upload-data', 'contents'),
    prevent_initial_call=True,
)
def handle_file_selection(filename, contents):
    '''
    Load one of the data formats: csv, tsv or json as a datframe and extract the classification labels to populate the dropdown
    '''
    if contents is None:
        return "No file selected.", [], {'display': 'block'}

    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        if 'csv' in filename:
            print("Reading csv")
            df = load_data(io.StringIO(decoded.decode('utf-8')), format='csv')
        elif 'tsv' in filename:
            print("Reading tsv")
            df = load_data(io.StringIO(decoded.decode('utf-8')))
        elif 'json' in filename:
            print("Reading json")
            df = load_data(io.StringIO(decoded.decode('utf-8')), format='json')
        else:
            return 'Please upload a .csv, .tsv or .json file.', [], {'display': 'block'}

        classifications = get_classification_counts(df)
        classifications_sorted_by_count = [classification for classification, count in sorted(classifications.items(),
                                                                                              key=lambda x: x[1], reverse=True)]
        cache.set('df', df)

        if 'embedding' in df.columns:
            return f"Selected file: {filename} with embeddings", [{'label': label, 'value': label}
                                                                  for label in classifications_sorted_by_count], {'display': 'none'}
        else:
            return f"Selected file: {filename}", [{'label': label, 'value': label}
                                                  for label in classifications_sorted_by_count], {'display': 'block'}
    except Exception as e:
        print(e)
        return 'There was an error processing this file.', [], {'display': 'block'}


@app.callback(
    Output('topic-model-training-checklist', 'value'),
    Input('param-search-checklist', 'value'),
    Input('topic-model-dropdown', 'value'),
    prevent_initial_call=True
)
def update_topic_model_training(selected_options, topic_model_value):
    if selected_options or topic_model_value == DEFAULT_TOPIC_MODEL:
        return ['topic_model_training']
    else:
        return dash.no_update


def create_layout(classifications):

    filename_modal = dcc.Loading(
        type="default",
        children=html.Div(id='filename-modal', style={'display': 'none'}, children=[
            html.Div([
                html.Label('Enter the filename:'),
                dcc.Input(id='filename-input', type='text',
                          placeholder='Filename'),
                html.Div([
                    dbc.Checklist(
                        options=[{'label': 'Include embeddings',
                                  'value': 'embeddings'}],
                        value=['embeddings'],
                        id="filename-embeddings-checklist",
                    ),
                ], style={'display': 'none'}, id='filename-embeddings-checklist-div'),
                html.Button('Save', id='filename-confirm'),
                html.Button('Cancel', id='filename-cancel'),
            ], className='modal-content')
        ])
    )

    snippets_modal = dbc.Modal(
        [
            dbc.ModalHeader([
                dbc.ModalTitle(id="snippets-modal-title"),
            ]),
            dbc.ModalFooter([
                html.Div([
                            dcc.Dropdown(
                                id="dropdown-assign",
                                options=[],
                            ),
                            dbc.Button(
                                "Assign", id="btn-assign", n_clicks=0)
                            ], style={'width': '100%'}),
                dbc.Button(
                            "Select All", id="btn-select-all", n_clicks=0),
                dcc.Input(id='snippet-search', type='text',
                              placeholder='Search Snippets'),
                dbc.Button("Remove Cluster",
                           id="btn-remove-cluster", n_clicks=0),
                dbc.Button('Add New Cluster', id='snippets-modal-add-new-cluster-btn', n_clicks=0),
            ]
            ),
            dbc.ModalBody(id="snippets-modal-content",
                          children='Snippets listed'),
        ],
        id="snippets-modal",
        is_open=False,
        scrollable=True,
        size="lg"
    )

    upload_modal = dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    )

    label_selection = html.Div([
        html.Div(children='Select classification labels for inference'),
        dcc.Dropdown(
            id='label-dropdown',
            options=[{'label': label, 'value': label}
                     for label in classifications],
            multi=True
        ),
    ])

    main_layout = html.Div([
        snippets_modal,
        label_selection,
        html.Div([
            dcc.Checklist(
                id='embedding-training-checklist',
                options=[{'label': 'Train the embedding model',
                          'value': 'embedding_training'}],
                value=[]
            ),
            dcc.Dropdown(
                id='embedding-model-dropdown',
                options=get_all_model_options(
                    EMBEDDING_MODEL_DIR),
                value=[DEFAULT_EMBEDDING_MODEL],
                multi=False
            )], id='embedding-model-dropdown-div'
        ),
        html.Div([
            dcc.Checklist(
                id='topic-model-training-checklist',
                options=[{'label': 'Train the topic model',
                          'value': 'topic_model_training'}],
                value=['topic_model_training']
            ),
            dcc.Dropdown(
                id='topic-model-dropdown',
                options=get_all_model_options(
                    TOPIC_MODEL_DIR),
                value=[DEFAULT_TOPIC_MODEL],
                multi=False
            )], id='topic-model-dropdown-div'
        ),
        dcc.Checklist(
            id='param-search-checklist',
            options=[{'label': 'Run parameter search',
                      'value': 'parameter_search'}],
        ),
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
                    min=2,
                    max=100,
                    # TODO make it dynamically int(len(df_inference)/4) if len(df_inference) > 0 else 100
                    step=1,
                    marks={i: str(i) for i in itertools.chain(range(10, 101, 10), [2])},
                    # TODO make it dynamically marks={i: str(i) for i in itertools.chain(range(10, int(len(df_inference)/4)+1 if len(df_inference) > 0 else 101, int(len(df_inference)/40) if len(df_inference) > 0 else 10), [2])},
                    value=[50, 80],
                ),
            ]),
            html.Div([
                html.Label('Number of iterations'),
                dcc.Input(id='bayesian-iterations', type='number', value=5),
            ])
        ], id='params-range-div', style={'display': 'none'}),  # Hidden by default

        html.Div([
            html.Label('Minimum cluster size'),
            dcc.Input(id='min-cluster-size', type='number', value=DEFAULT_MIN_CLUSTER_SIZE),

            html.Label('Min samples'),
            dcc.Input(id='min-samples', type='number', value=DEFAULT_MIN_SAMPLES),

            html.Label('Number of components'),
            dcc.Input(id='n-components', type='number', value=DEFAULT_N_COMPONENTS),

            html.Label('Number of neighbors'),
            dcc.Input(id='n-neighbors', type='number', value=DEFAULT_N_NEIGHBORS),
        ], id='params-div', style={'display': 'block'}),
        html.Div([
            html.Label('Top n keywords'),
            dcc.Input(id='top-n-topics', type='number', value=DEFAULT_TOP_N_TOPIC_WORDS),
        ], id='other-params-div', style={'display': 'block'}),
        # Checkbox for removing outliers
        dcc.Checklist(
            id='outlier-checklist',
            options=[{'label': 'Reduce outliers', 'value': 'reduce_outliers'}],
            value=[],
        ),

        html.Div([
            html.Div(children='Select clusters to merge'),
            html.Div(
                id='topic-merge-dropdown-div',
                children=[
                    dcc.Dropdown(
                        id='topic-merge-dropdown',
                        multi=True
                    )
                ]
            )
        ], id='merge-div', style={'display': 'none'}),

        html.Button('Run', id='run-btn', n_clicks=0),
        html.Button('Save state', id='save-state-btn', n_clicks=0),
        html.Button('Load state', id='load-state-btn', n_clicks=0),
        html.Button('Cluster Management', id='cluster-managment-btn',
                    n_clicks=0),
        html.Button('Save File', id='save-file-btn',
                    n_clicks=0),
        html.Button('Save Embedding Model', id='save-embedding-model-btn',
                    n_clicks=0),
        html.Button('Save Topic Model', id='save-topic-model-btn',
                    n_clicks=0),
        filename_modal,

        html.Div(id='output-container'),
        html.Div(id='output-container-2'),
        html.Div(id='output-container-3'),
        dbc.Modal([
            dbc.ModalHeader([
                dbc.ModalTitle(
                            "Cluster Management"),
            ]),
            dbc.ModalBody([
                              dbc.Button('Add New Cluster', id='add-new-cluster-btn', n_clicks=0),
                              dbc.Button('Recalculate Cluster Info', id='recalculate-cluster-info-btn', n_clicks=0),
                              html.Div(id='topic-list', children=[])
                          ])
            ],
            id="cluster-dialog",
            is_open=False,
            scrollable=True,
            size="xl"
        ),
        html.Hr(),
        html.Div(id='hyperparameters-display'),
        dcc.Loading(
            id="loading-controls-and-graph",
            type="default",
            children=dcc.Graph(figure={}, id='controls-and-graph')),
    ])

    app_layout = [
        dcc.Store(id='snippet-processing-strategy', data=DEFAULT_SNIPPET_PROCESSING_STRATEGY),
        dcc.Store(id='first-run-flag'),
        dcc.Store(id='last-opened-save-modal'),
        dcc.Store(id='current-topic-id', storage_type='local'),
        dcc.ConfirmDialog(
            id='add-new-cluster-confirm-dialog',
        ),
        dcc.ConfirmDialog(
            id='state-confirm-dialog',
        ),
        upload_modal,
        dcc.Loading(
            id="loading",
            type="default",
            children=html.Div(id='selected-file-output')
        ),
        html.Hr(),
        main_layout,
    ]

    app.layout = html.Div(children=app_layout, id='app-layout-div')


if __name__ == '__main__':

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists(f'{OUTPUT_DIR}/{DATA_DIR}'):
        os.makedirs(f'{OUTPUT_DIR}/{DATA_DIR}')
    if not os.path.exists(f'{OUTPUT_DIR}/{EMBEDDING_MODEL_DIR}'):
        os.makedirs(f'{OUTPUT_DIR}/{EMBEDDING_MODEL_DIR}')
    if not os.path.exists(f'{OUTPUT_DIR}/{TOPIC_MODEL_DIR}'):
        os.makedirs(f'{OUTPUT_DIR}/{TOPIC_MODEL_DIR}')

    create_layout([])
    app.run_server(debug=True, host='0.0.0.0', port=9000)