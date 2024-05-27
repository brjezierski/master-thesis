import os
import sys
import time
import random
import base64
import io

from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd
from hyperopt import hp
from tqdm.notebook import tqdm
from flask_caching import Cache

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, ctx, ALL
from dash.dependencies import Input, Output, State

from components import (
    create_filename_modal,
    create_snippets_modal,
    create_upload_modal,
    create_label_selection,
    create_embedding_model_section,
    create_topic_model_section,
    create_parameter_sliders,
    create_parameter_inputs,
    create_other_parameter_inputs,
    create_outlier_checklist,
    create_merge_section,
    create_buttons,
    create_cluster_management_modal
)

from utils import (
    remove_stopwords,
    create_replace_no_tags_embeddings,
    load_data,
    replace_hashtags_in_named_entity_tags,
    str_to_numpy_converter,
    filter_top_values,
    get_classification_counts
)

from MyBERTopic import MyBERTopic
from MyTrainer import MyTrainer
from MySentenceTransformer import MySentenceTransformer
from ParameterOptimizer import ParameterOptimizer

tqdm.pandas()

OUTPUT_DIR = '../data/output'
DATA_DIR = 'data'
EMBEDDING_MODEL_DIR = 'embedding_models'
REPLACE_HASHTAG_IN_NAMED_ENTITIES_DIR = 'replacing_#_in_NE'
REMOVE_NAMED_ENTITY_TAGS = 'removing_NE'

TOPIC_MODEL_DIR = 'topic_models'

DEFAULT_TOPIC_MODEL = ''
DEFAULT_EMBEDDING_MODEL = 'all-MiniLM-L12-v2'
DEFAULT_MIN_CLUSTER_SIZE = 7 #38
DEFAULT_MIN_SAMPLES = 4
DEFAULT_N_COMPONENTS = 2
DEFAULT_N_NEIGHBORS = 4 # 51
DEFAULT_TOP_N_TOPIC_WORDS = 5 # 10

DEFAULT_SNIPPET_PROCESSING_STRATEGY = 'replace_hashtag_in_named_entities' # 'remove_named_entity_tags', 'replace_hashtag_in_named_entities'


class TopicModelingApp:
    '''
    Class for the main application that runs the topic modeling dashboard.
    It contains methods for setting up the directories, creating the layout, and setting up the callbacks.    
    '''

    def __init__(self):
        self.app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
        self.cache = Cache(self.app.server, config={
            'CACHE_TYPE': 'filesystem',
            'CACHE_DIR': '../data/cache-directory',
            'CACHE_DEFAULT_TIMEOUT': 0,
            'CACHE_THRESHOLD': 200
        })
        self.params = {
            "penalty": 'label_count',
            "occurrence_cutoff": 2,
            "snippet_column_name": 'embedding_input',
            "max_evals": 10,
            "bayesian_search_metric": 'topic_diversity+bert',
            "epochs": 1,
            "unfreeze_layers": 2,
            "batch_size": 32,
            "val_dev_size": 10
        }
        self.output_dir = OUTPUT_DIR
        self.setup_directories()
        self.create_layout([])
        self.setup_callbacks()

    def run(self):
        self.app.run_server(debug=True, host='0.0.0.0', port=9000)

    def setup_directories(self):
        '''
        Create the necessary directories for the application models and data.
        '''
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(f'{self.output_dir}/{DATA_DIR}'):
            os.makedirs(f'{self.output_dir}/{DATA_DIR}')
        if not os.path.exists(f'{self.output_dir}/{EMBEDDING_MODEL_DIR}'):
            os.makedirs(f'{self.output_dir}/{EMBEDDING_MODEL_DIR}')
        if not os.path.exists(f'{self.output_dir}/{TOPIC_MODEL_DIR}'):
            os.makedirs(f'{self.output_dir}/{TOPIC_MODEL_DIR}')

    def create_layout(self, classifications: List[str]) -> None:
        '''
        Create the main layout for the Dash application.

        Args:
            classifications (List[str]): List of classification labels.
        '''
        filename_modal = create_filename_modal()
        snippets_modal = create_snippets_modal()
        upload_modal = create_upload_modal()
        label_selection = create_label_selection(classifications)
        embedding_model_section = create_embedding_model_section(self.retrieve_all_model_options, EMBEDDING_MODEL_DIR, DEFAULT_EMBEDDING_MODEL)
        topic_model_section = create_topic_model_section(self.retrieve_all_model_options, TOPIC_MODEL_DIR, DEFAULT_TOPIC_MODEL)
        parameter_sliders = create_parameter_sliders()
        parameter_inputs = create_parameter_inputs(DEFAULT_MIN_CLUSTER_SIZE, DEFAULT_MIN_SAMPLES, DEFAULT_N_COMPONENTS, DEFAULT_N_NEIGHBORS, DEFAULT_TOP_N_TOPIC_WORDS)
        other_parameter_inputs = create_other_parameter_inputs(DEFAULT_TOP_N_TOPIC_WORDS)
        outlier_checklist = create_outlier_checklist()
        merge_section = create_merge_section()
        buttons = create_buttons()
        cluster_management_modal = create_cluster_management_modal()

        main_layout = html.Div([
            snippets_modal,
            label_selection,
            embedding_model_section,
            topic_model_section,
            dcc.Checklist(
                id='param-search-checklist',
                options=[{'label': 'Run parameter search', 'value': 'parameter_search'}],
            ),
            parameter_sliders,
            parameter_inputs,
            other_parameter_inputs,
            outlier_checklist,
            merge_section,
            buttons,
            filename_modal,
            html.Div(id='output-container'),
            html.Div(id='output-container-2'),
            html.Div(id='output-container-3'),
            cluster_management_modal,
            html.Hr(),
            html.Div(id='hyperparameters-display'),
            dcc.Loading(id="loading-controls-and-graph", type="default", children=dcc.Graph(figure={}, id='controls-and-graph')),
        ])

        app_layout = [
            dcc.Store(id='snippet-processing-strategy', data=DEFAULT_SNIPPET_PROCESSING_STRATEGY),
            dcc.Store(id='first-run-flag'),
            dcc.Store(id='last-opened-save-modal'),
            dcc.Store(id='current-topic-id', storage_type='local'),
            dcc.ConfirmDialog(id='add-new-cluster-confirm-dialog'),
            dcc.ConfirmDialog(id='state-confirm-dialog'),
            upload_modal,
            dcc.Loading(id="loading", type="default", children=html.Div(id='selected-file-output')),
            html.Hr(),
            main_layout,
        ]

        self.app.layout = html.Div(children=app_layout, id='app-layout-div')

    def setup_callbacks(self):
        self.app.callback(
            Output('run-btn', 'disabled'),
            Input('label-dropdown', 'value'),
            Input('topic-merge-dropdown', 'value')
        )(self.enable_run_button_callback)

        self.app.callback(
            Output('params-div', 'style'),
            Output('params-range-div', 'style'),
            Input('param-search-checklist', 'value')
        )(self.enable_params_div_callback)

        self.app.callback(
            Output('snippet-processing-strategy', 'data'),
            Input('embedding-model-dropdown', 'value'),
            Input('topic-model-dropdown', 'value'),
            State('snippet-processing-strategy', 'data'),
        )(self.assign_selected_models_callback)

        self.app.callback(
            Output('state-confirm-dialog', 'displayed'),
            Output('state-confirm-dialog', 'message'),
            Output('app-layout-div', 'children'),
            Input('save-state-btn', 'n_clicks'),
            Input('load-state-btn', 'n_clicks'),
            State('app-layout-div', 'children'),
            State('controls-and-graph', 'figure'),
            prevent_initial_call=True
        )(self.save_and_load_state_callback)

        self.app.callback(
            Output('embedding-model-dropdown', 'options'),
            Output('topic-model-dropdown', 'options'),
            Input('embedding-model-dropdown-div', 'n_clicks'),
            Input('topic-model-dropdown-div', 'n_clicks'),
        )(self.fill_topic_and_embedding_model_options_callback)

        self.app.callback(
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
        )(self.run_modeling_callback)

        self.app.callback(
            Output('cluster-dialog', 'is_open'),
            Output('topic-list', 'children'),
            Output('add-new-cluster-confirm-dialog', 'displayed'),
            Output('add-new-cluster-confirm-dialog', 'message'),
            Input('cluster-management-btn', 'n_clicks'),
            Input("snippets-modal", "is_open"),
            Input('add-new-cluster-btn', 'n_clicks'),
            Input('snippets-modal-add-new-cluster-btn', 'n_clicks'),
            Input('recalculate-cluster-info-btn', 'n_clicks'),
            prevent_initial_call=True
        )(self.show_cluster_dialog_callback)

        self.app.callback(
            Output("dropdown-assign", "style"),
            Input({"type": "topic-name", "index": ALL}, "value"),
            State({"type": "topic-name", "index": ALL}, "id"),
        )(self.update_cluster_names_callback)

        self.app.callback(
            Output("output-container", "style"),
            Input("snippets-checklist", "value"),
            prevent_initial_call=True,
        )(self.handle_selected_snippets_callback)

        self.app.callback(
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
        )(self.show_cluster_dialog_content_callback)

        self.app.callback(
            Output("snippets-checklist", "value"),
            Output("snippets-checklist", "options"),
            Input("btn-assign", "n_clicks"),
            State("snippets-checklist", "value"),
            State("dropdown-assign", "value"),
            State("current-topic-id", 'data'),
            prevent_initial_call=True,
        )(self.handle_assign_snippets_callback)

        self.app.callback(
            Output('upload-modal', 'is_open'),
            Input('upload-btn', 'n_clicks'),
            State('upload-modal', 'is_open'),
        )(self.toggle_modal_callback)

        self.app.callback(
            Output('selected-file-output', 'children'),
            Output('label-dropdown', 'options'),
            Output('embedding-model-dropdown-div', 'style'),
            State('upload-data', 'filename'),
            Input('upload-data', 'contents'),
            prevent_initial_call=True,
        )(self.handle_file_selection_callback)

        self.app.callback(
            Output('topic-model-training-checklist', 'value'),
            Input('param-search-checklist', 'value'),
            Input('topic-model-dropdown', 'value'),
            prevent_initial_call=True
        )(self.update_topic_model_training_callback)

        self.app.callback(
            Output('label-dropdown', 'value'),
            Input('select-all-labels-btn', 'n_clicks'),
            State('label-dropdown', 'options')
        )(self.select_all_classification_labels_callback)

        self.app.callback(
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
        )(self.handle_saving_modal_callback)

    def enable_run_button_callback(self, selected_labels: List[str], selected_clusters_to_merge: List[int]) -> bool:
        '''
        Enable the "Run" button when classification labels are selected.
        If any clusters are selected to merge, ensure it is at least two.
        
        Args:
        - selected_labels (List[str]): Selected classification labels (Input).
        - selected_clusters_to_merge (List[int]): Selected clusters to merge (Input).

        Returns:
        - bool: True if the 'Run' button should be enabled, False otherwise.
        '''
        if not selected_clusters_to_merge:
            return not bool(selected_labels)
        else:
            if len(selected_clusters_to_merge) >= 2:
                return False
            print("You need to select at least two clusters to merge")
            return True

    def enable_params_div_callback(self, param_search_value: bool) -> Tuple[Dict[str, str], Dict[str, str]]:
        '''
        Enable or disable parameter search div based on whether the Bayesian search has been selected.
        
        Parameters:
        - param_search_value: bool - Value indicating if the Bayesian search was checked (Input).

        Returns:
        - Tuple[Dict[str, str], Dict[str, str]]: Styles for 'params-div' and 'params-range-div'.
        '''
        if not param_search_value or 'parameter_search' not in param_search_value:
            return {'display': 'block'}, {'display': 'none'}
        return {'display': 'none'}, {'display': 'block'}

    def assign_selected_models_callback(self, embedding_model_value: str, topic_model_value: str, snippet_processing_strategy: str) -> str:
        '''
        Assigns the selected embedding and topic models to the cache and updates the snippet processing strategy.

        Parameters:
        - embedding_model_value: str - Selected embedding model from the dropdown (Input).
        - topic_model_value: str - Selected topic model from the dropdown (Input).
        - snippet_processing_strategy: str - Current snippet processing strategy (State).

        Returns:
        - str: Updated snippet processing strategy.
        '''
        if type(embedding_model_value) == list:
            embedding_model_value = embedding_model_value[0]
        if type(topic_model_value) == list:
            topic_model_name = topic_model_value[0]
        else:
            topic_model_name = topic_model_value
        self.cache.set('topic_model_name', topic_model_name)

        # Read from huggingface
        if not embedding_model_value or DEFAULT_EMBEDDING_MODEL in embedding_model_value:
            embedding_model = MySentenceTransformer(DEFAULT_EMBEDDING_MODEL)
        # Read locally with a path
        else:
            if REPLACE_HASHTAG_IN_NAMED_ENTITIES_DIR in embedding_model_value:
                snippet_processing_strategy = 'replace_hashtag_in_named_entities'
            elif REMOVE_NAMED_ENTITY_TAGS in embedding_model_value:
                snippet_processing_strategy = 'remove_named_entity_tags'
            embedding_model = MySentenceTransformer(
                    f'{self.output_dir}/{EMBEDDING_MODEL_DIR}/{embedding_model_value}')
        self.cache.set('embedding_model', embedding_model)
        return snippet_processing_strategy

    def save_and_load_state_callback(self, save_state_n_clicks: int, load_state_n_clicks: int, app_layout: List, fig: Dict):
        '''
        Handles saving and loading the app state to and from cache.

        Parameters:
        - save_state_n_clicks: int - Number of clicks on the 'Save State' button (Input).
        - load_state_n_clicks: int - Number of clicks on the 'Load State' button (Input).
        - app_layout: list - Layout of the application (State).
        - fig: dict - Current state of the figure (State).

        Returns:
        - Tuple[bool, str, list]: A tuple containing:
            - A boolean indicating if the state was saved.
            - A string message for the confirmation dialog.
            - The updated application layout.
        '''
        if ctx.triggered_id == 'save-state-btn':
            self.save_state(app_layout, fig)
            return True, 'State saved.', app_layout
        elif ctx.triggered_id == 'load-state-btn':
            return False, '', self.cache.get('app_layout')
        raise dash.exceptions.PreventUpdate

    def fill_topic_and_embedding_model_options_callback(self, embedding_model_n_clicks: int, topic_model_n_clicks: int) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        '''
        Fills the training dropdowns with the embedding and topic model options.

        Parameters:
        - embedding_model_n_clicks: int - Number of clicks on the embedding model div (Input).
        - topic_model_n_clicks: int - Number of clicks on the topic model div (Input).

        Returns:
        - Tuple[List[Dict[str, str]], List[Dict[str, str]]]: Options for embedding and topic model dropdowns.
        '''
        embedding_options = self.retrieve_all_model_options(
            EMBEDDING_MODEL_DIR)
        topic_model_options = self.retrieve_all_model_options(
            TOPIC_MODEL_DIR)
        return embedding_options, topic_model_options

    def run_modeling_callback(self, n_clicks: int, is_cluster_dialog_open: bool, selected_labels: Sequence[str], 
                            classification_labels: Sequence[str], embedding_training_checklist: Sequence[str], 
                            topic_model_training_checklist: Sequence[str], reduce_outliers: bool, param_search: bool,
                            min_cluster_size_value: int, min_samples_value: int, n_components_value: int, 
                            n_neighbors_value: int, top_n_topic_words_value: int, min_cluster_size_range: Sequence[int], 
                            min_samples_range: Sequence[int], n_components_range: Sequence[int], n_neighbors_range: Sequence[int], 
                            bayesian_iterations: int, hyperparam_string: str, fig: Dict, first_run_flag: bool, 
                            snippet_processing_strategy: str, app_layout: List) -> Tuple[dict, list, bool]:
        '''
        Manages the workflow involved in initializing and retrieving cached models and data, 
        updating the figure when the label dropdown changes, 
        and reflecting topic assignment changes when the cluster dialog is closed. 
        When the "Run" button is clicked, it prepares data, performs a Bayesian search for hyperparameters if needed, 
        and trains the topic model on the first run or reduces outliers if specified. 
        It then updates the document embeddings visualization, updates the topics, and saves the application state, 
        returning the updated figure, hyperparameters display, and first run flag. 
        If none of these actions are triggered, the function prevents updates to the app.

        Parameters:
        - n_clicks (int): Number of clicks on the run button (input).
        - is_cluster_dialog_open (bool): State of the cluster dialog (input).
        - selected_labels (Sequence[str]): Labels selected by the user (input).
        - classification_labels (Sequence[str]): All available classification labels (state).
        - embedding_training_checklist (Sequence[str]): Embedding training options selected by the user (state).
        - topic_model_training_checklist (Sequence[str]): Topic model training options selected by the user (state).
        - reduce_outliers (bool): Whether to reduce outliers in the model (state).
        - param_search (bool): Whether to perform parameter search (state).
        - min_cluster_size_value (int): Minimum cluster size value for HDBSCAN (state).
        - min_samples_value (int): Minimum samples value for HDBSCAN (state).
        - n_components_value (int): Number of components for dimensionality reduction (state).
        - n_neighbors_value (int): Number of neighbors for dimensionality reduction (state).
        - top_n_topic_words_value (int): Number of top words for each topic (state).
        - min_cluster_size_range (Sequence[int]): Range of minimum cluster size for Bayesian search (state).
        - min_samples_range (Sequence[int]): Range of minimum samples for Bayesian search (state).
        - n_components_range (Sequence[int]): Range of components for Bayesian search (state).
        - n_neighbors_range (Sequence[int]): Range of n neighbors for Bayesian search (state).
        - bayesian_iterations (int): Number of iterations for Bayesian search (state).
        - hyperparam_string (str): Hyperparameters display string (state).
        - fig (dict): Current state of the figure (state).
        - first_run_flag (bool): Flag indicating whether it is the first run (state).
        - snippet_processing_strategy (str): Strategy for processing snippets (state).
        - app_layout (list): Layout of the application (state).

        Returns:
        - Tuple[dict, list, bool]: Updated figure, hyperparameters display list, and first run flag.
        '''
        random.seed(42)
        embedding_model = self.cache.get('embedding_model')
        try:
            topic_model = self.cache.get('topic_model')
        except:
            topic_model = None
        topic_model_name = self.cache.get('topic_model_name')
        df = self.cache.get('df')


        if ctx.triggered_id == 'label-dropdown':
            return fig, hyperparam_string, True

        classifications = {classification_label['value']: i for i, classification_label in enumerate(classification_labels)}

        is_topic_model_trained = 'topic_model_training' in topic_model_training_checklist or topic_model_name == "DEFAULT_TOPIC_MODEL"

        # On closing the cluster dialog, update the figure since the topic assignment might have changed
        if ctx.triggered_id == 'cluster-dialog' and not is_cluster_dialog_open:
            fig = topic_model.reduce_dim_and_visualize_documents(reuse_reduced_embeddings=False) # reuse reduced_embeddings to not have to recompute them and save time 
            self.save_state(app_layout, fig)
            return fig, hyperparam_string, first_run_flag

        if ctx.triggered_id == 'run-btn':
            if 'top_classification' not in df.columns:
                df['top_classification'] = df.apply(
                    lambda row: filter_top_values(row, selected_labels), axis=1)
                self.cache.set('df', df)

            if first_run_flag or param_search:
                self.params['inference_labels'] = selected_labels
                df_training, df_inference = self.split_and_encode_data(selected_labels, classifications, snippet_processing_strategy,
                        train_embeddings='embedding_training' in embedding_training_checklist)
            if param_search:
                self.params["max_evals"] = bayesian_iterations
                self.params = {**self.params, **{'topic_model_name': topic_model_name, 'embedding_model': embedding_model, 'top_n_topic_words': top_n_topic_words_value, 'is_topic_model_trained': is_topic_model_trained}}
                hdbscan_umap_params, bayesian_search_results = self.run_bayesian_search(
                        df_training,
                        df_inference,
                        is_topic_model_trained, 
                        min_cluster_size_range, 
                        min_samples_range, 
                        n_components_range, 
                        n_neighbors_range
                    )

                hyperparameters_display_list = [html.Div("Trials:"), html.Br()]
                for epoch, result in enumerate(bayesian_search_results):
                    space, loss, label_count = result
                    hyperparameters_display_list += [html.Div(f"Epoch {epoch+1}: loss {loss:.2f}, label count {label_count} for {ParameterOptimizer.get_hyperparameter_string(space)}"), html.Br()]
                hyperparameters_display_list += [html.Div(f"Best hyperparameters {ParameterOptimizer.get_hyperparameter_string(hdbscan_umap_params)}"), html.Br()]
            else:
                hdbscan_umap_params = {
                    "min_cluster_size": min_cluster_size_value,
                    "min_samples": min_samples_value,
                    "n_components": n_components_value,
                    "n_neighbors": n_neighbors_value,
                    "top_n_topic_words": top_n_topic_words_value
                    }
                hyperparameters_display_list = [
                    html.Div(f"Hyperparameters: {ParameterOptimizer.get_hyperparameter_string(hdbscan_umap_params)}"), 
                    html.Br()
                    ]

            if first_run_flag:
                self.params = {**self.params, **hdbscan_umap_params}
                topic_model = MyBERTopic.create_and_train(
                    self.params, 
                    embedding_model, 
                    topic_model_name, 
                    df_training, 
                    df_inference, 
                    reduce_outliers, 
                    is_topic_model_trained, 
                    DEFAULT_TOPIC_MODEL, 
                    f'{self.output_dir}/{TOPIC_MODEL_DIR}/{topic_model_name}'
                    )
                first_run_flag = False
            elif reduce_outliers:
                topic_model.reduce_outliers_and_update()
                topic_model.remove_topic_ids_from_topic_labels()
                topic_model.update_topics()

            fig = topic_model.reduce_dim_and_visualize_documents()
            topic_model.update_topics()
            self.save_state(app_layout, fig)
            self.cache.set('topic_model', topic_model)
            return fig, hyperparameters_display_list, first_run_flag
        else:
            raise dash.exceptions.PreventUpdate

    def show_cluster_dialog_callback(self, cluster_management_n_clicks: int, is_open: bool, 
        add_new_cluster_n_clicks: int, snippets_modal_add_new_cluster_n_clicks: int, 
        recalculate_cluster_info_n_clicks: int) -> Tuple[bool, List[html.Div], bool, str]:
        '''
        Manages the cluster dialog display and updates based on user interactions.
        It updates the topic sizes, adds a new cluster, recalculates the cluster info, and creates a list of topic details.

        Parameters:
        - cluster_management_n_clicks: int - Number of clicks on the cluster management button (Input).
        - is_open: bool - Indicates if the snippets modal is open (Input).
        - add_new_cluster_n_clicks: int - Number of clicks on the add new cluster button (Input).
        - snippets_modal_add_new_cluster_n_clicks: int - Number of clicks on the add new cluster button within the snippets modal (Input).
        - recalculate_cluster_info_n_clicks: int - Number of clicks on the recalculate cluster info button (Input).

        Returns:
        - Tuple[bool, List[html.Div], bool, str]: A tuple containing:
            - A boolean indicating if the cluster dialog should be open.
            - A list of HTML Div elements representing the topic details.
            - A boolean indicating if the confirmation dialog should be displayed.
            - A string message for the confirmation dialog.
        '''
        topic_model = self.cache.get('topic_model')

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

        for i, row in topic_info_df.iterrows():
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

        self.cache.set('topic_model', topic_model)
        return True, topic_details, is_confirmation_displayed, f'New topic {new_topic_label} was just added.'

    def update_cluster_names_callback(self, topic_names: Sequence[str], topic_ids: Sequence[Dict]) -> Dict[str, str]:
        '''
        Updates the stored cluster names when any cluster name is changed.

        Parameters:
        - topic_names: Sequence[str] - Sequence of topic names entered by the user (Input).
        - topic_ids: Sequence[Dict] - Sequence of topic IDs (State).

        Returns:
        - dict[str, str]: Dummy output containing a dictionary with the style information for the dropdown.
        '''
        try:
            topic_model = self.cache.get('topic_model')
        except:
            topic_model = None

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

        self.cache.set('topic_model', topic_model)
        return {'display': 'block'}

    def handle_selected_snippets_callback(self, selected_snippets: Sequence[str]) -> Dict[str, str]:
        '''
        Updates the cache when snippets are selected in a cluster dialog.

        Parameters:
        - selected_snippets: dict - A dictionary containing selected snippets (Input).

        Returns:
        - dict[str, str]: Dummy output containing a dictionary with the style information for the dropdown.
        '''
        self.cache.set("selected_snippets", selected_snippets)
        return {'display': 'block'}

    def show_cluster_dialog_content_callback(self, btn_show_clicks: Sequence[int], search_term: str, is_open: bool, btn_remove_clicks: int,
                                    btn_select_all_clicks: int, current_topic_id: int) -> Tuple[bool, dbc.Checklist, List[dict[str, str]], str, int]:
        '''
        Shows the cluster dialog content based on user interactions.

        Parameters:
        - btn_show_clicks: Sequence[int] - Sequence of clicks on the show snippets button (Input).
        - search_term: str - Search term entered by the user (Input).
        - is_open: bool - Indicates if the snippets modal is open (Input).
        - btn_remove_clicks: int - Number of clicks on the remove cluster button (Input).
        - btn_select_all_clicks: int - Number of clicks on the select all button (Input).
        - current_topic_id: int - Current topic ID (State).

        Returns:
        - Tuple[bool, dbc.Checklist, List[dict[str, str]], str, int]: A tuple containing:
            - A boolean indicating if the cluster dialog should be open.
            - A checklist of snippets.
            - A list of cluster available in the assignment dropdown.
            - A string indicating the title of the snippets modal.
            - An integer indicating the current topic ID.
        '''
        topic_model = self.cache.get('topic_model')
        selected_snippets = self.cache.get("selected_snippets")

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

        self.cache.set('topic_model', topic_model)

        return is_open, checklist, [{'label': cluster_name, 'value': cluster_id} for cluster_id, cluster_name in topic_labels_excluding_current.items()], f"Snippets for Cluster {current_topic_id}" if current_topic_id != -1 else "Outliers", current_topic_id

    def handle_assign_snippets_callback(self, assign_clicks: int, selected_snippets: Sequence[str], selected_topic: str, current_topic_id: int) -> Tuple[Sequence[str], Sequence[Dict[str, int]]]:
        '''
        Handles moving snippets to a different cluster.

        Parameters:
        - assign_clicks: int - Number of clicks on the assign button (Input).
        - selected_snippets: Sequence[str] - Sequence of selected snippets (Input).
        - selected_topic: str - Selected topic (Input).
        - current_topic_id: int - Current topic ID (State).

        Returns:
        - Tuple[Sequence[str], Sequence[Dict[str, int]]]: A tuple containing:
            - A sequence of snippets assigned to the cluster.
            - A sequence of snippets together with their numerical id.
        '''
        topic_model = self.cache.get('topic_model')

        if not assign_clicks or not selected_snippets:
            raise dash.exceptions.PreventUpdate

        # Move the selected snippets to the target cluster
        for snippet_ind in selected_snippets:
            topic_model.topics_[snippet_ind] = selected_topic

        topic_model.recalculate_topic_sizes()
        
        self.cache.set('topic_model', topic_model)

        if current_topic_id > -2:
            return [], topic_model.get_documents_from_topic(current_topic_id)
        else:
            raise dash.exceptions.PreventUpdate

    def toggle_modal_callback(self, upload_clicks: int, is_open: bool) -> bool:
        '''
        Toggles the upload modal.

        Parameters:
        - upload_clicks: int - Number of clicks on the upload button (Input).
        - is_open: bool - Indicates if the modal is open (Input).

        Returns:
        - bool: A boolean indicating if the modal should be open.
        '''
        if upload_clicks:
            return not is_open
        return is_open

    def handle_file_selection_callback(self, filename: str, contents: str) -> Tuple[str, List[Dict[str, str]], Dict[str, str]]:
        '''
        Loads one of the data formats: csv, tsv or json as a dataframe with snippets 
        and extracts the classification labels to populate the dropdown.

        Parameters:
        - filename: str - Name of the file selected (Input).
        - contents: str - Contents of the file selected (Input).

        Returns:
        - Tuple[str, List[Dict[str, str]], Dict[str, str]]: A tuple containing:
            - A string indicating the selected file or an error.
            - A list of classification labels.
            - A dictionary containing the style information for the embedding model dropdown.
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

            # Any empty value for classification is replaced with string 'NONE'
            df[f'classification'] = df['classification'].apply(lambda x: 'NONE' if pd.isnull(x) else x)

            # remove all rows whose snippet or replace is not string
            df = df[df['snippet'].apply(lambda x: isinstance(x, str))]
            df = df[df['replace'].apply(lambda x: isinstance(x, str))]
            # drop all rows with empty snippets or replace
            df = df[df['snippet'].apply(lambda x: len(x) > 0)]
            df = df[df['replace'].apply(lambda x: len(x) > 0)]
            classifications = get_classification_counts(df)
            classifications_sorted_by_count = [classification for classification, count in sorted(classifications.items(),
                                                                                                key=lambda x: x[1], reverse=True)]
            self.cache.set('df', df)

            if 'embedding' in df.columns:
                return f"Selected file: {filename} with embeddings", [{'label': label, 'value': label}
                                                                    for label in classifications_sorted_by_count], {'display': 'none'}
            else:
                return f"Selected file: {filename}", [{'label': label, 'value': label}
                                                    for label in classifications_sorted_by_count], {'display': 'block'}
        except Exception as e:
            print(e)
            return f'There was an error processing this file.\n{e}', [], {'display': 'block'}

    def update_topic_model_training_callback(self, selected_options: Sequence[str], topic_model_value: str) -> Sequence[str]:
        '''
        Updates the topic model training checklist based on the selected options.
        By default the the model training is on as topic_model_training is selected.

        Parameters:
        - selected_options: Sequence[str] - Selected options (Input).
        - topic_model_value: str - Selected topic model (Input).

        Returns:
        - Sequence[str]: A sequence of selected options.
        '''
        if selected_options or topic_model_value == DEFAULT_TOPIC_MODEL:
            return ['topic_model_training']
        else:
            raise dash.exceptions.PreventUpdate
    
    def select_all_classification_labels_callback(self, n_clicks: int, options: List[Dict[str, str]]) -> List[str]:
        '''
        Selects all labels in the classification label dropdown.

        Parameters:
        - n_clicks: int - Number of clicks on the select all labels button (Input).
        - options: List[Dict[str, str]] - List of classification labels (State).

        Returns:
        - List[str]: A list of selected labels.
        '''
        if n_clicks:
            return [option['value'] for option in options]
        return []

    def handle_saving_modal_callback(self, save_file_clicks: int, save_embedding_model_clicks: int, save_topic_model_clicks: int,
                                    confirm_clicks: int, cancel_clicks: int, filename: str, embeddings_checklist_value: List[str],
                                    last_opened_save_modal: str, snippet_processing_strategy: str) -> Tuple[Dict[str, str], Dict[str, str], str]:
        '''
        Handles the saving modal for the embeddings and topic model.

        Parameters:
        - save_file_clicks: int - Number of clicks on the save file button (Input).
        - save_embedding_model_clicks: int - Number of clicks on the save embedding model button (Input).
        - save_topic_model_clicks: int - Number of clicks on the save topic model button (Input).
        - confirm_clicks: int - Number of clicks on the confirm button (Input).
        - cancel_clicks: int - Number of clicks on the cancel button (Input).
        - filename: str - Name of the file to save (State).
        - embeddings_checklist_value: List[str] - Selected options for saving embeddings (State).
        - last_opened_save_modal: str - Last opened save modal (State).
        - snippet_processing_strategy: str - Strategy for processing snippets (State).

        Returns:
        - Tuple[Dict[str, str], Dict[str, str], str]: A tuple containing:
            - A dictionary containing the style information for the filename modal.
            - A dictionary containing the style information for the embeddings checklist.
            - A string indicating the last opened save modal (between the topic and embedding one).
        '''
        embedding_checklist_style = {'display': 'block'} if last_opened_save_modal == 'save-file-btn' else {
            'display': 'none'}
        ctx = dash.callback_context

        print('save_file_clicks', save_file_clicks)
        print('save_embedding_model_clicks', save_embedding_model_clicks)
        print('save_topic_model_clicks', save_topic_model_clicks)
        print('confirm_clicks', confirm_clicks)
        print('cancel_clicks', cancel_clicks)
        print('filename', filename)
        print('embeddings_checklist_value', embeddings_checklist_value)
        print('last_opened_save_modal', last_opened_save_modal)
        print('snippet_processing_strategy', snippet_processing_strategy)


        if ctx.triggered:
            triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if triggered_id.startswith('save'):
                last_opened_save_modal = triggered_id
                embedding_checklist_style = {'display': 'block'} if last_opened_save_modal == 'save-file-btn' else {
                    'display': 'none'}
                return {'display': 'block'}, embedding_checklist_style, last_opened_save_modal
            if triggered_id == 'filename-confirm':
                if filename:
                    self.save_output(
                        filename, 'embeddings' in embeddings_checklist_value, last_opened_save_modal, snippet_processing_strategy)
                    return {'display': 'none'}, {'display': 'none'}, last_opened_save_modal
                else:
                    return {'display': 'block'}, embedding_checklist_style, last_opened_save_modal
            if triggered_id == 'filename-cancel':
                return {'display': 'none'}, {'display': 'none'}, last_opened_save_modal

        return {'display': 'none'}, {'display': 'none'}, last_opened_save_modal


    def split_and_encode_data(self, selected_labels: Sequence[str], classifications: Dict[str, int],
                            snippet_processing_strategy: str, train_embeddings: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        '''
        Modify the snipptes according to the selected processing strategy, 
        train the sentence embedding model if needed, encode the snippets, 
        and split into inference and training sets

        Parameters:
        - selected_labels (Sequence[str]): Selected classification labels.
        - classifications (Dict[str, int]): Dictionary of classification labels to their index.
        - snippet_processing_strategy (str): Strategy for processing snippets (remove_named_entity_tags or replace_hashtag_in_named_entities).
        - train_embeddings (bool): Whether to train the embeddings.

        Returns:
        - Tuple[pd.DataFrame, pd.DataFrame]: Dataframes for training and inference.
        '''
        df = self.cache.get('df')
        embedding_model = self.cache.get('embedding_model')

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
                trainer = MyTrainer([df_training], [classifications], embedding_model, self.params["epochs"], self.params["batch_size"], unfreeze_layers = self.params["unfreeze_layers"], snippet_column_name=self.params['snippet_column_name'] if 'snippet_column_name' in self.params else 'snippet')
                embedding_model = trainer.fit()
                self.cache.set('embedding_model', embedding_model)
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
        print('Sample of the snippet encoding format', df_training["embedding_input"].to_list()[:2])
        self.cache.set('df', df)
        return df_training, df_inference

    def run_bayesian_search(self, df_training: pd.DataFrame, df_inference: pd.DataFrame, is_topic_model_trained: bool, 
                            min_cluster_size_range: Sequence[int], min_samples_range: Sequence[int], n_components_range: Sequence[int],
                            n_neighbors_range: Sequence[int]) -> Tuple[Dict[str, Any], List[Tuple[Dict[str, Any], float, int]]]:
        '''
        Run the Bayesian search to obtain best hyperparameters.

        Parameters:
        - df_training (pd.DataFrame): Dataframe for training.
        - df_inference (pd.DataFrame): Dataframe for inference.
        - is_topic_model_trained (bool): Whether the topic model is trained.
        - min_cluster_size_range (Sequence[int]): Range of minimum cluster size for Bayesian search.
        - min_samples_range (Sequence[int]): Range of minimum samples for Bayesian search.
        - n_components_range (Sequence[int]): Range of components for Bayesian search.
        - n_neighbors_range (Sequence[int]): Range of n neighbors for Bayesian search.

        Returns:
        - Tuple[Dict[str, Any], List[Tuple[Dict[str, Any], float, int]]]: A tuple containing:
            - A dictionary of the best hyperparameters.
            - A list of tuples containing the hyperparameters, loss, and label count.
        '''
        hspace = {
            "n_neighbors": hp.choice('n_neighbors', n_neighbors_range),
            "n_components": hp.choice('n_components', n_components_range),
            "min_cluster_size": hp.choice('min_cluster_size', min_cluster_size_range),
            "min_samples": hp.choice('min_samples', min_samples_range),
            "is_topic_model_trained": is_topic_model_trained,
            "random_state": 42
        }
        label_lower = 28
        label_upper = 50
        parameter_optimizer = ParameterOptimizer(df_training, df_inference)
        best_params, best_clusters, trials = parameter_optimizer.bayesian_search(space=hspace,
                                                                                label_lower=label_lower,
                                                                                label_upper=label_upper,
                                                                                params=self.params)
        return best_params, parameter_optimizer.bayesian_search_results

    def retrieve_all_model_options(self, dir: str) -> List[Dict[str, str]]:
        '''
        Retrieve all the models in the specified directory

        Parameters:
        - dir (str): Directory to retrieve models from.

        Returns:
        - List[Dict[str, str]]: List of dictionaries containing the model options.
        '''
        if dir == EMBEDDING_MODEL_DIR:
            subdir_list = [subdir for subdir in os.listdir(
                f'{self.output_dir}/{dir}/') if "." not in subdir]
            file_list = []
            for subdir in subdir_list:
                file_list += [f'{subdir}/{file}' for file in os.listdir(
                    f'{self.output_dir}/{dir}/{subdir}') if not file.startswith(".")]

            pretrained_embedding_models = [
                {'label': f'{REPLACE_HASHTAG_IN_NAMED_ENTITIES_DIR}/{DEFAULT_EMBEDDING_MODEL}',
                    'value': f'{REPLACE_HASHTAG_IN_NAMED_ENTITIES_DIR}/{DEFAULT_EMBEDDING_MODEL}'},
                {'label': f'{REMOVE_NAMED_ENTITY_TAGS}/{DEFAULT_EMBEDDING_MODEL}', 'value': f'{REMOVE_NAMED_ENTITY_TAGS}/{DEFAULT_EMBEDDING_MODEL}'}]

            return [{'label': file, 'value': file} for file in file_list] + pretrained_embedding_models
        elif dir == TOPIC_MODEL_DIR:
            file_list = [file for file in os.listdir(
                f'{self.output_dir}/{dir}') if not file.startswith(".")]
            pretrained_topic_models = [
                {'label': DEFAULT_TOPIC_MODEL, 'value': DEFAULT_TOPIC_MODEL}]
            return [{'label': file, 'value': file} for file in file_list] + pretrained_topic_models
        else:
            return []

    def save_state(self, app_layout: List, figure: Dict) -> None:
        '''
        Save the current state of the app to the cache. 

        Parameters:
        - app_layout (list): Layout of the application.
        - figure (dict): Current state of the figure.
        '''
        def set_figure_at_id(d: Dict, target_id: str, fig: Dict) -> None:
            '''
            Recursively set the 'figure' property at the specified 'id' in a nested dictionary.

            Parameters:
            - d: The input dictionary
            - target_id: The 'id' at which to set the 'figure'
            - fig: The value to set for the 'figure' property
            '''
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
        self.cache.set('app_layout', app_layout)
        print("State saved.")

    def save_output(self, filename, is_embedding_included: bool, last_opened_save_modal: str, snippet_processing_strategy: str) -> None:
        '''
        Save the model to a file for the topic model, embedding model, 
        or save the snippets with new classifications to a file.

        Parameters:
        - filename (str): Name of the file to save.
        - is_embedding_included (bool): Whether the embedding should be included.
        - last_opened_save_modal (str): Last opened save modal (can be save-file-btn, save-embedding-model-btn, or save-topic-model-btn).        
        '''
        topic_model = self.cache.get('topic_model')
        embedding_model = self.cache.get('embedding_model')

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
            entire_df.to_csv(f'{self.output_dir}/{DATA_DIR}/{filename}.tsv', sep='\t', index=False)

        # Save embedding model
        elif last_opened_save_modal == 'save-embedding-model-btn':
            if snippet_processing_strategy == 'replace_hashtag_in_named_entities':
                filename = f'{REPLACE_HASHTAG_IN_NAMED_ENTITIES_DIR}/{filename}'
            elif snippet_processing_strategy == 'remove_named_entity_tags':
                filename = f'{REMOVE_NAMED_ENTITY_TAGS}/{filename}'
            embedding_model.save(
                f'{self.output_dir}/{EMBEDDING_MODEL_DIR}/{filename}')

        # Save topic model
        elif last_opened_save_modal == 'save-topic-model-btn':
            topic_model.save(f'{self.output_dir}/{TOPIC_MODEL_DIR}/{filename}')
        self.cache.set('topic_model', topic_model)


if __name__ == '__main__':

    app = TopicModelingApp()
    app.run()

