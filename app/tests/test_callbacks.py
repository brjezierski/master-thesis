from flask_caching import Cache
import pytest
from dash.exceptions import PreventUpdate
import os
import sys
import shutil
import base64
from dash._callback_context import context_value
from dash._utils import AttributeDict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from app import TopicModelingApp, OUTPUT_DIR, DATA_DIR, EMBEDDING_MODEL_DIR, TOPIC_MODEL_DIR, DEFAULT_EMBEDDING_MODEL, DEFAULT_TOPIC_MODEL, DEFAULT_SNIPPET_PROCESSING_STRATEGY, DEFAULT_MIN_CLUSTER_SIZE, DEFAULT_MIN_SAMPLES, DEFAULT_N_COMPONENTS, DEFAULT_N_NEIGHBORS, DEFAULT_TOP_N_TOPIC_WORDS

TEST_FILE_PATH = "/Users/bartek/Desktop/master-thesis/app/data/input/generated_file.tsv"
EMBEDDING_MODEL_FILENAME = 'test_embedding_model'

@pytest.fixture
def app_instance():
    '''
    Instantiate the TopicModelingApp class before each test
    '''
    app = TopicModelingApp()
    app.cache = Cache(app.app.server, config={
            'CACHE_TYPE': 'filesystem',
            'CACHE_DIR': 'tests/cache-directory',
            'CACHE_DEFAULT_TIMEOUT': 0,
            'CACHE_THRESHOLD': 200
    })
    app.output_dir = app.output_dir.replace('../', '')

    return app

def test_setup_directories(app_instance):
    '''
    Initialize the app and set up directories.
    Check if directories were created.
    '''
    app_instance.setup_directories()

    assert os.path.exists(OUTPUT_DIR)
    assert os.path.exists(f'{OUTPUT_DIR}/{DATA_DIR}')
    assert os.path.exists(f'{OUTPUT_DIR}/{EMBEDDING_MODEL_DIR}')
    assert os.path.exists(f'{OUTPUT_DIR}/{TOPIC_MODEL_DIR}')

def test_run_button(app_instance):
    '''
    Check if the run button is disabled at start.
    '''
    app_instance.setup_callbacks()
    callbacks = app_instance.app.callback_map
    assert 'run-btn.disabled' in callbacks

def test_handle_file_selection_callback_no_file_selected(app_instance):
    '''
    Test when no file is selected.
    '''
    result = app_instance.handle_file_selection_callback(None, None)
    assert result[0] == "No file selected."
    assert result[1] == []
    assert result[2] == {'display': 'block'}

def test_handle_file_selection_callback_tsv_file_selected(app_instance):
    '''
    Test when a TSV file is selected
    '''
    with open(TEST_FILE_PATH, 'r', encoding='utf-8') as file:
        file_contents = file.read()
    
    encoded_contents = f'data:text/tab-separated-values;base64,{base64.b64encode(file_contents.encode("utf-8")).decode("utf-8")}'

    result = app_instance.handle_file_selection_callback(TEST_FILE_PATH, encoded_contents)
    print(result[0])
    assert result[0].endswith("generated_file.tsv")
    assert len(result[1]) > 0
    assert 'display' in result[2]

def test_run_modeling_callback(app_instance):
    '''
    Test the callback function that runs the modeling process.
    The callback function should return a figure with data.
    '''
    app_instance.create_layout([])

    # Simulate inputs
    n_clicks = 1
    is_cluster_dialog_open = False
    selected_labels = ['ENTITY', 'OTHER']
    classification_labels = [{'label': 'ENTITY', 'value': 'ENTITY'}, {'label': 'OTHER', 'value': 'OTHER'}, {'label': 'PARTNERSHIP', 'value': 'PARTNERSHIP'}, {'label': 'STAFFING', 'value': 'STAFFING'}, {'label': 'FINANCIALS', 'value': 'FINANCIALS'}, {'label': 'INNOVATION', 'value': 'INNOVATION'}, {'label': 'REGULATORY', 'value': 'REGULATORY'}, {'label': 'CSR', 'value': 'CSR'}, {'label': 'DECLINE', 'value': 'DECLINE'}, {'label': 'TECHNOLOGY', 'value': 'TECHNOLOGY'}, {'label': 'MARKET_MOVE', 'value': 'MARKET_MOVE'}, {'label': 'ACQUISITION', 'value': 'ACQUISITION'}, {'label': 'GROWTH', 'value': 'GROWTH'}, {'label': 'PARTNER', 'value': 'PARTNER'}, {'label': 'LEGAL', 'value': 'LEGAL'}, {'label': 'PRODUCT_LAUNCH', 'value': 'PRODUCT_LAUNCH'}, {'label': 'STRATEGY', 'value': 'STRATEGY'}, {'label': 'AWARDS', 'value': 'AWARDS'}, {'label': 'GROW', 'value': 'GROW'}]
    embedding_training_checklist = []
    topic_model_training_checklist = ['topic_model_training']
    reduce_outliers = False
    param_search = False
    min_cluster_size_range = []
    min_samples_range = []
    n_components_range = []
    n_neighbors_range = []
    bayesian_iterations = 5
    hyperparam_string = ""
    fig = {}
    first_run_flag = True
    app_layout = app_instance.app.layout

    # Set the State of the callback
    min_cluster_size_value = DEFAULT_MIN_CLUSTER_SIZE
    min_samples_value = DEFAULT_MIN_SAMPLES
    n_components_value = DEFAULT_N_COMPONENTS
    n_neighbors_value = DEFAULT_N_NEIGHBORS
    top_n_topic_words_value = DEFAULT_TOP_N_TOPIC_WORDS
    snippet_processing_strategy = DEFAULT_SNIPPET_PROCESSING_STRATEGY
    context_value.set(AttributeDict(**{"triggered_inputs": [{"prop_id": "run-btn.n_clicks"}]}))

    # Call the callback function
    try:
        output_fig, output_hyperparameters_display, output_first_run_flag = app_instance.run_modeling_callback(
            n_clicks,
            is_cluster_dialog_open,
            selected_labels,
            classification_labels,
            embedding_training_checklist,
            topic_model_training_checklist,
            reduce_outliers,
            param_search,
            min_cluster_size_value,
            min_samples_value,
            n_components_value,
            n_neighbors_value,
            top_n_topic_words_value,
            min_cluster_size_range,
            min_samples_range,
            n_components_range,
            n_neighbors_range,
            bayesian_iterations,
            hyperparam_string,
            fig,
            first_run_flag,
            snippet_processing_strategy,
            app_layout
        )
    except PreventUpdate:
        assert False, "Callback prevented update unexpectedly"

    # Assert that the output figure has 'data' attribute
    assert 'data' in output_fig, "'data' attribute not found in the figure"
    assert len(output_fig['data']) > 0, "'data' list is empty"


def test_add_new_cluster_increases_count(app_instance):
    '''
    Test the callback function that adds a new cluster to the topic model.
    '''
    app_instance.create_layout([])
    topic_model = app_instance.cache.get('topic_model')
    initial_cluster_count = len(topic_model.topic_labels_)

    cluster_management_n_clicks = 0
    is_open = True
    add_new_cluster_n_clicks = 1
    snippets_modal_add_new_cluster_n_clicks = 0
    recalculate_cluster_info_n_clicks = 0

    context_value.set(AttributeDict(**{"triggered_inputs": [{"prop_id": "add-new-cluster-btn.n_clicks"}]}))
    try:
            is_open, topic_details, is_confirmation_displayed, message = app_instance.show_cluster_dialog_callback(
                cluster_management_n_clicks,
                is_open,
                add_new_cluster_n_clicks,
                snippets_modal_add_new_cluster_n_clicks,
                recalculate_cluster_info_n_clicks
            )
    except PreventUpdate:
        assert False, "Callback prevented update unexpectedly"

    # Get the updated number of clusters
    topic_model_updated = app_instance.cache.get('topic_model')
    updated_cluster_count = len(topic_model_updated.topic_labels_)

    # Assert that the count of clusters has increased by one
    assert updated_cluster_count == initial_cluster_count + 1, f"Expected {initial_cluster_count + 1} clusters, but got {updated_cluster_count}"

def test_update_cluster_names_callback(app_instance):
    '''
    Test the callback function that updates the names of the clusters.
    The callback function should update the name of the cluster with index -1 to 'outliers'.
    '''
    app_instance.create_layout([])
    topic_model = app_instance.cache.get('topic_model')
    topic_names = list(topic_model.topic_labels_.values())
    topic_ids = [{'type': 'topic-name', 'index': index} for index in topic_model.topic_labels_.keys()]

    # Add a new name 'outliers' to topic -1
    new_topic_names = topic_names[:]
    if -1 in topic_model.topic_labels_:
        index = list(topic_model.topic_labels_.keys()).index(-1)
        new_topic_names[index] = 'outliers'
    
    context_value.set(AttributeDict(**{"triggered_inputs": [{"prop_id": '{"type":"topic-name","index":-1}.value'}]}))
    try:
        output = app_instance.update_cluster_names_callback(
            new_topic_names,
            topic_ids
        )
    except PreventUpdate:
        assert False, "Callback prevented update unexpectedly"

    updated_topic_model = app_instance.cache.get('topic_model')
    assert updated_topic_model.topic_labels_[-1] == 'outliers', "Topic -1 name was not updated correctly"


def test_remove_cluster_decreases_count(app_instance):
    '''
    Test the callback function that removes a cluster from the topic model.
    The callback function should remove the first cluster from the topic model.
    '''
    app_instance.create_layout([])
    topic_model = app_instance.cache.get('topic_model')
    initial_cluster_count = len(topic_model.topic_labels_)

    btn_show_clicks = [0] * initial_cluster_count
    search_term = ""
    is_open = True
    btn_remove_clicks = 1
    btn_select_all_clicks = 0
    current_topic_id = list(topic_model.topic_labels_.keys())[0]  # Assume we remove the first cluster

    context_value.set(AttributeDict(**{"triggered_inputs": [{"prop_id": "btn-remove-cluster.n_clicks"}]}))
    try:
        is_open, checklist, dropdown_options, modal_title, current_topic_id = app_instance.show_cluster_dialog_content_callback(
            btn_show_clicks,
            search_term,
            is_open,
            btn_remove_clicks,
            btn_select_all_clicks,
            current_topic_id
        )
    except PreventUpdate:
        assert False, "Callback prevented update unexpectedly"

    # Get the updated number of clusters
    topic_model_updated = app_instance.cache.get('topic_model')
    updated_cluster_count = len(topic_model_updated.topic_labels_)

    # Assert that the count of clusters has decreased by one
    assert updated_cluster_count == initial_cluster_count - 1, f"Expected {initial_cluster_count - 1} clusters, but got {updated_cluster_count}"


def test_save_embedding_model(app_instance):
    app_instance.create_layout([])
    embedding_model = app_instance.cache.get('embedding_model')

    # Simulate saving an embedding model
    save_file_clicks = 0
    save_embedding_model_clicks = 1
    save_topic_model_clicks = 0
    confirm_clicks = 1
    cancel_clicks = 0
    embeddings_checklist_value = []
    last_opened_save_modal = 'save-embedding-model-btn'
    snippet_processing_strategy = ''

    context_value.set(AttributeDict(**{"triggered_inputs": [{"prop_id": "filename-confirm.n_clicks"}]}))
    try:
        modal_style, embedding_checklist_style, last_opened_modal = app_instance.handle_saving_modal_callback(
            save_file_clicks,
            save_embedding_model_clicks,
            save_topic_model_clicks,
            confirm_clicks,
            cancel_clicks,
            EMBEDDING_MODEL_FILENAME,
            embeddings_checklist_value,
            last_opened_save_modal,
            snippet_processing_strategy
        )
    except PreventUpdate:
        assert False, "Callback prevented update unexpectedly"

    # Check if the embedding model file exists and then delete it
    embedding_model_path = f'{app_instance.output_dir}/{EMBEDDING_MODEL_DIR}/{EMBEDDING_MODEL_FILENAME}'
    assert os.path.exists(embedding_model_path), f"Expected file {embedding_model_path} does not exist"

def teardown_module(module):
    shutil.rmtree(f'{OUTPUT_DIR.replace("../", "")}/{EMBEDDING_MODEL_DIR}/{EMBEDDING_MODEL_FILENAME}')

