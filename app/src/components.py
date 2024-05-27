from dash import dcc,  html
import dash_bootstrap_components as dbc
import itertools
from typing import List

def create_filename_modal() -> dcc.Loading:
    '''
    Create the filename modal for exporting the dataframe 
    with newly labeled snippets with or without embeddings.

    Returns:
        dcc.Loading: The filename modal component.
    '''
    return dcc.Loading(
        type="default",
        children=html.Div(id='filename-modal', style={'display': 'none'}, children=[
            html.Div([
                html.Label('Enter the filename:'),
                dcc.Input(id='filename-input', type='text', placeholder='Filename'),
                html.Div([
                    dbc.Checklist(
                        options=[{'label': 'Include embeddings', 'value': 'embeddings'}],
                        value=['embeddings'],
                        id="filename-embeddings-checklist",
                    ),
                ], style={'display': 'none'}, id='filename-embeddings-checklist-div'),
                html.Button('Save', id='filename-confirm'),
                html.Button('Cancel', id='filename-cancel'),
            ], className='modal-content')
        ])
    )

def create_snippets_modal() -> dbc.Modal:
    '''
    Create the snippets modal for displaying, selecting and searching through snippets.
    Add remove and add cluster buttons.

    Returns:
        dbc.Modal: The snippets modal component.
    '''
    return dbc.Modal(
        [
            dbc.ModalHeader([
                dbc.ModalTitle(id="snippets-modal-title"),
            ]),
            dbc.ModalFooter([
                html.Div([
                    dcc.Dropdown(id="dropdown-assign", options=[]),
                    dbc.Button("Assign", id="btn-assign", n_clicks=0)
                ], style={'width': '100%'}),
                dbc.Button("Select All", id="btn-select-all", n_clicks=0),
                dcc.Input(id='snippet-search', type='text', placeholder='Search Snippets'),
                dbc.Button("Remove Cluster", id="btn-remove-cluster", n_clicks=0),
                dbc.Button('Add New Cluster', id='snippets-modal-add-new-cluster-btn', n_clicks=0),
            ]),
            dbc.ModalBody(id="snippets-modal-content", children='Snippets listed'),
        ],
        id="snippets-modal",
        is_open=False,
        scrollable=True,
        size="lg"
    )

def create_upload_modal() -> dcc.Upload:
    '''
    Create the upload modal for a file with snippets.

    Returns:
        dcc.Upload: The upload modal component.
    '''
    return dcc.Upload(
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

def create_label_selection(classifications: List[str]) -> html.Div:
    '''
    Create the label selection dropdown for which labels to use during inference.

    Args:
        classifications (List[str]): List of available classification labels.

    Returns:
        html.Div: The label selection component.
    '''
    return html.Div([
        html.Div(children='Select classification labels for inference'),
        dcc.Dropdown(
            id='label-dropdown',
            options=[{'label': label, 'value': label} for label in classifications],
            multi=True
        ),
        html.Button('Select All', id='select-all-labels-btn', n_clicks=0)
    ])

def create_embedding_model_section(get_all_model_options: callable, embedding_model_dir: str, default_embedding_model: str) -> html.Div:
    '''
    Create the section for selecting an embedding model and checking whether it should be trained.

    Args:
        get_all_model_options (callable): Function to get all model options.
        embedding_model_dir (str): Directory for embedding models.
        default_embedding_model (str): Default embedding model.

    Returns:
        html.Div: The embedding model section component.
    '''
    return html.Div([
        dcc.Checklist(
            id='embedding-training-checklist',
            options=[{'label': 'Train the embedding model', 'value': 'embedding_training'}],
            value=[]
        ),
        dcc.Dropdown(
            id='embedding-model-dropdown',
            options=get_all_model_options(embedding_model_dir),
            value=[default_embedding_model],
            multi=False
        )
    ], id='embedding-model-dropdown-div')

def create_topic_model_section(get_all_model_options: callable, topic_model_sir: str, default_topic_model: str) -> html.Div:
    '''
    Create the section for selecting a topic model and checking whether it should be trained.

    Args:
        get_all_model_options (callable): Function to get all model options.
        topic_model_sir (str): Directory for topic models.
        default_topic_model (str): Default topic model.

    Returns:
        html.Div: The topic model section component.
    '''
    return html.Div([
        dcc.Checklist(
            id='topic-model-training-checklist',
            options=[{'label': 'Train the topic model', 'value': 'topic_model_training'}],
            value=['topic_model_training']
        ),
        dcc.Dropdown(
            id='topic-model-dropdown',
            options=get_all_model_options(topic_model_sir),
            value=[default_topic_model],
            multi=False
        )
    ], id='topic-model-dropdown-div')

def create_parameter_sliders() -> html.Div:
    '''
    Create the parameter sliders for various clustering dimensionality reduction hyper-parameters
    for hyper-parameter search.

    Returns:
        html.Div: The parameter sliders component.
    '''
    return html.Div([
        html.Div([
            html.Label('Min cluster size'),
            dcc.RangeSlider(
                id='min-cluster-size-range',
                min=0,
                max=80,
                step=1,
                marks={i: str(i) for i in range(0, 81, 10)},
                value=[7, 18],
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
                value=[2, 12],
            ),
        ]),
        html.Div([
            html.Label('Number of neighbors'),
            dcc.RangeSlider(
                id='n-neighbors-range',
                min=2,
                max=100,
                step=1,
                marks={i: str(i) for i in itertools.chain(range(10, 101, 10), [2])},
                value=[4, 15],
            ),
        ]),
        html.Div([
            html.Label('Number of iterations'),
            dcc.Input(id='bayesian-iterations', type='number', value=5),
        ])
    ], id='params-range-div', style={'display': 'none'})

def create_parameter_inputs(default_min_cluster_size: int, default_min_samples: int, default_n_components: int, default_n_neighbors: int, default_top_n_topic_words: int) -> html.Div:
    '''
    Create the parameter input fields for various clustering dimensionality reduction hyper-parameters.

    Args:
        default_min_cluster_size (int): Default minimum cluster size.
        default_min_samples (int): Default minimum samples.
        default_n_components (int): Default number of components.
        default_n_neighbors (int): Default number of neighbors.
        default_top_n_topic_words (int): Default top N topic words.

    Returns:
        html.Div: The parameter input fields component.
    '''
    return html.Div([
        html.Label('Minimum cluster size'),
        dcc.Input(id='min-cluster-size', type='number', value=default_min_cluster_size),

        html.Label('Min samples'),
        dcc.Input(id='min-samples', type='number', value=default_min_samples),

        html.Label('Number of components'),
        dcc.Input(id='n-components', type='number', value=default_n_components),

        html.Label('Number of neighbors'),
        dcc.Input(id='n-neighbors', type='number', value=default_n_neighbors),
    ], id='params-div', style={'display': 'block'})

def create_other_parameter_inputs(default_top_n_topic_words: int) -> html.Div:
    '''
    Create the input fields for any parameters which should be available as input for both the Bayesian search and regular execution.

    Returns:
        html.Div: The input fields component for other parameters.
    '''
    return html.Div([
                html.Label('Top n keywords'),
                dcc.Input(id='top-n-topics', type='number', value=default_top_n_topic_words),
            ], id='other-params-div', style={'display': 'block'})

def create_outlier_checklist() -> dcc.Checklist:
    '''
    Create the outlier checklist.

    Returns:
        dcc.Checklist: The outlier checklist component.
    '''
    return dcc.Checklist(
        id='outlier-checklist',
        options=[{'label': 'Reduce outliers', 'value': 'reduce_outliers'}],
        value=[]
    )

def create_merge_section() -> html.Div:
    '''
    Create the merge section for merging clusters.

    Returns:
        html.Div: The merge section component.
    '''
    return html.Div([
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
    ], id='merge-div', style={'display': 'none'})

def create_buttons() -> html.Div:
    '''
    Create the action buttons.

    Returns:
        html.Div: The buttons component.
    '''
    return html.Div([
        html.Button('Run', id='run-btn', n_clicks=0),
        html.Button('Save state', id='save-state-btn', n_clicks=0),
        html.Button('Load state', id='load-state-btn', n_clicks=0),
        html.Button('Cluster Management', id='cluster-management-btn', n_clicks=0),
        html.Button('Save File', id='save-file-btn', n_clicks=0),
        html.Button('Save Embedding Model', id='save-embedding-model-btn', n_clicks=0),
        html.Button('Save Topic Model', id='save-topic-model-btn', n_clicks=0),
    ])

def create_cluster_management_modal() -> dbc.Modal:
    '''
    Create the cluster management modal together with two cluster buttons.

    Returns:
        dbc.Modal: The cluster management modal component.
    '''
    return dbc.Modal([
        dbc.ModalHeader([
            dbc.ModalTitle("Cluster Management"),
        ]),
        dbc.ModalBody([
            dbc.Button('Add New Cluster', id='add-new-cluster-btn', n_clicks=0),
            dbc.Button('Recalculate Cluster Info', id='recalculate-cluster-info-btn', n_clicks=0),
            html.Div(id='topic-list', children=[])
        ])
    ], id="cluster-dialog", is_open=False, scrollable=True, size="xl")
