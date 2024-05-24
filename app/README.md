
# Docker

```
docker build -t  topic-modelling .
docker run -h localhost -p 9000:9000 --name tm topic-modelling
```

Open [http://0.0.0.0:9000/](http://0.0.0.0:9000/)

# Errors

If you get max_df < min_df error, it means that the input file is too short for given hyperparameters.

# Input format

tsv, csv or json with required columns classification, snippet, and replace and optional embedding column

# More

Each embedding model can be trained on two formats of snippets, either the format I used for my thesis research with named entity tags removed or the format used by Sebastian for training (with named entity tags modified like in this example: #COMPANY -> <COMPANY>). Cache is used to preserve variables instead of global variables (justification [here](https://dash.plotly.com/sharing-data-between-callbacks)).

# Hyperparameters

- min_samples - The simplest intuition for what min_samples does is provide a measure of how conservative you want you clustering to be. The larger the value of min_samples you provide, the more conservative the clustering – more points will be declared as noise, and clusters will be restricted to progressively more dense areas.

- min_cluster_size - The primary parameter to effect the resulting clustering is min_cluster_size. Ideally this is a relatively intuitive parameter to select – set it to the smallest size grouping that you wish to consider a cluster. 

- n_neighbors - This parameter controls how UMAP balances local versus global structure in the data. It does this by constraining the size of the local neighborhood UMAP will look at when attempting to learn the manifold structure of the data. This means that low values of n_neighbors will force UMAP to concentrate on very local structure (potentially to the detriment of the big picture), while large values will push UMAP to look at larger neighborhoods of each point when estimating the manifold structure of the data, losing fine detail structure for the sake of getting the broader of the data.

# Files
- app - front-end
- chatintents - a sample of ChatIntents library which provides functionality for hyperparameter search
- clustering_utils
- embeddings_training_utils
- utils
- octis - contains diversity loss from OCTIS library instead of including OCTIS in the environment due to version incompatibility
- GlanosBERTopic - extension of BERTopic library
- GlanosSentenceTransformers - extension of SentenceTransformers library

# Functionalities to test
- train embedding model
- train topic model
- reduce outliers at first run
- reduce outliers at second run
- bayesian search
- different number for top n keywords
- cluster management:
  - change topic name and exit to see if graph is updated
  - add topic and assign a few snippets to it, click 'recalculate cluster info' and see if the new cluster info got updated
  - remove cluster, all the snippets assigned to this cluster should be moved to outliers
  - select all snippets in a cluster and move them to another cluster
- save file with embeddings and load it
- save file without embeddings and load it
- save topic model and check if can be selected
- save embedding model and check if can be selected
- exit, load state, check if correctly loaded 

# TODOs
- save state as a file with name when clicking save state
- when clicking load state allow the user to choose from saved states
- when no inference labels are selected, all labels should be used for inference. This worked initially with params["is_supervised"] (look for this var and the code that was commented out)
- add an interface to accomodate for different objectives for the hyperparameter search
- for the “Save File” option it would be good if the vector format would be cleaner, i.e. no new-line, comma-separated vector entries
