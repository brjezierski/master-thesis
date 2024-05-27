
# Run locally

Install conda environment and run `python app.py`


# Run with Docker

```
docker build -t  topic-modelling .
docker run -h localhost -p 9000:9000 --name tm topic-modelling
```

Open [http://0.0.0.0:9000/](http://0.0.0.0:9000/)

# Test

Execute 
```
bash run_tests.sh
```


# Errors

If you get max_df < min_df error, it means that the input file is too short for given hyperparameters.

# Input format

tsv, csv or json with required columns classification, snippet, and replace and optional embedding column

# More

Each embedding model can be trained on two formats of snippets, either the format I used for my thesis research with named entity tags removed or the format used by Sebastian for training (with named entity tags modified like in this example: #COMPANY -> <COMPANY>). Cache is used to preserve variables instead of global variables (justification [here](https://dash.plotly.com/sharing-data-between-callbacks)).

# Hyperparameters

- min_samples - The simplest intuition for what min_samples does is provide a measure of how conservative you want you clustering to be. The larger the value of min_samples you provide, the more conservative the clustering – more points will be declared as noise, and clusters will be restricted to progressively more dense areas. -> outlier number

- min_cluster_size - The primary parameter to effect the resulting clustering is min_cluster_size. Ideally this is a relatively intuitive parameter to select – set it to the smallest size grouping that you wish to consider a cluster. 

- n_neighbors - This parameter controls how UMAP balances local versus global structure in the data. It does this by constraining the size of the local neighborhood UMAP will look at when attempting to learn the manifold structure of the data. This means that low values of n_neighbors will force UMAP to concentrate on very local structure (potentially to the detriment of the big picture), while large values will push UMAP to look at larger neighborhoods of each point when estimating the manifold structure of the data, losing fine detail structure for the sake of getting the broader of the data.

# Files
- app.py - The `TopicModelingApp` class is the core of a topic modeling dashboard application built using Dash. It initializes the app with necessary styles and cache configurations, sets up directories for storing models and data, and defines the main layout of the application. The layout includes various user interface components such as modals, dropdowns, checklists, sliders, and graphs. The class also sets up multiple callbacks to handle user interactions, process data uploads, manage model training and evaluation, and update the user interface dynamically. These callbacks facilitate functionalities like enabling buttons, displaying modals, running topic modeling processes, and managing the application's state. Overall, this file orchestrates the entire user experience and functionality of the topic modeling dashboard.
- components.py - This file defines several utility functions to create different components for a topic modeling dashboard using Dash and Dash Bootstrap Components. These components include modals for file naming, snippet display, and file uploads, as well as sections for selecting labels, embedding models, and topic models. It also sets up parameter sliders and inputs for clustering hyperparameters, checklists for outlier reduction, and sections for merging clusters. Additionally, it provides various action buttons and a cluster management modal to facilitate user interactions within the dashboard application.
- MyBERTopic.py - The `MyBERTopic` class extends the BERTopic model to include customized initialization and additional functionalities tailored for specific topic modeling tasks. The constructor sets up UMAP for dimensionality reduction, HDBSCAN for clustering, and CountVectorizer for text vectorization, alongside embedding and representation models. The class includes methods to train or load the topic model, generate topics, manage topics (add, remove, merge), and update topic-related structures. It also provides capabilities for document retrieval based on topics and visualization of documents in reduced dimensional space. These enhancements facilitate a more flexible and comprehensive topic modeling process suited to diverse applications.
- MyDataset.py - The `MyDataset` class processes a DataFrame of labeled text snippets for use in training a sentence transformer model with triplet loss. It splits the data into training, development, and test sets, ensuring paired samples for validation and testing. The class generates triplets from labeled data, where each triplet consists of an anchor, a positive example (same label), and a negative example (different label). Additionally, it prepares the data for training by converting snippets into the required InputExample format and creates appropriate data loaders and evaluators for model training and evaluation.
- MyTrainer.py - The `MySentenceTransformer` class extends the functionality of `SentenceTransformer`, providing methods for embedding documents and words while also offering options to freeze all or specific layers of the model.
- ParameterOptimizer.py - The `ParameterOptimizer` class orchestrates the optimization of hyperparameters for the UMAP + HDBSCAN model, leveraging Bayesian search via Hyperopt, and evaluates the resulting clustering performance using metrics such as overlap loss and topic diversity, facilitating informed decision-making in model configuration for clustering tasks. Additionally, it provides utilities for generating clusters, computing clustering metrics, and assessing topic diversity, enhancing the interpretability and effectiveness of the clustering process.
- utils.py - This script provides functions for data preprocessing, text cleaning, embedding creation, and classification label manipulation, facilitating tasks such as loading data from TSV or JSON files, filtering classifications based on occurrence cutoffs, and generating embeddings from text. Additionally, it offers utilities for computing cosine similarity between embeddings, replacing NaN values, and extracting relevant classification labels, enhancing the efficiency and effectiveness of text processing workflows.
- tests/test_callbacks.py - a few callback tests


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
- end-to-end testign script
