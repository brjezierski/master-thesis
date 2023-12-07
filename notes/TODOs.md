# Shared

How to structure results:
- compare different similarity training approaches (summarize what has been done in the notebook) 
  - S3BERT approach - graph for FEATURE_DIM, write down how many epochs (params["LOSSES"] = ["consistency", "distill2"] )
  - best weighted average - YES
  - get a loss graph for the best
  - normal cosine similarity approach - YES
  - pre-processing steps: pairing only different companies, replace, replace_no_tags, ...
  - company embeddings:
    - glanos vs sbert - YES
    - best ratio for company embeddings (main vs not main definitions) - YES
  - different starting models - YES
  - get big dataset
  - possibly:
    - different pairing of similarity 
- compare different classification training approaches 
  - snippet vs replace vs replace_no_tags - YES
  - 1 vs 2 epochs - YES
  - unfreeze layers - YES
  - EXCLUDE_ENTITY_OTHER - YES
  - different starting models - YES
  - old dataset vs new dataset - ?
  - two times the size of the smaller dataset - YES
  - add labeled data - first get best params
- compare combined training and one-after-another training (first classification - YES, then similarity and the other way around) 
- clustering:
  - models to compare:
    - cosine similarity
    - distill-consistency similarity
    - combined 
    - classification
  - datasets:
    - snippet
    - replace
    - replace /wo tags
  - metrics for Bayesian search
    - overlap loss
    - diversity loss
    - more losses: [OCTIS](https://github.com/MIND-Lab/OCTIS) - LATER
- topic modelling comparison - MAYBE

To do:
- use socomp GPUs - ask Miriam about installing packages 
- label the big dataset with company names
- compare semi- and supervised 
- figure out best loss for bayesian search
- test the approach of two training methods one after another: first classification, then similarity
- investigate other topic modelling models
- associate right rows (within each topic)
- investigate outlier cluster
- associate new points to exisiting clusters

Done:
- replace company names with a placeholder 
- measure time for dimensionality reduction vs hdbscan clustering (5-10 seconds how big the file is?)
- for each model report cluserability and topic overlap 
- remove duplicates
- make sure that replace and original use the same test_data
- save best classification model
- use best classification model for similarity
- use best classification model for clustering
- use E5 - worse 
- use topicBERT
- cluster the replace dataset
- train combined classification/similarity model
- explainability model - none found
- BERT to predict companies - what for????
- remove stopwords
- for classification: give it two datasets and two objectives - indeed turned out to be a better approach
- how about using a completely unrelated topic for testing? yes, this is my approach to the newest similarity training
- discovered - tags skew the dimensionality reduction
- dissociating companies - kinda worked by decreasing the median top company occurence but the loss is still not great
- rerun all, for chat intents
- ssh bjezierski@social-medium-3.soc.in.tum.de - password:bjezierski
- figure out GPU
- look at what Miriam sent
- ask Christian:
  - for country and company labels
  - some interface?
  - usage: should it work for a specific topic not in the training data? current approach is that we have different topics and I train within these topics 
- to finish the hyperparam search

# Similarity training

To do:
- compare DistillLoss2 results to randomly initialized embeddings
- what if we use just keyword similarity as a training metric?
- stack embeddings instead of similarities - not possible?
- use E5 
- see for which pairs the difference between the labels and predictions is the greatest
- write a training loss to approximate the residual vector to overall similarity and the lexical parts of the embeddings to a corresponding lexical similarity
- include the embedding for the word that expresses best what the sentence is like, e.g. "to ..."
- bigger dataset

Done:
- combine country and company embeddings
- try a different country def to additional def ratio
- improve company embeddings
    - include country info
    - use SBERT embeddings
- get the highest spearman
- separate the code into executable scripts
- include classification, keywords
- research fine-tuning with a small dataset
- change frozen layers
- clean short snippets from a training set (lower loss)
- use the triplet (classification) model to evaluate (0.49428657399060333, 0.9184827961965057)
- use S3BERT


# Triplet loss training
What to do:
- start with similarity embeddings

Done:
- tried 4 triplet losses - result is the same
- 5 epochs 
- check how the new embeddings perform on the similarity dataset
- try unfreezing last layer (worse results)
- increase the training dataset
- start /w S3BERT
- figure out why cosine similarity is negative - not relevant anymore
