# Notebooks
- chatintents - a tutorial notebook for clustering /w hyperparamter search and topic extraction
- clustering - old, better refer to ClusteringProduction
- company_embeddings - create company embeddings dictionary from cn_descriptors_top (definition and additional definitions), used in the sentence_similarity notebook
- lexical_embeddings - create keywords and classification embeddings used in the sentence_similarity notebook
- sentence_similarity - create a dataset with calculated sentence similarities as well as lexical (country, keywords, company, classification)
- embedding_analysis - analyze different models (laser, sbert, glove) for company embeddings 
- classification - prepare the dataset with classification labels, train sentence embeddings with a triplet loss
- embeddings-training - on Google Drive, training sentence embeddings based on the dataset from sentence_similarity

# Important directories
- S3BERT/s3bert_all-MiniLM-L12-v2 - trained S3BERT model
- S3BERT/src - source code used for original embeddings-training