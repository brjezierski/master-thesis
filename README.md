# Notebooks
- company_embeddings - create company embeddings dictionary from cn_descriptors_top (definition and additional definitions), used in the sentence_similarity notebook
- company_prediction - train a question answering model to predict company names to snippets
- combined_training - train sentence embedding models using both similarity and classification training
- lexical_embeddings - create keywords and classification embeddings used in the sentence_similarity notebook
- sentence_similarity - create a dataset with calculated sentence similarities as well as lexical (country, keywords, company, classification)
- embedding_analysis - analyze different models (laser, sbert, glove) for company embeddings 
- classification - prepare the dataset with classification labels, train sentence embeddings with a triplet loss
- embeddings-training - on Google Drive, training sentence embeddings based on the dataset from sentence_similarity
- country similarity - investigates interpretability of SBERT embeddings for countries
- clustering snippets - extrinsical evaluation of sentence embedding models using clustering and topic modelling 
 - similarity_embeddings_training - train sentence embeddings with the similarity training (using Google Colab)


 # Models
- Similarity (SBERT): brjezierski/sentence-embeddings-similarity
- Similarity trained on ai+car dataset: brjezierski/sentence-embeddings-similarity-ai-car
- Similarity distill-consistency model trained on consulting dataset: brjezierski/sentence-embeddings-similarity-distill-consistency
- Similarity distill-consistency model trained on ai+car dataset: brjezierski/sentence-embeddings-similarity-ai_car-distill-consistency
- Classification on consulting: brjezierski/sentence-embeddings-classification
- First classification on ai+car dataset, then similarity on consulting dataset: brjezierski/sentence-embeddings-combined-ai_car-class-consulting-sim
- First similarity on consulting dataset, then classification on ai+car dataset: brjezierski/sentence-embeddings-combined-consulting-sim-ai_car-class
- Together similarity on consulting dataset, and classification on ai+car dataset: brjezierski/sentence-embeddings-combined-consulting-sim-and-ai_car-class
- Together similarity and classification on ai+car dataset: brjezierski/sentence-embeddings-combined-ai_car-sim-and-class
- First similarity, then classification on ai+car dataset: brjezierski/sentence-embeddings-combined-ai_car-sim-class
- First classification, then similarity on ai+car dataset: brjezierski/sentence-embeddings-combined-ai_car-class-sim
- Together classification and similarity on consulting dataset: brjezierski/combined-consulting





