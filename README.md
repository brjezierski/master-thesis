# Research

## Sentence embeddings
[CSE: Conceptual Sentence Embeddings based on Attention Model](https://aclanthology.org/P16-1048.pdf)  

[MINILM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers](https://arxiv.org/pdf/2002.10957.pdf)  

[Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/pdf/1908.10084.pdf)  
  - siamese and triplet network structures
  - averaging BERT output layer 
  - We fine-tune SBERT on NLI data
  - other methods:
    - Universal Sentence Encoder (Cer et al., 2018) trains a transformer network and augments unsupervised learning with training on SNLI.
    - InferSent (Conneau et al., 2017) uses labeled data of the Stanford Natural Language Inference dataset (Bowman et al., 2015) and the MultiGenre NLI dataset (Williams et al., 2018) to train a siamese BiLSTM network with max-pooling over the output
    - Humeau et al. (2019) addresses the run-time overhead of the cross-encoder from BERT and present a method (poly-encoders) to compute a score between m context vectors and pre computed candidate embeddings using attention.
  - data:
    - SNLI is a collection of 570,000 sentence pairs annotated with the labels contradiction, eintailment, and neutral
    - Semantic Textual Semilarity (STS) benchmark (Cer et al., 2017)
      - STS tasks 2012 - 2016 (Agirre et al., 2012, 2013, 2014, 2015, 2016), 
      - [the STS benchmark (Cer et al., 2017)](https://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark) - https://huggingface.co/datasets/stsb_multi_mt/discussions, 
      - and the SICK-Relatedness dataset (Marelli et al., 2014). These datasets provide labels between 0 and 5 on the semantic relatedness of sentence pairs
    - [MultipleNegativesRankingLoss](https://www.sbert.net/docs/package_reference/losses.html#multiplenegativesrankingloss) is currently the best method to train sentence embeddings. As training data, we need text-pairs (textA, textB) where we want textA and textB close in vector space. This can be anything like (question, answer), (text, summary), (paper, related_paper), (input, response).
  - functions:
    - Classification Objective Function. We concatenate the sentence embeddings u and v with the element-wise difference |u−v| and multiply it with the trainable weight Wt 
    - Regression Objective Function. The cosinesimilarity between the two sentence embeddings u and v
    - Triplet Objective Function. Given an anchor sentence a, a positive sentence p, and a negative sentence n, triplet loss tunes the network such that the distance between a and p is smaller than the distance between a and n


[KEPLER: A Unified Model for Knowledge Embedding and Pre-trained Language Representation](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00360/98089/KEPLER-A-Unified-Model-for-Knowledge-Embedding-and)
  - knowledge embedding (KE) methods (Bordes et al., 2013; Yang et al., 2015; Sun et al., 2019) 
  - Unlike previous knowledge-enhanced PLM works (Zhang et al., 2019; Peters et al., 2019), we do not modify the Transformer encoder structure to add external entity linkers or knowledge-integration layers. It means that our model has no additional inference overhead compared to vanilla PLMs, and it makes applying KEPLER in downstream tasks as easy as RoBERTa.
  - In this paper, we explore three simple but effective ways: entity descriptions as embeddings, entity and relation descriptions as embeddings, and entity embeddings conditioned on relations
  - Knowledge Embedding: KE methods have been extensively studied. Conventional KE models define different scoring functions for relational triplets. For example, TransE (Bordes et al., 2013) treats tail entities as translations of head entities and uses L1-norm or L2-norm to score triplets, while DistMult (Yang et al., 2015) uses matrix multiplications and ComplEx (Trouillon et al., 2016) adopts complex operations based on it. RotatE (Sun et al., 2019) combines the advantages of both of them.

[Technological troubleshooting based on sentence embedding with deep transformers](https://link.springer.com/content/pdf/10.1007/s10845-021-01797-w.pdf?pdf=button)  

[Translating Embeddings for Modeling Multi-relational Data](https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf) - TransE, an energy-based model for learning low-dimensional embeddings of entities  

A way of testing sentence encodings is to apply them on Sentences Involving Compositional Knowledge (SICK) corpus[13] for both entailment (SICK-E) and relatedness (SICK-R).



[A Comparative Study on Structural and Semantic Properties of Sentence Embeddings](https://arxiv.org/pdf/2009.11226.pdf) - on [git](https://github.com/akalino/semanticstructural-sentences.), using sentence embeddings for relation extraction 
  - evaluating the extent to which sentences
carrying similar senses are embedded in close proximity subspaces, and if we can exploit that structure to align sentences
to a knowledge graph
  - to evaluate the degree to which different sentence
embedding methods can be mapped via linear methods to their
semantically grounded counterparts 
  - clusterability -  our hypothesis is that embedding techniques that exhibit higher degrees of clusterability are able to capture more of the syntactic and semantic regularities of the input text data sources
  - different sentence embedding approaches analyzed
    - GloVe-mean - taking an average of word embeddings as a sentence embedding
    - Skip-Thought model generates an encoding for a center sentence and uses that encoding to predict k sentences to the left and right (an extension is the Quick-Thought model)
    - InferSentV1 and InferSentV2
    - BERT
    - LASER
    - geometric embedding algorithm (GEM) that focuses on tuning existing pretrained language representations. By analyzing the geometry of the subspace generated by building a matrix A = d × n for the n words in a given sentence  

[Business2Vec: Identifying similar businesses using embeddings](https://medium.com/@eniola.alese/business2vec-identifying-similar-businesses-using-embeddings-part-i-82962fd3ecac) - PV-DBOW with negative sampling for training

## Tutorials

[Notebook for fine-tuning sentence embeddings](https://huggingface.co/blog/how-to-train-sentence-transformers)
[Fine-Tuning Sentence Transformers with MNR Loss](https://towardsdatascience.com/fine-tuning-sentence-transformers-with-mnr-loss-cd6a26685b81)


## Few shot learning
[Easy to read summary](https://www.nttdata.com/global/en/-/media/nttdataglobal/1_files/services/data-and-intelligence/few-shot_learning_in_natural_language_processing.pdf?rev=37e2bca167ad474485aa5de0978851f3)
[Beyond Fine-tuning: Few-Sample Sentence Embedding Transfer](https://aclanthology.org/2020.aacl-main.47.pdf) - for text classification tasks by concatenating a an original vector, 

# Models
[sentence-transformers/all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) - the one we're using
[knowBERT](https://github.com/allenai/kb)

# Ideas
- model distillation (for later on)  
- few shot learning (fine-tuning on some Glanos data)
  - [presentation](https://www.nttdata.com/global/en/-/media/nttdataglobal/1_files/services/data-and-intelligence/few-shot_learning_in_natural_language_processing.pdf?rev=37e2bca167ad474485aa5de0978851f3)



Knowledge in sentence embeddings 
3 research questions:  
- semantic faithfulness (Diego) - there are 8 or 10 people, negation (60k sentences)
- ontology downstream task 
- machine translation downstream task - dataset (WMT - quality estimation), comment, 
- topic modelling - repo from Miriam 
- I have a sentence embedding, what knowledge can I get out of it 
- identifying  


