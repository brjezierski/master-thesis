from sentence_transformers import SentenceTransformer

class GlanosSentenceTransformer(SentenceTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def embed_documents(self, documents, verbose=False):
        """
        merge_topics method of BERTopic requires embed_documents and gives an error otherwise
        """
        embeddings = self.encode(documents, show_progress_bar=verbose)
        return embeddings
    
    def embed_words(self, words, verbose=False):
        """
        merge_topics method of BERTopic requires embed_words and gives an error otherwise
        """
        embeddings = self.encode(words, show_progress_bar=verbose)
        return embeddings
