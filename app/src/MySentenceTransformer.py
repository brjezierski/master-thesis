from typing import List, Tuple
from numpy import ndarray
from sentence_transformers import SentenceTransformer
from torch import Tensor

class MySentenceTransformer(SentenceTransformer):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def embed_documents(self, documents: list[str], verbose: bool = False) -> Tuple[List[Tensor], ndarray, Tensor]:
        '''
        merge_topics method of BERTopic requires embed_documents and gives an error otherwise
        '''
        embeddings = self.encode(documents, show_progress_bar=verbose)
        return embeddings
    
    def embed_words(self, words: list[str], verbose: bool = False) -> Tuple[List[Tensor], ndarray, Tensor]:
        '''
        merge_topics method of BERTopic requires embed_words and gives an error otherwise
        '''
        embeddings = self.encode(words, show_progress_bar=verbose)
        return embeddings
    
    def freeze_all_layers(self) -> None:
        '''
        Freeze all layers of the model
        '''
        for name, param in self.named_parameters():
            param.requires_grad = False

    def freeze_except_last_layers(self, n: int = 2) -> None:
        '''
        Freeze all layers except the last n layers
        '''
        layernames = list(
            [name for name, _ in self.named_parameters() if "layer." in name])
        layerids = [name.split("layer.")[1].split(".")[0] for name in layernames]
        layerids = set([int(lid) for lid in layerids])
        layerids = sorted(list(layerids))
        lastn = layerids[-n:]
        lastn = ["layer." + str(lid) for lid in lastn]

        for name, param in self.named_parameters():
            lid = None
            if "layer." in name:
                lid = "layer." + name.split("layer.")[1].split(".")[0]
            if lid and lid in lastn:
                continue
            else:
                param.requires_grad = False