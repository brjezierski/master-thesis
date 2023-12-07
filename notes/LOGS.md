# 22-05-23

Skip thoughts is not working (EOFError when running the original code, wrong tf version when running elvisyjlin). Alsop, this is an annoying error
```
ValueError: Object arrays cannot be loaded when allow_pickle=False
```
Will [downgrading numpy](https://stackoverflow.com/questions/55890813/how-to-fix-object-arrays-cannot-be-loaded-when-allow-pickle-false-for-imdb-loa) help? No, still stuck at EOFError. Will attempt [`pip install skipthoughts`](https://pypi.org/project/skipthoughts/) now. I tried it with the files downloaded earlier and got `_pickle.UnpicklingError: pickle data was truncated`. So, I will try to reinstall it now. For now I placed data.zip in trash.
It works! Now gotta figure out how to use it, no tutorial online. [This](https://github.com/jamesoneill12/NearestNeighborLanguageModeling/blob/25aa4a79cc48a953ba39fa651393f4d46d90a32c/models/embeddings/sentence/sent_reps.py#L8) might be helpful.
Also, I managed to get the chat intents library working to determine the right number of clusters. I will use the best hyperparams I got to evaluate clusterability of this dataset with different sentence embeddings.

# 23-5-23
Gotta run this to make laser work
```
python -m laserembeddings download-models
```
The [pip rep for skipthoughts](https://github.com/Cadene/skip-thoughts.torch/tree/master/pytorch)

# 24-05-23
The command I use
```
python cluster.py -kw labeled_companies_glove.pkl -k 5 -plot -tooltip_df companies_tooltip.csv -color tooltip
```

# 18-07-23
I am modifying the training dataset for the similarity training to exclude snippets less than 40 char long which do not have a verb. I am combining into the new pairs in such a way that they do not have two same companies. I saved them in similarity-training-data/sbert-company-filtered. It is not performing any better.

Then I tried using the classification model and I got negative cosine similarity (which is weird and unexpected), sbert similarity has no negative cosine similarity. [The answer might be here](https://vaibhavgarg1982.medium.com/why-are-cosine-similarities-of-text-embeddings-almost-always-positive-6bd31eaee4d5)