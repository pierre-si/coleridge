# %%
"""The purpose of this notebook is to check if publication clustering (with spacy and words vectors) correlates with their dataset_labels: i.e. if publications citing the same datasets appear in the same cluster. We can compute the datasets_labels similarity by using co-occurrences in a BI manner, a Jaccard distance or a word vectors similarity. Issue: when using dataset_label instead of dataset_title, some titles will refer to the same label. We need to take into account that these titles, although different should increase the similarity. This should be implicit when using word vectors. This can be done with Jaccard if we apply the Jaccard distance on the title strings."""
# %%
import json
from pathlib import Path

from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import spacy
# %%
df_train = pd.read_csv('../input/smaller/train.csv')
df_title_sets = df_train.groupby('Id')['dataset_title'].agg(set)
# %% Publication's datasets similarity based on Jaccard
def jaccard_sim(set1, set2):
    inter = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return float(inter) / union
# %%
n_pub = len(df_title_sets)
jaccard_matrix = np.identity(n_pub)
# %%
for i in range(n_pub):
    for j in range(i+1, n_pub):
        jaccard_matrix[i][j] = jaccard_sim(df_title_sets.iloc[i], df_title_sets.iloc[j])
# %%
jaccard_matrix = jaccard_matrix + jaccard_matrix.T - np.identity(n_pub)
# %%
fig, ax = plt.subplots()
im = ax.imshow(jaccard_matrix)
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Jaccard sim', rotation=-90, va="bottom")
# %% Publications' text similarity
nlp = spacy.load("en_core_web_lg")
publication_folder = Path('../input/smaller/train')
id_to_doc = {}
for stem in tqdm(df_title_sets.index):
    with open(publication_folder/(stem+'.json'), 'r') as f:
        publication = json.load(f)
    id_to_doc[stem] = nlp(' '.join([section['text'] for section in publication]))
# %%
word_sim = np.identity(n_pub)
# %%
for i in range(n_pub):
    for j in range(i+1, n_pub):
        word_sim[i][j] = id_to_doc[df_title_sets.index[i]].similarity(id_to_doc[df_title_sets.index[j]])
# %%
word_sim = word_sim + word_sim.T - np.identity(n_pub)
# %%
from statistics import mean
jaccard_matrix -= 2*np.identity(n_pub)
# %%
for jaccard_val in np.unique(jaccard_matrix)[1:]:
    print("Average similarity among the", np.sum(jaccard_matrix == jaccard_val), "docs with Jaccard index", '{:.2f}'.format(jaccard_val), ':', '{:.3f}'.format(mean(word_sim[jaccard_matrix == jaccard_val])))
# %%
# Publications with jaccard index of 0 (no shared dataset) have a similarity of .95.
# Publications with a jaccard index greater than 0 (at least one shared dataset) have a similarity equal or greater than .97