import numpy as np
import pandas as pd
import umap
from sklearn.manifold import TSNE

df = pd.read_pickle("esm_emb.pkl")

embedding = umap.UMAP(n_neighbors=20).fit_transform(df.embedding.to_list())

X_embedded = TSNE(
    n_components=2, learning_rate="auto", init="random", perplexity=3
).fit_transform(np.array(df.embedding.to_list()))


df["umap_0"] = embedding[:, 0]
df["umap_1"] = embedding[:, 1]

df["tsne_0"] = X_embedded[:, 0]
df["tsne_1"] = X_embedded[:, 1]

df.to_pickle("esm_emb_umap.pkl")
