import itertools
import os

import dgl
import dgl.function as fn
import numpy as np
import pandas as pd
import torch
from dgl.data import DGLDataset
from dgl.data.utils import (
    download,
    save_graphs,
    load_graphs,
    generate_mask_tensor,
    idx2mask,
)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from utils import Util


class IMDbDataset(DGLDataset):
    """IMDb Movie Dataset, with a single heterogeneous graph

    Statistics
    -----
    * Nodes: 4278 movie, 5257 actor, 2081 director
    * Edges: 12828 movie-actor, 4278 movie-director
    * Number of classes: 3
    * Movie node partition: 400 train, 400 valid, 3478 test

    Attributes
    -----
    * num_classes: Number of classes
    * metapaths: Metapaths used
    * predict_ntype: Node type to predict

    Movie node attributes
    -----
    * feat: tensor(4278, 1299) Bag-of-words representation of plot keywords
    * label: tensor(4278) 0: Action, 1: Comedy, 2: Drama
    * train_mask, val_mask, test_mask: tensor(4278)

    Actor node attributes
    -----
    * feat: tensor(5257, 1299) Average movie features

    Director node attributes
    -----
    * feat: tensor(2081, 1299) Average movie features
    """

    _url = "https://raw.githubusercontent.com/Jhy1993/HAN/master/data/imdb/movie_metadata.csv"
    _seed = 42

    def __init__(self):
        super().__init__("imdb", self._url)

    def download(self):
        file_path = os.path.join(self.raw_dir, "imdb.csv")
        if not os.path.exists(file_path):
            download(self.url, path=file_path)

    def save(self):
        save_graphs(
            os.path.join(self.save_path, self.name + "_dgl_graph.bin"), [self.g]
        )

    def load(self):
        graphs, _ = load_graphs(
            os.path.join(self.save_path, self.name + "_dgl_graph.bin")
        )
        self.g = graphs[0]
        for k in ("train_mask", "val_mask", "test_mask"):
            self.g.nodes["movie"].data[k] = self.g.nodes["movie"].data[k].bool()

    def process(self):
        self.data = (
            pd.read_csv(os.path.join(self.raw_dir, "imdb.csv"), encoding="utf8")
            .dropna(axis=0, subset=["actor_1_name", "director_name"])
            .reset_index(drop=True)
        )
        self.labels = self._extract_labels()
        self.movies = list(sorted(m.strip() for m in self.data["movie_title"]))
        self.directors = list(sorted(set(self.data["director_name"])))
        self.actors = list(
            sorted(
                set(
                    itertools.chain.from_iterable(
                        self.data[c].dropna().to_list()
                        for c in ("actor_1_name", "actor_2_name", "actor_3_name")
                    )
                )
            )
        )
        self.g = self._build_graph()
        self._add_ndata()

    def _extract_labels(self):
        """Extract movie genres as labels and remove movies of other genres."""
        labels = np.full(len(self.data), -1)
        for i, genres in self.data["genres"].items():
            for genre in genres.split("|"):
                if genre == "Action":
                    labels[i] = 0
                    break
                elif genre == "Comedy":
                    labels[i] = 1
                    break
                elif genre == "Drama":
                    labels[i] = 2
                    break
        other_idx = np.where(labels == -1)[0]
        self.data = self.data.drop(other_idx).reset_index(drop=True)
        return np.delete(labels, other_idx, axis=0)

    def _build_graph(self):
        ma, md = set(), set()
        for m, row in self.data.iterrows():
            d = self.directors.index(row["director_name"])
            md.add((m, d))
            for c in ("actor_1_name", "actor_2_name", "actor_3_name"):
                if row[c] in self.actors:
                    a = self.actors.index(row[c])
                    ma.add((m, a))
        ma, md = list(ma), list(md)
        ma_m, ma_a = [e[0] for e in ma], [e[1] for e in ma]
        md_m, md_d = [e[0] for e in md], [e[1] for e in md]
        return dgl.heterograph(
            {
                ("movie", "ma", "actor"): (ma_m, ma_a),
                ("actor", "am", "movie"): (ma_a, ma_m),
                ("movie", "md", "director"): (md_m, md_d),
                ("director", "dm", "movie"): (md_d, md_m),
            }
        )

    def _split_idx(self, samples, train_size, val_size, random_state=None):
        """Split samples into training, validation, and test sets, satisfying the following conditions (expressed as floating-point numbers):

        * 0 < train_size < 1
        * 0 < val_size < 1
        * train_size + val_size < 1

        :param samples: list/ndarray/tensor of samples
        :param train_size: int or float If it is an integer, it represents the absolute number of training samples; otherwise, it represents the proportion of training samples in the entire dataset
        :param val_size: int or float If it is an integer, it represents the absolute number of validation samples; otherwise, it represents the proportion of validation samples in the entire dataset
        :param random_state: int, optional Random seed
        :return: (train, val, test) with the same type as samples
        """
        train, val = train_test_split(
            samples, train_size=train_size, random_state=random_state
        )
        if isinstance(val_size, float):
            val_size *= len(samples) / len(val)
        val, test = train_test_split(
            val, train_size=val_size, random_state=random_state
        )
        return train, val, test

    def _add_ndata(self):
        vectorizer = CountVectorizer(min_df=5)
        features = vectorizer.fit_transform(
            self.data["plot_keywords"].fillna("").values
        )
        self.g.nodes["movie"].data["feat"] = torch.from_numpy(
            features.toarray()
        ).float()
        self.g.nodes["movie"].data["label"] = torch.from_numpy(self.labels).long()

        # Actor and director node features are the average of associated movie node features
        self.g.multi_update_all(
            {
                "ma": (fn.copy_u("feat", "m"), fn.mean("m", "feat")),
                "md": (fn.copy_u("feat", "m"), fn.mean("m", "feat")),
            },
            "sum",
        )

        n_movies = len(self.movies)
        train_idx, val_idx, test_idx = self._split_idx(
            np.arange(n_movies), 400, 400, self._seed
        )
        self.g.nodes["movie"].data["train_mask"] = generate_mask_tensor(
            idx2mask(train_idx, n_movies)
        )
        self.g.nodes["movie"].data["val_mask"] = generate_mask_tensor(
            idx2mask(val_idx, n_movies)
        )
        self.g.nodes["movie"].data["test_mask"] = generate_mask_tensor(
            idx2mask(test_idx, n_movies)
        )

    def has_cache(self):
        return os.path.exists(
            os.path.join(self.save_path, self.name + "_dgl_graph.bin")
        )

    def correlation_score(self):
        return Util.compute_correlation(self.g, "movie")

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError("This dataset has only one graph")
        return self.g

    def __len__(self):
        return 1

    @property
    def num_classes(self):
        return 3

    @property
    def metapaths(self):
        return [["ma", "am"], ["md", "dm"]]

    @property
    def predict_ntype(self):
        return "movie"
