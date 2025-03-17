import os

import dgl
import dgl.function as fn
import numpy as np
import scipy.io as sio
import torch
from dgl.data import DGLDataset
from dgl.data.utils import (
    _get_dgl_url,
    get_download_dir,
    download,
    save_graphs,
    load_graphs,
    generate_mask_tensor,
    idx2mask,
)

from util import Util


class ACMDataset(DGLDataset):
    """
    -----
    * Nodes：17351 author, 4025 paper, 72 field
    * Edges: 13407 paper-author, 4025 paper-field
    * 808 train, 401 valid, 2816 test

    """

    def __init__(self):
        super().__init__("ACM", _get_dgl_url("dataset/ACM.mat"))

    def download(self):
        file_path = os.path.join(self.raw_dir, "ACM.mat")
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
            self.g.nodes["paper"].data[k] = self.g.nodes["paper"].data[k].bool()

    def process(self):
        url = "dataset/ACM.mat"
        data_path = get_download_dir() + "/ACM.mat"
        download(_get_dgl_url(url), path=data_path)

        data = sio.loadmat(data_path)
        p_vs_l = data["PvsL"].tocsr()  # paper-field?
        p_vs_a = data["PvsA"].tocsr()  # paper-author
        p_vs_t = data["PvsT"].tocsr()  # paper-term, bag of words
        p_vs_c = data["PvsC"].tocsr()  # paper-conference, labels come from that

        # We assign
        # (1) KDD papers as class 0 (data mining),
        # (2) SIGMOD and VLDB papers as class 1 (database),
        # (3) SIGCOMM and MobiCOMM papers as class 2 (communication)
        conf_ids = [0, 1, 9, 10, 13]
        label_ids = [0, 1, 2, 2, 1]

        p_vs_c_filter = p_vs_c[
            :, conf_ids
        ]  # select only the columns (conferences) we care about
        p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[
            0
        ]  # select only the rows that have at least one reference to a conference in conf_ids
        p_vs_l = p_vs_l[p_selected]
        p_vs_a = p_vs_a[p_selected]
        p_vs_t = p_vs_t[p_selected]
        p_vs_c = p_vs_c[p_selected]

        self.g = dgl.heterograph(
            {
                ("paper", "pa", "author"): p_vs_a.nonzero(),
                ("author", "ap", "paper"): p_vs_a.transpose().nonzero(),
                ("paper", "pf", "field"): p_vs_l.nonzero(),
                ("field", "fp", "paper"): p_vs_l.transpose().nonzero(),
            }
        )
        paper_features = torch.FloatTensor(p_vs_t.toarray())  # (4025, 1903)

        pc_p, pc_c = p_vs_c.nonzero()
        paper_labels = np.zeros(len(p_selected), dtype=np.int64)
        for conf_id, label_id in zip(conf_ids, label_ids):
            paper_labels[pc_p[pc_c == conf_id]] = label_id
        paper_labels = torch.from_numpy(paper_labels)

        float_mask = np.zeros(len(pc_p))
        for conf_id in conf_ids:
            pc_c_mask = pc_c == conf_id
            float_mask[pc_c_mask] = np.random.permutation(
                np.linspace(0, 1, pc_c_mask.sum())
            )
        train_idx = np.where(float_mask <= 0.2)[0]
        val_idx = np.where((float_mask > 0.2) & (float_mask <= 0.3))[0]
        test_idx = np.where(float_mask > 0.3)[0]

        num_paper_nodes = self.g.num_nodes("paper")
        train_mask = generate_mask_tensor(idx2mask(train_idx, num_paper_nodes))
        val_mask = generate_mask_tensor(idx2mask(val_idx, num_paper_nodes))
        test_mask = generate_mask_tensor(idx2mask(test_idx, num_paper_nodes))

        self.g.nodes["paper"].data["feat"] = paper_features
        self.g.nodes["paper"].data["label"] = paper_labels
        self.g.nodes["paper"].data["train_mask"] = train_mask
        self.g.nodes["paper"].data["val_mask"] = val_mask
        self.g.nodes["paper"].data["test_mask"] = test_mask

        self.g.multi_update_all(
            {"pa": (fn.copy_u("feat", "m"), fn.mean("m", "feat"))}, "sum"
        )
        self.g.nodes["field"].data["feat"] = torch.eye(self.g.num_nodes("field"))

    def has_cache(self):
        return os.path.exists(
            os.path.join(self.save_path, self.name + "_dgl_graph.bin")
        )

    def score(self):
        return Util.compute_homogeneity(
            self.g, "paper", [("author", "ap"), ("field", "fp")]
        )

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
        return [["pa", "ap"], ["pf", "fp"]]

    @property
    def predict_ntype(self):
        return "paper"
