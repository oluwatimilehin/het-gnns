import dgl
import dgl.function as fn
from dgl import DGLGraph

import torch
import torch.nn as nn

from random import randrange, choice
from typing import List, Dict


class SimpleGen:
    @classmethod
    def generate(
        cls,
        n_node_types,
        n_het_edge_types,
        n_nodes_per_type,
        n_edges_per_type,
        n_edges_across_types,
        n_features=5,
        train_split=0.6,
        val_split=0.5,
    ):
        node_types = [f"node_type{i}" for i in range(n_node_types)]
        hom_edge_types = [f"hom_et{i}" for i in range(n_node_types)]

        hom_edges = {}
        for i in range(n_node_types):
            edge_type = (node_types[i], hom_edge_types[i], node_types[i])

            edges = (
                torch.randint(0, n_nodes_per_type, (n_edges_per_type,)),
                torch.randint(0, n_nodes_per_type, (n_edges_per_type,)),
            )

            hom_edges[edge_type] = edges

        het_edges = {}
        for i in range(n_het_edge_types):
            source_index = randrange(n_node_types)
            sink_index = choice([i for i in range(n_node_types) if i != source_index])

            source_node = node_types[source_index]
            dest_node = node_types[sink_index]
            edge_type = (source_node, f"{source_node}-{dest_node}", dest_node)

            edges = (
                torch.randint(0, n_nodes_per_type, (n_edges_across_types,)),
                torch.randint(0, n_nodes_per_type, (n_edges_across_types,)),
            )

            het_edges[edge_type] = edges

        het_graph = dgl.heterograph(
            hom_edges | het_edges,
            num_nodes_dict={ntype: n_nodes_per_type for ntype in node_types},
        )

        het_graph = cls.__fill(
            het_graph,
            n_features=n_features,
            train_split=train_split,
            val_split=val_split,
        )

        return het_graph
    
    @classmethod
    def generate_const_het_edge_types_per_node_type(
        cls,
        n_node_types,
        n_het_edge_types_per_node_type,
        n_nodes_per_type,
        n_edges_per_type,
        n_edges_across_types,
        n_features=5,
        train_split=0.6,
        val_split=0.5,
    ):
        node_types = [f"node_type{i}" for i in range(n_node_types)]
        hom_edge_types = [f"hom_et{i}" for i in range(n_node_types)]

        hom_edges = {}
        for i in range(n_node_types):
            edge_type = (node_types[i], hom_edge_types[i], node_types[i])

            edges = (
                torch.randint(0, n_nodes_per_type, (n_edges_per_type,)),
                torch.randint(0, n_nodes_per_type, (n_edges_per_type,)),
            )

            hom_edges[edge_type] = edges

        het_edges = {}
        for nt in range(n_node_types):
            for i in range(n_het_edge_types_per_node_type):
                sink_index = nt
                source_index = choice([i for i in range(n_node_types) if i != sink_index])
                # source_index = nt
                # sink_index = choice([i for i in range(n_node_types) if i != source_index])

                source_node = node_types[source_index]
                dest_node = node_types[sink_index]
                edge_type = (source_node, f"{source_node}-{dest_node}", dest_node)
                reverse_edge_type = (dest_node, f"{dest_node}-{source_node}", source_node)

                edges = (
                    torch.randint(0, n_nodes_per_type, (n_edges_across_types // 2,)),
                    torch.randint(0, n_nodes_per_type, (n_edges_across_types // 2,)),
                )
                reverse_edges = (
                    torch.randint(0, n_nodes_per_type, (n_edges_across_types // 2,)),
                    torch.randint(0, n_nodes_per_type, (n_edges_across_types // 2,)),
                )

                het_edges[edge_type] = edges
                het_edges[reverse_edge_type] = reverse_edges

        het_graph = dgl.heterograph(
            hom_edges | het_edges,
            num_nodes_dict={ntype: n_nodes_per_type for ntype in node_types},
        )

        het_graph = cls.__fill(
            het_graph,
            n_features=n_features,
            train_split=train_split,
            val_split=val_split,
        )

        return het_graph

    @classmethod
    def label(
        cls,
        hg: DGLGraph,
        n_classes: int,
        node_feat_importance: float,
        hom_edge_importance: float,
        het_edge_importance: float
        # hom_edge_importance_factor: float,  # i.e. hom edges are 'x' times as important as het edges
    ):
        n_features = hg.nodes[hg.ntypes[0]].data["feat"].shape[1]

        node_type_weights = {}
        for n_type in hg.ntypes:
            node_type_weights[n_type] = nn.Linear(n_features, n_classes)

        # For normalizing;
        # Since we 'sum' across edge types on a node, we don't want the het edges for a node that's the target for many heterogeneous edge types to completely overwhelm the homogenous edges
        num_het_edges_per_dest_ntype = {}
        for src_type, _, dest_type in hg.canonical_etypes:
            if src_type != dest_type:
                num_het_edges_per_dest_ntype[dest_type] = (
                    num_het_edges_per_dest_ntype.get(dest_type, 0) + 1
                )

        funcs = {}
        for src_type, etype, dest_type in hg.canonical_etypes:
            edge_weight = nn.Linear(n_features, n_classes)

            node_val = edge_weight(hg.nodes[src_type].data["feat"])
            if src_type == dest_type:
                node_val = hom_edge_importance * node_val
            else:
                node_val = het_edge_importance * node_val / num_het_edges_per_dest_ntype[dest_type]

            dest_node_weights = node_type_weights[dest_type]
            dest_node_feat = hg.nodes[dest_type].data["feat"]

            node_val = node_val + (
                node_feat_importance * dest_node_weights(dest_node_feat)
            )

            hg.nodes[src_type].data["Wh_%s" % etype] = node_val
            funcs[etype] = (
                fn.copy_u("Wh_%s" % etype, "m"),
                fn.mean("m", "h"),
            )  # Take the mean across all messages for an edge type

        def apply(preactivations):
            # Use the maximum aggregated value at each node as the label
            return {"label": torch.argmax(preactivations.data["h"], dim=-1)}

        hg.multi_update_all(funcs, "sum", apply)  # Sum across edge types for each node

        return hg

    @classmethod
    def get_metapaths(cls, hg: DGLGraph, starting_node_type: str) -> List[List[str]]:
        # TODO: This looks at two hops, but we can extend it to varying metapath lengths
        source_to_targets: Dict[str, Dict[str, str]] = {}

        for src_type, etype, dest_type in hg.canonical_etypes:
            if src_type == dest_type:
                continue
            source_to_targets[src_type] = source_to_targets.get(src_type, {}) | {
                dest_type: etype
            }

        dests_for_start_node: Dict[str, str] = source_to_targets.get(
            starting_node_type, {}
        )
        metapaths = []

        for target, etype in dests_for_start_node.items():
            dests_for_target = source_to_targets.get(target, {})

            metapath = [etype]
            if starting_node_type in dests_for_target:
                metapath.append(
                    dests_for_target[starting_node_type]
                )  # Get the edge that points to the starting node

                metapaths.append(metapath)
        
        print("metapaths", metapaths)
        return metapaths

    # @classmethod
    # def get_metapaths(cls, hg: DGLGraph, starting_node_type: str) -> List[List[str]]:
    #     # TODO: This looks at two hops, but we can extend it to varying metapath lengths
    #     # source_to_targets: Dict[str, Dict[str, str]] = {}

    #     # for src_type, etype, dest_type in hg.canonical_etypes:
    #     #     if src_type == dest_type:
    #     #         continue
    #     #     source_to_targets[src_type] = source_to_targets.get(src_type, {}) | {
    #     #         dest_type: etype
    #     #     }

    #     # dests_for_start_node: Dict[str, str] = source_to_targets.get(
    #     #     starting_node_type, {}
    #     # )
    #     # metapaths = []

    #     # for target, etype in dests_for_start_node.items():
    #     #     dests_for_target = source_to_targets.get(target, {})

    #     #     metapath = [etype]
    #     #     if starting_node_type in dests_for_target:
    #     #         metapath.append(
    #     #             dests_for_target[starting_node_type]
    #     #         )  # Get the edge that points to the starting node

    #     #         metapaths.append(metapath)
        
    #     # print("metapaths", metapaths)

    #     metapaths = []
    #     # for simplicity, we just have metapaths be exactly the edge types. Can extend
    #     for src_type, etype, dest_type in hg.canonical_etypes:
    #         metapaths.append([etype])
    #     return metapaths

    @classmethod
    def __fill(
        cls, hg: DGLGraph, n_features: int, train_split: float, val_split: float
    ):
        for ntype in hg.ntypes:
            num_nodes = hg.num_nodes(ntype)

            # here we generate normally distributed random features
            # hg.nodes[ntype].data["feat"] = torch.randn(num_nodes, n_features)

            # to try to get a bit better signal, especially as the weight matrices are random, doing one hot category for input feature
            indices = torch.randint(0, n_features, (num_nodes, ))
            hg.nodes[ntype].data["feat"] = nn.functional.one_hot(indices, n_features).float()

            train_mask = torch.bernoulli(torch.ones(num_nodes) * train_split)
            inv_train_mask = 1 - train_mask

            val_mask = torch.bernoulli(inv_train_mask * val_split)
            test_mask = inv_train_mask - val_mask

            hg.nodes[ntype].data["train_mask"] = train_mask.to(torch.bool)
            hg.nodes[ntype].data["val_mask"] = train_mask.to(torch.bool)
            hg.nodes[ntype].data["test_mask"] = test_mask.to(torch.bool)

        return hg
