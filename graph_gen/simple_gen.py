import dgl
import torch
import random
from random import randrange

#Plan for graph generation:
# Give the number of node types, number of edge types
# Also give the let's say homogeneous sparsity factor and the heterogeneous sparsity factor---number of edges within node type and betweeen node types
# Also give the graph generator for homogeneous and for heterogeneous
# Then we will generate essentially using the generators all the edges and stuff
# We will just give some random node labels, and that should be the graph

def simple_gen(n_node_types, n_hetero_edge_types, n_nodes_per_type, n_edges_per_type, n_edges_across_types):
    node_type_names = ["nt"+str(i) for i in range(n_node_types)]
    homo_edge_type_names = ["hom_et"+str(i) for i in range(n_node_types)]
    hetero_edge_type_names = ["het_et"+str(i) for i in range(n_hetero_edge_types)]
    
    # homo_edge_type_tuples = [(node_type_names[i], homo_edge_type_names[i], node_type_names[i]) for i in range(n_node_types)]
    all_edges = {}

    # adding homogeneous graph edges
    for i in range(n_node_types):
        edge_type_tuple = (node_type_names[i], homo_edge_type_names[i], node_type_names[i])
        edges = torch.randint(0, n_nodes_per_type, (n_edges_per_type,2))
        all_edges[edge_type_tuple] = edges

    # hetero_edge_nodes = [[randrange(n_node_types), randrange(n_node_types)] for _ in range(n_hetero_edge_types)]
    # hetero_edge_type_tuples = [(node_type_names[hetero_edge_nodes[i]], hetero_edge_type_names[i], node_type_names[hetero_edge_nodes[i]]) for i in range(n_hetero_edge_types)]

    # all_edge_type_tuples = homo_edge_type_tuples + hetero_edge_type_tuples

    # adding heterogeneous graph edges
    for i in range(n_hetero_edge_types):
        source_node_type = randrange(n_node_types)
        sink_node_type = randrange(n_node_types)
        edge_type_tuple = (source_node_type, hetero_edge_type_names[i], sink_node_type)
        # technically fine to do it like this as all node types have same # nodes
        # If we have different numbers of nodes per node type (which would make sense)
        # Then split this into two different vector that then just concat
        edges = torch.randint(0, n_nodes_per_type, (n_edges_per_type,2))
        all_edges[edge_type_tuple] = edges

    hetero_graph = dgl.heterograph(
        all_edges
    )

    return node_type_names, homo_edge_type_names, hetero_edge_type_names, hetero_graph

#val_split represents the percent of non-training nodes that are validation
#So val_prob from the start is train*val, and then test is train*(1-val)
def simple_fill(feat_dim, node_type_names, hetero_graph, train_split=0.6, val_split=0.5):
    for node_type in node_type_names:
        num_nodes = hetero_graph.num_nodes(node_type)
        hetero_graph.nodes[node_type].data["feat"] = torch.randn(num_nodes, feat_dim)
        # hetero_graph.nodes[node_type].data["label"] = torch.randint(
        #     0, num_classes, (num_nodes,)
        # )
        train_mask = torch.bernoulli(torch.ones(num_nodes)*train_split)
        inv_train_mask = 1-train_mask
        val_mask = torch.bernoulli(inv_train_mask*val_split)
        test_mask = inv_train_mask-val_mask
        hetero_graph.nodes[node_type].data["train_mask"] = train_mask
        hetero_graph.nodes[node_type].data["val_mask"] = val_mask
        hetero_graph.nodes[node_type].data["test_mask"] = test_mask

# Lol shouldn't write from scratch should use dgl
# def simple_label(hetero_graph, n_labels, node_type_names, homo_edge_type_names, hetero_edge_type_names, node_importance, homo_neighbor_importance, hetero_neighbor_importance):
#     feat_dim = hetero_graph.nodes[node_type_names[0]].data["feat"].shape[1]
#     node_labels = {}
#     hetero_node_labels = {}
#     hetero_node_counts = {}
#     for node_type in node_type_names:
#         node_labeller = torch.nn.Linear(feat_dim, n_labels)
#         node_labels[node_type] = node_importance * node_labeller(hetero_graph.nodes[node_type].data["feat"])

#         edge_labeler = torch.nn.Linear(2*feat_dim, n_labels)
#         edge_labels = edge_labeler()

#         num_nodes = hetero_graph.num_nodes(node_type)
#         hetero_node_labels[node_type] = torch.zeros(num_nodes, n_labels)
#         hetero_node_counts[node_type] = 0
    