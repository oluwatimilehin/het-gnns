import graph_gen.simple_gen as sg
from trainers.HGTTrainer import HGTTrainer

if __name__ == "__main__":
    n_hetero_features = 20
    n_user_classes = 5

    node_type_names, homo_edge_type_names, hetero_edge_type_names, hetero_graph = sg.simple_gen(3, 4, 100, 500, 200)
    hetero_graph = sg.simple_fill(n_hetero_features, node_type_names, hetero_graph, train_split=0.6, val_split=0.5)
    hetero_graph = sg.simple_label(hetero_graph, n_user_classes, 1, 1, 1)

    hgt_trainer = HGTTrainer(
        hetero_graph,
        input_dim=n_hetero_features,
        output_dim=n_user_classes,
    )
    hgt_trainer.run(predicted_node_type="nt0")