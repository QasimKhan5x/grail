from torch.utils.data import Dataset
import timeit
import os
import logging
import lmdb
import numpy as np
import json
import pickle
import dgl
from utils.graph_utils import ssp_multigraph_to_dgl, incidence_matrix
from utils.data_utils import process_files, save_to_file, plot_rel_dist
from .graph_sampler import *
import pdb


def generate_subgraph_datasets(
    params, splits=["train", "valid"], saved_relation2id=None, max_label_value=None
):

    testing = "test" in splits
    adj_list, triplets, entity2id, relation2id, id2entity, id2relation = process_files(
        params.file_paths, saved_relation2id
    )

    # plot_rel_dist(adj_list, os.path.join(params.main_dir, f'data/{params.dataset}/rel_dist.png'))

    data_path = os.path.join(params.data_dir, f"data/{params.dataset}/relation2id.json")
    if not os.path.isdir(data_path) and not testing:
        with open(data_path, "w") as f:
            json.dump(relation2id, f)

    graphs = {}

    for split_name in splits:
        graphs[split_name] = {
            "triplets": triplets[split_name],
            "max_size": params.max_links,
        }

    # Sample train and valid/test links
    for split_name, split in graphs.items():
        logging.info(f"Sampling negative links for {split_name}")
        split["pos"], split["neg"] = sample_neg(
            adj_list,
            split["triplets"],
            params.num_neg_samples_per_link,
            max_size=split["max_size"],
            constrained_neg_prob=params.constrained_neg_prob,
        )

    if testing:
        directory = os.path.join(params.main_dir, "data/{}/".format(params.dataset))
        save_to_file(
            directory,
            f"neg_{params.test_file}_{params.constrained_neg_prob}.txt",
            graphs["test"]["neg"],
            id2entity,
            id2relation,
        )

    links2subgraphs(adj_list, graphs, params, max_label_value)


def get_kge_embeddings(dataset, kge_model):
    path = "./experiments/kge_baselines/{}_{}".format(kge_model, dataset)
    node_features = np.load(os.path.join(path, "entity_embedding.npy"))
    with open(os.path.join(path, "id2entity.json")) as json_file:
        kge_id2entity = json.load(json_file)
        kge_entity2id = {v: int(k) for k, v in kge_id2entity.items()}

    return node_features, kge_entity2id

class SubgraphDataset(Dataset):
    """Extracted, labeled, subgraph dataset -- DGL Only"""

    def __init__(
        self,
        db_path,
        db_name_pos,
        db_name_neg,
        raw_data_paths,
        included_relations=None,
        add_traspose_rels=False,
        num_neg_samples_per_link=1,
        use_kge_embeddings=False,
        dataset="",
        kge_model="",
        file_name="",
    ):

        # Store all parameters needed to initialize later
        self.db_path = db_path
        self.db_name_pos = db_name_pos
        self.db_name_neg = db_name_neg
        self.raw_data_paths = raw_data_paths
        self.included_relations = included_relations
        self.add_traspose_rels = add_traspose_rels
        self.num_neg_samples_per_link = num_neg_samples_per_link
        self.use_kge_embeddings = use_kge_embeddings
        self.dataset = dataset
        self.kge_model = kge_model
        self.file_name = file_name

        # Initialize LMDB related attributes to None
        self.main_env = None
        self.db_pos = None
        self.db_neg = None

        # Initialize other attributes that do not require LMDB
        self.node_features, self.kge_entity2id = (
            get_kge_embeddings(dataset, kge_model)
            if use_kge_embeddings
            else (None, None)
        )

        # Process files to get graph-related data
        ssp_graph, triplets, entity2id, relation2id, id2entity, id2relation = process_files(
            raw_data_paths, included_relations
        )
        self.num_rels = len(ssp_graph)

        # Add transpose matrices to handle both directions of relations.
        if add_traspose_rels:
            ssp_graph_t = [adj.T for adj in ssp_graph]
            ssp_graph += ssp_graph_t

        # The effective number of relations after adding symmetric adjacency matrices and/or self connections
        self.aug_num_rels = len(ssp_graph)
        self.graph = ssp_multigraph_to_dgl(ssp_graph)
        self.ssp_graph = ssp_graph
        self.id2entity = id2entity
        self.id2relation = id2relation

        # Initialize stats to None; they will be loaded from LMDB
        self.max_n_label = np.array([0, 0])
        self.avg_subgraph_size = None
        self.min_subgraph_size = None
        self.max_subgraph_size = None
        self.std_subgraph_size = None
        self.avg_enc_ratio = None
        self.min_enc_ratio = None
        self.max_enc_ratio = None
        self.std_enc_ratio = None
        self.avg_num_pruned_nodes = None
        self.min_num_pruned_nodes = None
        self.max_num_pruned_nodes = None
        self.std_num_pruned_nodes = None

        # Initialize LMDB and load stats
        self._initialize_lmdb()
        self._load_stats()

        # Set n_feat_dim based on max_n_label and node_features
        label_feat_dim = self.max_n_label[0] + 1 + self.max_n_label[1] + 1
        node_feat_dim = self.node_features.shape[1] if self.node_features is not None else 0
        self.n_feat_dim = label_feat_dim + node_feat_dim

        logging.info(
            f"Max distance from sub : {self.max_n_label[0]}, Max distance from obj : {self.max_n_label[1]}"
        )

        # Load num_graphs_pos and num_graphs_neg from LMDB
        with self.main_env.begin(db=self.db_pos) as txn:
            self.num_graphs_pos = int.from_bytes(
                txn.get("num_graphs".encode()), byteorder="little"
            )
        with self.main_env.begin(db=self.db_neg) as txn:
            self.num_graphs_neg = int.from_bytes(
                txn.get("num_graphs".encode()), byteorder="little"
            )

    def _initialize_lmdb(self):
        if self.main_env is None:
            self.main_env = lmdb.open(
                self.db_path, readonly=True, max_dbs=3, lock=False
            )
            self.db_pos = self.main_env.open_db(self.db_name_pos.encode())
            self.db_neg = self.main_env.open_db(self.db_name_neg.encode())

    def _load_stats(self):
        with self.main_env.begin() as txn:
            self.max_n_label[0] = int.from_bytes(
                txn.get("max_n_label_sub".encode()), byteorder="little"
            )
            self.max_n_label[1] = int.from_bytes(
                txn.get("max_n_label_obj".encode()), byteorder="little"
            )

            self.avg_subgraph_size = struct.unpack(
                "f", txn.get("avg_subgraph_size".encode())
            )[0]
            self.min_subgraph_size = struct.unpack(
                "f", txn.get("min_subgraph_size".encode())
            )[0]
            self.max_subgraph_size = struct.unpack(
                "f", txn.get("max_subgraph_size".encode())
            )[0]
            self.std_subgraph_size = struct.unpack(
                "f", txn.get("std_subgraph_size".encode())
            )[0]

            self.avg_enc_ratio = struct.unpack("f", txn.get("avg_enc_ratio".encode()))[0]
            self.min_enc_ratio = struct.unpack("f", txn.get("min_enc_ratio".encode()))[0]
            self.max_enc_ratio = struct.unpack("f", txn.get("max_enc_ratio".encode()))[0]
            self.std_enc_ratio = struct.unpack("f", txn.get("std_enc_ratio".encode()))[0]

            self.avg_num_pruned_nodes = struct.unpack(
                "f", txn.get("avg_num_pruned_nodes".encode())
            )[0]
            self.min_num_pruned_nodes = struct.unpack(
                "f", txn.get("min_num_pruned_nodes".encode())
            )[0]
            self.max_num_pruned_nodes = struct.unpack(
                "f", txn.get("max_num_pruned_nodes".encode())
            )[0]
            self.std_num_pruned_nodes = struct.unpack(
                "f", txn.get("std_num_pruned_nodes".encode())
            )[0]

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove LMDB environment and DB handles from the state to prevent pickling
        state['main_env'] = None
        state['db_pos'] = None
        state['db_neg'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reinitialize LMDB environment and load stats
        self._initialize_lmdb()
        self._load_stats()
        # Recompute n_feat_dim after reloading stats
        label_feat_dim = self.max_n_label[0] + 1 + self.max_n_label[1] + 1
        node_feat_dim = self.node_features.shape[1] if self.node_features is not None else 0
        self.n_feat_dim = label_feat_dim + node_feat_dim

    def __getitem__(self, index):
        # Ensure LMDB is initialized
        self._initialize_lmdb()
        with self.main_env.begin(db=self.db_pos) as txn:
            str_id = "{:08}".format(index).encode("ascii")
            serialized_data = txn.get(str_id)
            if serialized_data is None:
                raise KeyError(f"No data found for index {index}")
            deserialized = deserialize(serialized_data)
            nodes_pos, r_label_pos, g_label_pos, n_labels_pos = deserialized.values()
            subgraph_pos = self._prepare_subgraphs(nodes_pos, r_label_pos, n_labels_pos)

        subgraphs_neg = []
        r_labels_neg = []
        g_labels_neg = []
        with self.main_env.begin(db=self.db_neg) as txn:
            for i in range(self.num_neg_samples_per_link):
                neg_index = index + i * self.num_graphs_pos
                str_id = "{:08}".format(neg_index).encode("ascii")
                serialized_neg = txn.get(str_id)
                if serialized_neg is None:
                    raise KeyError(f"No negative data found for index {neg_index}")
                deserialized_neg = deserialize(serialized_neg)
                nodes_neg, r_label_neg, g_label_neg, n_labels_neg = deserialized_neg.values()
                subgraphs_neg.append(
                    self._prepare_subgraphs(nodes_neg, r_label_neg, n_labels_neg)
                )
                r_labels_neg.append(r_label_neg)
                g_labels_neg.append(g_label_neg)

        return (
            subgraph_pos,
            g_label_pos,
            r_label_pos,
            subgraphs_neg,
            g_labels_neg,
            r_labels_neg,
        )

    def __len__(self):
        return self.num_graphs_pos

    def _prepare_subgraphs(self, nodes, r_label, n_labels):
        # Step 1: Create the subgraph from the given nodes
        subgraph = self.graph.subgraph(nodes, store_ids=True)

        # Step 2: Assign edge types using the original edge IDs stored in subgraph.edata[dgl.EID]
        subgraph.edata["type"] = self.graph.edata["type"][subgraph.edata[dgl.EID]]

        # Step 3: Assign edge labels, ensuring the tensor is on the same device as the subgraph
        subgraph.edata["label"] = torch.full(
            (subgraph.number_of_edges(),),
            r_label,
            dtype=torch.long,
            device=subgraph.device,
        )

        # Step 4: Ensure that the subgraph has at least one edge between the root nodes with the specified relation
        # Assumption: The first two nodes in `nodes` are the root nodes and correspond to subgraph nodes 0 and 1
        if len(nodes) < 2:
            raise ValueError(
                "The `nodes` list must contain at least two nodes representing the roots."
            )

        root_subgraph_nodes = [0, 1]  # Assuming nodes[0] and nodes[1] are the roots

        # Verify that the subgraph has nodes 0 and 1
        if subgraph.number_of_nodes() < 2:
            raise ValueError(
                "Subgraph must contain at least two nodes for root relations."
            )

        # Check if an edge exists between root nodes (0 and 1)
        if subgraph.has_edges_between(root_subgraph_nodes[0], root_subgraph_nodes[1]):
            # Retrieve all edge IDs between root nodes
            edges_btw_roots = subgraph.edge_ids(
                root_subgraph_nodes[0], root_subgraph_nodes[1]
            )

            # Check if any of these edges have the specified relation label
            rel_link = (subgraph.edata["type"][edges_btw_roots] == r_label).nonzero(
                as_tuple=True
            )[0]
        else:
            rel_link = torch.tensor([], dtype=torch.long, device=subgraph.device)

        # If no such edge exists, add one and assign its type and label
        if rel_link.numel() == 0:
            # Add a new edge from node 0 to node 1
            subgraph = dgl.add_edges(
                subgraph,
                torch.tensor([root_subgraph_nodes[0]], device=subgraph.device),
                torch.tensor([root_subgraph_nodes[1]], device=subgraph.device),
            )

            # Assign the relation label to the new edge
            # The new edge is the last edge in the edge list
            new_edge_id = subgraph.number_of_edges() - 1
            subgraph.edata["type"][new_edge_id] = r_label
            subgraph.edata["label"][new_edge_id] = r_label

        # Step 5: Map the IDs read by GraIL to the entity IDs as registered by the KGE embeddings
        if self.kge_entity2id:
            try:
                kge_nodes = [self.kge_entity2id[self.id2entity[n]] for n in nodes]
            except KeyError as e:
                raise KeyError(
                    f"Entity mapping failed for node {e.args[0]}. Check `id2entity` and `kge_entity2id` mappings."
                )
        else:
            kge_nodes = None

        # Step 6: Retrieve node features if available
        if self.node_features is not None and kge_nodes is not None:
            try:
                n_feats = self.node_features[kge_nodes]
            except IndexError as e:
                raise IndexError(
                    f"Node feature retrieval failed for node indices {kge_nodes}."
                )
        else:
            n_feats = None

        # Step 7: Prepare additional features using the updated subgraph
        subgraph = self._prepare_features_new(subgraph, n_labels, n_feats)

        return subgraph

    def _prepare_features_new(self, subgraph, n_labels, n_feats=None):
        # One hot encode the node label feature and concat to node features
        n_nodes = subgraph.number_of_nodes()
        label_feats = np.zeros(
            (n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1)
        )
        label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
        label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1

        n_feats = (
            np.concatenate((label_feats, n_feats), axis=1)
            if n_feats is not None
            else label_feats
        )
        subgraph.ndata["feat"] = torch.FloatTensor(n_feats)

        head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels]).flatten()
        tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels]).flatten()
        n_ids = np.zeros(n_nodes)
        n_ids[head_id] = 1  # head
        n_ids[tail_id] = 2  # tail
        subgraph.ndata["id"] = torch.FloatTensor(n_ids)

        # self.n_feat_dim is already set in __init__, no need to set here
        return subgraph

    # If you have _prepare_features, you can keep it or remove it based on your usage
    def _prepare_features(self, subgraph, n_labels, n_feats=None):
        # One hot encode the node label feature and concat to node features
        n_nodes = subgraph.number_of_nodes()
        label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1))
        label_feats[np.arange(n_nodes), n_labels] = 1
        label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1
        n_feats = (
            np.concatenate((label_feats, n_feats), axis=1) if n_feats else label_feats
        )
        subgraph.ndata["feat"] = torch.FloatTensor(n_feats)
        self.n_feat_dim = n_feats.shape[
            1
        ]  # Find cleaner way to do this -- i.e. set the n_feat_dim
        return subgraph
