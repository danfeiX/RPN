import numpy as np
from rpn.utils.torch_utils import to_tensor, to_numpy
import torch

try:
    from torch_geometric.data import Data, batch
except ImportError as e:
    print('Warning: cannot import torch_geometric. This does not affect the core functionality of this repo.')


def add_supernodes(node_index, edge_index, supernode_clique):
    """
    Add supernodes to an existing graph defined by node_index and edge_index.
    Supernodes are defined by supernode_clique, which is a list of cliques (set of nodes)

    :param node_index: [N]
    :param edge_index: [E, 2]
    :param supernode_clique: [SN, C]
    :return: new_node_index: [N + SN], new_edge_index: [E + SN * (SN - 1) + C * 2 * SN]
    """
    num_sn = len(supernode_clique)
    sn_start_idx = node_index.max() + 1
    sn_node_index = np.arange(num_sn) + sn_start_idx
    sn_edge_index = [edge_index]

    # add bi-directional edge to the supernode from each node in the clique
    for sni, snc in zip(sn_node_index, supernode_clique):
        clique_edge = np.zeros((len(snc) * 2, 2), dtype=np.int64)
        clique_edge[:len(snc), 0] = snc
        clique_edge[len(snc):, 0] = sni
        clique_edge[:len(snc), 1] = sni
        clique_edge[len(snc):, 1] = snc
        sn_edge_index.append(clique_edge)

    # add connections among the supernodes
    sn_edge_index.append(fully_connected_edges(sn_node_index))
    return np.concatenate([node_index, sn_node_index]), np.concatenate(sn_edge_index)


def fully_connected_edges(node_index, self_connection=False):
    """
    Return fully connected edges (no self-connection)
    :param node_index: node indices
    :param self_connection:
    :return: [N * (N - 1), 2]
    """
    n = len(node_index)
    if not self_connection:
        edges = np.zeros([n * (n - 1), 2], dtype=np.int64)
    else:
        edges = np.zeros([n ** 2, 2], dtype=np.int64)
    count = 0
    for r in range(n):
        for c in range(n):
            if r != c or self_connection:
                edges[count, :] = [r, c]
                count += 1
    return edges


def split_graph_feature(node_feat, edge_feat, node_index_list, edge_index_list):
    """
    Split batched node and edge features (graph features) to individual lists
    :param node_feat: torch.Tensor of shape [N1 + N2 + ..., D1]
    :param edge_feat: torch.Tensor of shape [E1 + E2 + ..., D2]
    :param node_index_list: a list of node indices, in the form of numpy array
    :param edge_index_list: a list of edge indices, in the form of numpy array
    :return: node_feat_least: [[N1, D1], [N2, D1], ...], edge_feat_list: [[E1, D2], [E2, D2], ...]
    """
    node_feat_list = split_clique_feature(node_feat, node_index_list)
    edge_feat_list = split_clique_feature(edge_feat, edge_index_list)
    return node_feat_list, edge_feat_list


def split_clique_feature(clique_feat, clique_index_list):
    assert(isinstance(clique_index_list, (tuple, list)))
    num_element = [e.shape[0] for e in clique_index_list]
    assert(clique_feat.size(0) == np.sum(num_element))
    clique_feat_list = clique_feat.split(num_element, dim=0)
    return clique_feat_list


def collate_torch_graphs(node_feat, edge_feat, node_index_list, edge_index_list):
    """
    Collate a list of graphs and their features.

    :param node_feat: torch.Tensor of shape [N1 + N2 + ..., D1]
    :param edge_feat: torch.Tensor of shape [E1 + E2 + ..., D2]
    :param node_index_list: a list of node indices, in the form of numpy array
    :param edge_index_list: a list of edge indices, in the form of numpy array
    :return: a collated graph of type torch.geometric.data.Data
    """

    node_feat_list, edge_feat_list = split_graph_feature(node_feat, edge_feat, node_index_list, edge_index_list)

    graphs = []
    # TODO: vectorize this
    for nf, ef, n_idx, e_idx in zip(node_feat_list, edge_feat_list, node_index_list, edge_index_list):
        # add supernode to the graph
        supernode_clique = np.tile(n_idx[None, ...], (len(e_idx), 1))
        sn_n_idx, sn_e_idx = add_supernodes(n_idx, e_idx, supernode_clique)
        sn_feat = torch.cat([nf, ef], dim=0)
        torch_e_idx = to_tensor(sn_e_idx).long().t().contiguous().to(node_feat.device)
        graphs.append(Data(x=sn_feat, edge_index=torch_e_idx))

    batched_graphs = batch.Batch.from_data_list(graphs)

    num_node = [n.shape[0] for n in node_index_list]
    num_edge = [e.shape[0] for e in edge_index_list]
    assert(batched_graphs.x.shape[0] == (np.sum(num_node) + np.sum(num_edge)))

    return batched_graphs


def separate_graph_collated_features(collated_feat, node_index_list, edge_index_list):
    """
    Separate a collated feature by a list of graphs
    :param collated_feat: feature of shape [N + E, D]
    :param node_index_list: a list of node index
    :param edge_index_list: a list of edge index
    :return: separated node and edge features of shape [N, D] and [E, D], respectively
    """
    num_node = [n.shape[0] for n in node_index_list]
    num_edge = [e.shape[0] for e in edge_index_list]
    num_feat = np.sum(num_node) + np.sum(num_edge)
    assert(collated_feat.size(0) == num_feat)
    num_feat_list = [None] * (len(num_node) + len(num_edge))
    num_feat_list[::2] = num_node
    num_feat_list[1::2] = num_edge
    feat_list = collated_feat.split(num_feat_list, dim=0)
    node_feat = torch.cat(feat_list[::2], dim=0)
    edge_feat = torch.cat(feat_list[1::2], dim=0)
    assert(node_feat.shape[0] == np.sum(num_node))
    assert(edge_feat.shape[0] == np.sum(num_edge))
    return node_feat, edge_feat


def test_graph_collation():
    node_index, edge_index = construct_full_graph(5)
    node_input = torch.randn(10, 10)
    edge_input = [get_edge_features(node_input[:5], edge_index, lambda a, b: b - a),
                  get_edge_features(node_input[:5], edge_index, lambda a, b: b - a)]
    edge_input = torch.cat(edge_input, dim=0)
    node_index = [node_index, node_index]
    edge_index = [edge_index, edge_index]
    gs = collate_torch_graphs(node_input, edge_input, node_index, edge_index)
    ni, ei = separate_graph_collated_features(gs.x, node_index, edge_index)
    assert(to_numpy(torch.all(ei == edge_input)) == 1)
    assert(to_numpy(torch.all(ni == node_input)) == 1)


def construct_full_graph(num_objects, self_connection=False):
    node_index = np.arange(num_objects)
    edge_index = fully_connected_edges(node_index, self_connection=self_connection)
    return node_index, edge_index


def get_edge_features(node_features, edge_index, feature_func):
    return feature_func(node_features[edge_index[:, 0], ...], node_features[edge_index[:, 1], ...])


def main():
    node_idx = np.array([0, 1, 2], dtype=np.int64)
    edge_idx = np.array([[0, 1], [0, 2]], dtype=np.int64)
    supernode_clique = np.tile(node_idx[None, ...], (len(edge_idx), 1))
    new_node_idx, new_edge_idx = add_supernodes(node_idx, edge_idx, supernode_clique)
    node_features = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    edge_feat = get_edge_features(node_features, edge_idx, lambda a, b: b - a)
    print()


if __name__ == '__main__':
    main()