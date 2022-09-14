import os
from collections import defaultdict
import numpy as np
import torch
from sklearn.model_selection import train_test_split


class CiteSeer:
    def __init__(self, data_path="datasets/citeseer", n_train_data_per_class=20, n_test_data=1000):
        super().__init__()
        self.A = None  # adjacency matrix
        self.I = None  # identity matrix
        self.nodes = []  # node ids
        self.labels = []  # label per node
        self.feats = []  # bag-of-words per node

        self.num_classes = -1
        self.num_edges = 0
        self.num_nodes = 0
        self._load_data(data_path, n_train_data_per_class=n_train_data_per_class, n_test_data=n_test_data)

    def _load_data(self, data_path, n_train_data_per_class=20, n_test_data=1000):
        with open(os.path.join(data_path, 'citeseer.content'), 'r') as f:
            label_to_id = {}
            name_to_id = {}
            for line in f:
                raw = line.strip().split("\t")
                node_str = raw[0]
                label_str = raw[-1]
                feat = list(map(int, raw[1:-1]))
                if label_str not in label_to_id:
                    label_to_id[label_str] = len(label_to_id)
                label = label_to_id[label_str]

                if node_str not in name_to_id:
                    name_to_id[node_str] = len(name_to_id)
                node = name_to_id[node_str]
                self.nodes.append(node)
                self.labels.append(label)
                self.feats.append(feat)
            self.labels = np.array(self.labels)
            self.nodes = np.array(self.nodes)
            self.feats = np.array(self.feats)
            self.num_nodes = len(self.nodes)
            self.num_classes = max(self.labels + 1)

        with open(os.path.join(data_path, 'citeseer.cites'), 'r') as f:
            pairs = []
            for line in f:
                node_a, node_b = line.strip().split("\t")
                try:
                    node_a = name_to_id[node_a]
                    node_b = name_to_id[node_b]
                except KeyError:  # no label/feature
                    continue
                if node_a == node_b:  # pass self-edge
                    continue
                pairs.append((node_a, node_b))
            self.A = np.zeros((len(name_to_id), len(name_to_id)))
            self.I = np.eye(len(name_to_id))
            for node_a, node_b in pairs:
                self.A[node_a][node_b] = 1
                self.A[node_b][node_a] = 1
            self.num_edges = len(pairs)

        train_nodes, test_nodes, train_feats, test_feats, train_labels, test_labels = train_test_split(
            self.nodes,
            self.feats,
            self.labels,
            test_size=n_test_data,
            shuffle=True,
            stratify=self.labels,
            random_state=7770,
        )
        indices_per_label = defaultdict(list)
        for idx, label in enumerate(train_labels):
            if len(indices_per_label[label]) == n_train_data_per_class:
                continue
            indices_per_label[label].append(idx)
        train_indices = []
        for indices in indices_per_label.values():
            train_indices.extend(indices)
        train_nodes = train_nodes[train_indices]
        train_feats = train_feats[train_indices]
        train_labels = train_labels[train_indices]

        self.train_nodes = train_nodes
        self.train_feats = train_feats
        self.train_labels = train_labels
        self.test_nodes = test_nodes
        self.test_feats = test_feats
        self.test_labels = test_labels

    def __len__(self):
        return len(self.nodes)

    def __call__(self, split='all'):
        A_tilde = torch.LongTensor(self.A + self.I)
        if split == 'all':
            nodes = torch.LongTensor(self.nodes)
            feats = torch.FloatTensor(self.feats)
            labels = torch.LongTensor(self.labels)
        elif split == 'train':
            nodes = torch.LongTensor(self.train_nodes)
            feats = torch.FloatTensor(self.feats)
            labels = torch.LongTensor(self.train_labels)
        elif split == 'test':
            nodes = torch.LongTensor(self.test_nodes)
            feats = torch.FloatTensor(self.feats)
            labels = torch.LongTensor(self.test_labels)
        else:
            raise ValueError("split value is not in ('all', 'train', 'test')")
        return A_tilde, nodes, feats, labels


if __name__ == '__main__':
    dataset = CiteSeer()
    A_tilde, nodes, feats, labels = dataset(split='test')
    print('A_tilde', A_tilde.shape, A_tilde.dtype)
    print('nodes', nodes.shape, nodes.dtype)
    print('feats', feats.shape, feats.dtype)
    print('labels', labels.shape, labels.dtype)
