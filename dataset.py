import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split


class CiteSeer:
    def __init__(self, data_path="datasets/citeseer", n_test_data=1000):
        super().__init__()
        self.A = None  # adjacency matrix
        self.I = None  # identity matrix
        self.nodes = []  # node ids
        self.labels = []  # label per node
        self.num_classes = -1
        self._load_data(data_path, n_test_data=n_test_data)

    def _load_data(self, data_path, n_test_data=1000):
        with open(os.path.join(data_path, 'citeseer.edges'), 'r') as f:
            pairs = []
            for line in f:
                node_a, node_b, _ = map(int, line.split(","))
                pairs.append((node_a, node_b))
            max_value = max(max(x) for x in pairs)
            self.A = np.zeros((max_value, max_value))
            self.I = np.eye(max_value)
            for node_a, node_b in pairs:
                self.A[node_a - 1][node_b - 1] = 1
                self.A[node_b - 1][node_a - 1] = 1
        with open(os.path.join(data_path, 'citeseer.node_labels'), 'r') as f:
            for line in f:
                _, label = map(int, line.split(","))
                self.labels.append(label - 1)
            self.labels = np.array(self.labels)
            self.nodes = np.arange(len(self.labels))
            self.num_classes = max(self.labels + 1)

        train_nodes, test_nodes, train_labels, test_labels = train_test_split(
            self.nodes,
            self.labels,
            test_size=n_test_data,
            shuffle=True,
            stratify=self.labels,
            random_state=7770,
        )
        self.train_nodes = train_nodes
        self.train_labels = train_labels
        self.test_nodes = test_nodes
        self.test_labels = test_labels

    def __len__(self):
        return len(self.nodes)

    def __call__(self, split='all'):
        A_tilde = torch.LongTensor(self.A + self.I)
        if split == 'all':
            nodes = torch.LongTensor(self.nodes)
            labels = torch.LongTensor(self.labels)
        elif split == 'train':
            nodes = torch.LongTensor(self.train_nodes)
            labels = torch.LongTensor(self.train_labels)
        elif split == 'test':
            nodes = torch.LongTensor(self.test_nodes)
            labels = torch.LongTensor(self.test_labels)
        else:
            raise ValueError("split value is not in ('all', 'train', 'test')")
        return A_tilde, nodes, labels


if __name__ == '__main__':
    dataset = CiteSeer()
    A_tilde, nodes, labels = dataset(split='test')
    print('A_tilde', A_tilde.shape, A_tilde.dtype)
    print('nodes', nodes.shape, nodes.dtype)
    print('labels', labels.shape, labels.dtype)
