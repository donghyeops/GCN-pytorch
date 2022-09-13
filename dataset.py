import os
import numpy as np
import torch


class CiteSeer:
    def __init__(self, data_path="datasets/citeseer"):
        super().__init__()
        self.A = None  # adjacency matrix
        self.I = None  # identity matrix
        self.labels = []  # label per node
        self.num_classes = -1
        self._load_data(data_path)

    def _load_data(self, data_path):
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
            self.num_classes = max(self.labels + 1)

    def __call__(self):
        A_tilde = torch.LongTensor(self.A + self.I)
        labels = torch.LongTensor(self.labels)
        return A_tilde, labels


if __name__ == '__main__':
    dataset = CiteSeer()
    A_tilde, labels = dataset()
    print('A_tilde', A_tilde.shape, A_tilde.dtype)
    print('labels', labels.shape, labels.dtype)
