import torch
import torch.nn as nn
import torch.nn.functional as F


def get_gcn_a_hat(A_tilde):
    D_tilde = A_tilde.sum(axis=1) ** -0.5
    D_tilde = torch.nan_to_num(D_tilde, posinf=0)
    D_tilde = torch.diagflat(D_tilde)
    A_hat = torch.matmul(torch.matmul(D_tilde, A_tilde), D_tilde)
    return A_hat


class GCNLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        activation=F.relu,
        dropout=0.5,
        residual=False,
    ):
        super().__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.residual = residual and (in_features == out_features)

    def forward(self, x, A_hat):
        feat = self.dropout(x)
        feat = torch.matmul(A_hat, feat)
        feat = self.fc(feat)
        feat = self.activation(feat)

        if self.residual:
            x = x + feat
        else:
            x = feat
        return x


class GCNModel(nn.Module):
    def __init__(self, in_dim=3703, hidden_dim=16, n_layer=2, n_class=6, dropout=0.5, residual=True, embedding=False, n_node=None):
        super().__init__()
        self.embedding = embedding
        if embedding:
            self.n_node = n_node
            self.embedding = nn.Embedding(n_node, embedding_dim=in_dim)
        self.gcn_layers = nn.ModuleList()
        for i in range(n_layer):
            if i == n_layer - 1:  # last
                _in_dim = hidden_dim if i > 0 else in_dim
                _out_dim = n_class
                _dropout = dropout
                _activation = nn.Identity()
            elif i == 0:
                _in_dim = in_dim
                _out_dim = hidden_dim
                _dropout = dropout
                _activation = nn.ReLU()
            else:
                _in_dim = hidden_dim
                _out_dim = hidden_dim
                _dropout = 0
                _activation = nn.ReLU()
            self.gcn_layers.append(
                GCNLayer(
                    in_features=_in_dim,
                    out_features=_out_dim,
                    dropout=_dropout,
                    activation=_activation,
                    residual=residual
                )
            )

    def forward(self, A_tilde, x):
        if self.embedding:
            x = torch.arange(0, self.n_node, dtype=torch.long)
            x = x.to(A_tilde.device)
            x = self.embedding(x)
        A_hat = get_gcn_a_hat(A_tilde)
        for gcn_layer in self.gcn_layers:
            x = gcn_layer(x, A_hat)
        return x


if __name__ == '__main__':
    model = GCNModel()
    A_tilde = torch.randint(2, (5, 5)).float()
    A_tilde = torch.randint(2, (5, 5)).float()
    logits = model(A_tilde)
    print(logits.shape)
