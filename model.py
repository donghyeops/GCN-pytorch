import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        activation=F.relu,
        dropout=0.5
    ):
        super().__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, A_tilde):
        D_tilde = A_tilde.sum(axis=1) ** -0.5
        D_tilde = torch.nan_to_num(D_tilde, posinf=0)
        D_tilde = torch.diagflat(D_tilde)
        DAD = torch.matmul(torch.matmul(D_tilde, A_tilde), D_tilde)

        HW = self.fc(x)
        x = torch.matmul(DAD, HW)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class GCNModel(nn.Module):
    def __init__(self, n_node, n_feature=32, n_layer=2, n_class=10):
        super().__init__()
        self.n_node = n_node
        self.embedding = nn.Embedding(n_node, embedding_dim=n_feature)
        self.gcn_layers = nn.ModuleList()
        for i in range(n_layer):
            self.gcn_layers.append(GCNLayer(n_feature, n_feature))
        self.classifier = nn.Linear(n_feature, n_class)

    def forward(self, A_tilde):
        x = torch.arange(0, self.n_node, dtype=torch.long)
        x = x.to(A_tilde.device)
        feats = self.embedding(x)

        for gcn_layer in self.gcn_layers:
            feats = gcn_layer(feats, A_tilde)
        logits = self.classifier(feats)
        return logits


if __name__ == '__main__':
    model = GCNModel(n_node=5)
    A_tilde = torch.randint(2, (5, 5)).float()
    logits = model(A_tilde)
    print(logits.shape)
