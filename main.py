import argparse
import time
import random

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from dataset import CiteSeer
from model import GCNModel


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run(args):
    if args.dataset == 'citeseer':
        dataset = CiteSeer(
            n_train_data_per_class=args.n_train_data_per_class,
            n_test_data=args.n_test_data
        )
    else:
        raise ValueError(f"not found {args.dataset} dataset")
    print(f"Load Dataset ({args.dataset})")
    print(f" - node#: {dataset.num_nodes:,}")
    print(f" - edge#: {dataset.num_edges:,}")
    print(f" - class#: {dataset.num_classes:,}")
    print(f" - feat_size: {len(dataset.feats[0]):,}")
    print(f" - label_rate: {len(dataset.train_nodes) / len(dataset.nodes):,}")
    print('')

    model = GCNModel(
        in_dim=args.in_dim,
        hidden_dim=args.hidden_dim,
        n_layer=args.n_layer,
        n_class=dataset.num_classes,
        dropout=args.dropout,
        residual=args.residual,
        embedding=args.embedding,
        n_node=dataset.num_nodes,
    ).cuda()

    n_trainable_params = count_parameters(model)
    print(f'trainable params#: {n_trainable_params:,}')

    loss_func = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)

    print('Train Model')
    _st = time.time()
    for ep in range(args.n_epoch):
        train_loss, train_acc = train_model(
            model=model, dataset=dataset, optimizer=optimizer, loss_func=loss_func
        )
        test_loss, test_acc = test_model(model=model, dataset=dataset, loss_func=loss_func)
        duration = time.time() - _st
        lr = optimizer.param_groups[0]["lr"]
        print(
            f'[{ep}/{args.n_epoch}]'
            f' train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}'
            f' test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}'
            f' time: {duration//60:.0f}m {duration%60:.2f}s'
            f' lr: {lr}'
        )
        _st = time.time()
        scheduler.step()


def train_model(model, dataset, optimizer, loss_func):
    model.train()

    A_tilde, nodes, feats, labels = dataset(split='train')
    A_tilde = A_tilde.float().cuda()
    feats = feats.cuda()
    labels = labels.cuda()
    output = model(A_tilde, feats)[nodes]
    loss = loss_func(output, labels)

    model.zero_grad()
    loss.backward()
    optimizer.step()
    loss = loss.item()

    preds = output.argmax(axis=1)
    n_correct = torch.sum((preds.cpu() == labels.cpu())).item()
    n_sample = len(nodes)
    acc = n_correct / n_sample
    return loss, acc


def test_model(model, dataset, loss_func):
    model.eval()

    A_tilde, nodes, feats, labels = dataset(split='test')
    A_tilde = A_tilde.float().cuda()
    feats = feats.cuda()
    labels = labels.cuda()

    with torch.no_grad():
        output = model(A_tilde, feats)[nodes]
        loss = loss_func(output, labels)
        loss = loss.item()

    preds = output.argmax(axis=1)
    n_correct = torch.sum((preds.cpu() == labels.cpu())).item()
    n_sample = len(nodes)
    acc = n_correct / n_sample
    return loss, acc


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dim', type=int, default=3703)  # citeseer: 3703
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--n_layer', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--label_smoothing', type=float, default=0)
    parser.add_argument('--n_epoch', type=int, default=200)
    parser.add_argument('--lr_decay_step', type=int, default=100)
    parser.add_argument('--lr_decay_gamma', type=int, default=0.1)
    parser.add_argument('--residual', action="store_true", default=False)  # n_layer > 2 only
    parser.add_argument('--embedding', action="store_true", default=False)

    parser.add_argument('--dataset', type=str, default='citeseer')
    parser.add_argument('--n_train_data_per_class', type=int, default=20)
    parser.add_argument('--n_test_data', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=7770)
    args = parser.parse_args()
    print(args)

    set_seed(args.seed)

    run(args)
