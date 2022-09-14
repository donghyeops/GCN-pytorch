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
    dataset = CiteSeer(n_test_data=args.n_test_data)

    model = GCNModel(
        n_node=len(dataset),
        n_feature=args.n_unit,
        n_layer=args.n_layer,
        n_class=dataset.num_classes
    ).cuda()

    n_trainable_params = count_parameters(model)
    print(f'trainable params#: {n_trainable_params:,}')

    loss_func = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=0.1)

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

    A_tilde, nodes, labels = dataset(split='train')
    A_tilde = A_tilde.float().cuda()
    nodes = nodes
    labels = labels.cuda()
    output = model(A_tilde)[nodes]
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

    A_tilde, nodes, labels = dataset(split='test')
    A_tilde = A_tilde.float().cuda()
    nodes = nodes
    labels = labels.cuda()

    with torch.no_grad():
        output = model(A_tilde)[nodes]
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
    parser.add_argument('--n_unit', type=int, default=16)
    parser.add_argument('--n_layer', type=int, default=2)
    parser.add_argument('--weighing_factor', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--n_epoch', type=int, default=200)
    parser.add_argument('--lr_decay_step', type=int, default=200)
    parser.add_argument('--n_test_data', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=7770)
    args = parser.parse_args()
    print(args)

    set_seed(args.seed)

    run(args)
