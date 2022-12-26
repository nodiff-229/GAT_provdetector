import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.nn as dglnn
from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
from GNN_model.data_preprocess import *
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
import warnings

warnings.filterwarnings("ignore")

class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(
            dglnn.GraphConv(in_size, hid_size, activation=F.relu)
        )
        self.layers.append(dglnn.GraphConv(hid_size, out_size))
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h


def evaluate(g, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        # _, indices = torch.max(logits, dim=1)
        # correct = torch.sum(indices == labels)
        logits = F.sigmoid(logits)
        predicts = torch.where(logits > 0.5, 1, 0)
        correct = torch.sum(predicts == labels)

        ap = average_precision_score(labels, logits, pos_label=0)
        auc = roc_auc_score(labels, logits)
        f1 = f1_score(labels, predicts, pos_label=0)

        return correct.item() * 1.0 / len(labels), ap, auc, f1


def train(g, features, labels, masks, model):
    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    val_mask = masks[1]
    loss_fcn = nn.BCEWithLogitsLoss()
    # loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    # training loop
    for epoch in range(200):
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc, ap, auc, f1 = evaluate(g, features, labels, val_mask, model)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | Ap {:.4f}| Auc {:.4f}| F1-score {:.4f}".format(
                epoch, loss.item(), acc, ap, auc, f1
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="fake",
        help="Dataset name ('cora', 'citeseer', 'pubmed','fake','real').",
    )
    args = parser.parse_args()
    print(f"Training with DGL built-in GraphConv module.")

    # load and preprocess dataset
    transform = (
        AddSelfLoop()
    )  # by default, it will first remove self-loops to prevent duplication
    if args.dataset == "cora":
        data = CoraGraphDataset(transform=transform)
    elif args.dataset == "citeseer":
        data = CiteseerGraphDataset(transform=transform)
    elif args.dataset == "pubmed":
        data = PubmedGraphDataset(transform=transform)
    elif args.dataset == "fake":
        data = fake_dataset()
        data = dgl.remove_self_loop(data)
        data = dgl.add_self_loop(data)
    elif args.dataset == "real":
        pass
    else:
        raise ValueError("Unknown dataset: {}".format(args.dataset))
    g = data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = g.int().to(device)
    features = torch.concat((g.ndata["name"], g.ndata["type"]), dim=-1)
    labels = g.ndata["label"]
    masks = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]

    # create GCN model
    in_size = features.shape[1]
    # out_size = data.num_classes
    out_size = 1
    model = GCN(in_size, 16, out_size).to(device)

    # model training
    print("Training...")
    train(g, features, labels, masks, model)

    # test the model
    print("Testing...")
    acc, ap, auc, f1 = evaluate(g, features, labels, masks[2], model)
    print("Test accuracy {:.4f} | Ap {:.4f}| Auc {:.4f}| F1-score {:.4f}".format(acc, ap, auc, f1))
