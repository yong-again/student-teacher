import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

@torch.no_grad()
def feature_stat(model, loader, device):
    model.eval()
    features = []
    for x, *_ in loader:
        z = model(x.to(device), return_features=True)
        features.append(z.cpu())

    Z = torch.cat(features, 0) # [N, C]
    mu = Z.mean(0)
    var = Z.var(0, unbiased=False)
    l2 = Z.norm(dim=1).mean().item()

    # feature correlation
    X = (Z - mu) / (var.sqrt() + 1e-6)
    corr = (X.T @ X) / (len(Z) - 1)
    off = corr - torch.diag(torch.diag(corr))
    off_diag_mse = (off ** 2).mean().item()

    return {
        "feature_mean_mean" : mu.mean().item(),
        "feature_var_mean" : var.mean().item(),
        "feature_norm_mean" : l2,
        "feature_offdiag_mse": off_diag_mse
    }

def linear_probe(train_feats, train_y, val_feats, val_y):
    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(train_feats, train_y)
    pred = clf.predict(val_feats)
    return accuracy_score(val_y, pred)

def retrieval_at_k(feats, labels, k=5):
    nn = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(feats)
    D, I = nn.kneighbors(feats) # I : [N, K+1] 자기 자신 포함
    I = I[:, 1:]
    hits = (labels[I] == labels[:, None]).any(axis=1) # R@k
    return hits.mean()

def tsne_plot(feats, labels=None):
    Z2 = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pce').fit_transform(feats)
    plt.figure()
    if labels is None:
        plt.scatter(Z2[:, 0], Z2[:, 1], c=labels, s=3, cmap='tab10')
    plt.title("t-SNE of ResNet features:")
    plt.savefig('./tsne.png', dpi=300)

def mahalanobis_scores(Z):
    # Z = [N, C]
    mu = Z.mean(0, keepdims=True)
    S = np.cov(Z.T) + 1e-6*np.eye(Z.shape[1])
    iS = np.linalg.inv(S)
    d2 = ((Z-mu) @ iS * (Z-mu)).sum(1)
    return d2

