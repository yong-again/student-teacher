import torch
import torch.nn as nn
from einops import reduce


def knowledge_distillation(outputs, targets):
    # dim : [batch, vector]
    err = torch.norm(outputs - targets, dim=1)**2
    loss = torch.mean(err)
    return loss

def compactness_loss(outputs):
    # dim: [batch, vector]
    _, n = outputs.size()
    avg = torch.mean(outputs, axis=1)
    std = torch.std(outputs, axis=1)
    zt =  outputs.T - avg
    zt /= std
    corr = torch.matmul(zt.T, zt) / (n - 1)
    loss = torch.sum(torch.triu(corr, diagonal=1) ** 2)
    return loss

def student_loss(outputs, targets):
    # dim: [batch, h, w vector]
    err = reduce((outputs - targets)**2, 'b h w vec -> b h w', 'sum')
    loss = torch.mean(err)
    return loss


