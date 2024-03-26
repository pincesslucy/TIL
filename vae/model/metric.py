import torch
import numpy as np
from scipy.linalg import sqrtm


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def fid_score(real_feature, fake_feature):
    real_feature = real_feature[0].cpu().detach().numpy()
    fake_feature = fake_feature[0].cpu().detach().numpy()
    real_feature = real_feature.reshape(real_feature.shape[0], -1)
    fake_feature = fake_feature.reshape(fake_feature.shape[0], -1)
    mu1, sigma1 = real_feature.mean(axis=0), np.cov(real_feature, rowvar=False)
    mu2, sigma2 = fake_feature.mean(axis=0), np.cov(fake_feature, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid