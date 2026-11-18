"""Verbatim copy of `casper_k` from the CASPER-k project (casper_k/casperk_learning_study.py),
used only to check the JAX port."""
import torch


def casper_k(y, S, k=3, eps0=1e-3, gamma=0.0):
    B, h = y.shape
    nS = S.sum(1)
    big = nS >= k
    ym = y.masked_fill(~S, -1.0)
    idx = ym.topk(k, dim=1).indices
    Tm = torch.zeros_like(S).scatter_(1, idx, True)
    mT0 = y.masked_fill(~Tm, 2.0).min(1, keepdim=True).values
    off = torch.where(Tm, gamma * (y - mT0), torch.zeros_like(y))
    yp = y - off
    mT = yp.masked_fill(~Tm, 2.0).min(1, keepdim=True).values
    MT = yp.masked_fill(Tm, -1.0).max(1, keepdim=True).values
    eps = torch.clamp(mT.detach(), max=eps0)
    Lt = Tm & (yp <= MT + eps)
    Ht = (~Tm) & (yp >= mT - eps)
    Bt = Lt | Ht
    nL, nH = Lt.sum(1, keepdim=True), Ht.sum(1, keepdim=True)
    denom = (nL + nH).clamp(min=1)
    ut = ((yp * Bt).sum(1, keepdim=True) + nH * eps) / denom
    z_big = torch.where(Lt, ut, torch.where(Ht, ut - eps, yp))
    z_big = torch.where(nL > 0, z_big, yp) + off
    out_mass = (y * (~S)).sum(1, keepdim=True)
    z_small = torch.where(S, y + out_mass / nS.clamp(min=1).unsqueeze(1), torch.zeros_like(y))
    z = torch.where(big.unsqueeze(1), z_big, z_small)
    T = torch.where(big.unsqueeze(1), Tm, S)
    return z, T
