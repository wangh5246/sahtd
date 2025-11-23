# -*- coding: utf-8 -*-
# utils_/newsahtd_boost.py
# 仅提供 NewSAHTD 的“自适应门控 / 稳健回退 / 不确定度驱动采样 / 极小探针路由”工具函数。
# 不依赖第三方库（除 numpy），可零侵入接入你现有实现。

from __future__ import annotations
import numpy as np

# —— 稳健尺度与权重 —— #

def mad_scale(x: np.ndarray) -> float:
    """1.4826 * MAD；x 可含 NaN"""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return 0.0
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))

def huber_weights(resid: np.ndarray, c: float) -> np.ndarray:
    """Huber 权重：|r|<=c => 1；否则 => c/|r|"""
    r = np.asarray(resid, dtype=float)
    w = np.ones_like(r)
    m = np.abs(r) > c
    w[m] = np.where(np.abs(r[m]) > 0, c / np.abs(r[m]), 0.0)
    return w

def irls_huber_mean(y: np.ndarray, x0: float, c: float, max_iter: int = 20, tol: float = 1e-6) -> float:
    """一维 IRLS-Huber 稳健均值；无外部依赖"""
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]
    if y.size == 0:
        return float(x0)
    mu = float(x0)
    for _ in range(max_iter):
        r = y - mu
        w = huber_weights(r, c)
        den = float(np.sum(w)) + 1e-12
        mu_new = float(np.sum(w * y) / den)
        if abs(mu_new - mu) < tol:
            break
        mu = mu_new
    return mu

def trimmed_mean(y: np.ndarray, trim: float = 0.1) -> float:
    """双侧截尾均值"""
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]
    if y.size == 0:
        return 0.0
    v = np.sort(y)
    k = int(len(v) * trim)
    v = v[k: len(v) - k] if len(v) - 2 * k > 0 else v
    return float(np.mean(v))

# —— 不确定度与路由建议 —— #

def combine_uncertainty(post_var: float, resid_ratio: float, alpha: float = 0.7) -> float:
    """组合不确定度：alpha*方差 + (1-alpha)*高残差比例"""
    return float(alpha * post_var + (1 - alpha) * resid_ratio)

def suggest_kp(kp: int, u: float, u_lo: float, u_hi: float, kp_min: int, kp_max: int) -> int:
    """根据不确定度自调 Kp；返回新 Kp"""
    if u > u_hi:
        return min(kp_max, int(round(kp * 1.4)))
    if u < u_lo:
        return max(kp_min, int(round(kp * 0.85)))
    return kp

def need_probe_route(u: float, u_hi: float, route_ratio_so_far: float, route_ratio_min: float = 0.05) -> bool:
    """是否触发极小比例的 B 路探针（仅在不确定度高且本轮探针尚未满足时）"""
    return (u > u_hi) and (route_ratio_so_far < route_ratio_min)

def cap_weights(w: np.ndarray, cap_ratio: float = 0.10) -> np.ndarray:
    """工人权重上限裁剪，避免单点主导"""
    w = np.asarray(w, dtype=float)
    s = float(np.sum(w)) + 1e-12
    cap = cap_ratio * s
    return np.minimum(w, cap)

# —— 自适应 Huber 阈值（带 EMA 平滑） —— #

class CAdaptive:
    """维护 Huber c 的指数滑动：c_t = beta*c_{t-1} + (1-beta)*k*MAD"""
    def __init__(self, k: float = 1.5, beta: float = 0.9):
        self.k = float(k)
        self.beta = float(beta)
        self._c_prev = None

    def update(self, residual_like: np.ndarray) -> float:
        s = mad_scale(residual_like)
        c_now = max(1e-6, self.k * s)
        if self._c_prev is None:
            self._c_prev = c_now
        else:
            self._c_prev = self.beta * self._c_prev + (1 - self.beta) * c_now
        return float(self._c_prev)

# —— 稳健回退判据 —— #

def need_fallback(resid_ratio: float, sample_size: int, phi: float = 0.25, min_size: int = 12) -> bool:
    """高残差比例超过阈值或样本过少时回退"""
    if sample_size < min_size and resid_ratio > phi:
        return True
    return resid_ratio > phi
