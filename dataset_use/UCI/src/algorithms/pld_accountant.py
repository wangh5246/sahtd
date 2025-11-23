
# -*- coding: utf-8 -*-
"""
pld_accountant.py
=================
A lightweight privacy accountant with PLD-flavored API.
For practicality, we combine two tracks:
  - Pure-DP composition (e.g., Laplace/LDP): ε adds up; δ=0.
  - Gaussian/RDP track: accumulate Rényi DP and convert to (ε, δ) on demand.

API:
  acc = PLDAccountant(delta_target=1e-5)
  acc.add_pure_dp(eps)                 # pure (ε,0)-DP event
  acc.add_gaussian(sigma, sens=1.0)    # Gaussian mechanism (RDP)
  eps = acc.epsilon()                  # minimal ε for configured δ
  delta = acc.delta(epsilon_query)     # δ at given ε
  acc.reset()

This is a simple, self-contained accountant; for tight bounds you can replace it
with a Fourier/PLD accountant later without changing the call-sites.
"""
from __future__ import annotations
import math
from typing import Dict, List

class PLDAccountant:
    def __init__(self, delta_target: float = 1e-5, orders: List[float] = None):
        self.delta_target = float(delta_target)
        self.orders = orders or [1.25, 1.5, 2, 3, 5, 8, 10, 16, 32, 64, 128]
        self.reset()

    def reset(self):
        self.eps_pure = 0.0
        self.rdp: Dict[float, float] = {a: 0.0 for a in self.orders}

    # ---- adding events ----
    def add_pure_dp(self, eps: float):
        self.eps_pure += float(max(0.0, eps))

    def add_shuffle_ldp(self, eps_local: float, n: int):
        # effective epsilon after shuffling (very rough; replace with tight PLD later)
        if n <= 1:
            eff = eps_local
        else:
            eff = min(eps_local, abs(math.expm1(eps_local)) / math.sqrt(max(n,1)))
        self.add_pure_dp(eff)

    def add_gaussian(self, sigma: float, sens: float = 1.0):
        # RDP for Gaussian: epsilon(α) = α * sens^2 / (2 sigma^2)
        sigma2 = float(sigma)**2
        s2 = float(sens)**2
        for a in self.orders:
            self.rdp[a] += (a * s2) / (2.0 * sigma2)

    # ---- queries ----
    def epsilon(self, delta: float = None) -> float:
        """Return minimal ε for the configured (or given) δ."""
        delta = self.delta_target if delta is None else float(delta)
        # choose best order
        eps_candidates = []
        for a, v in self.rdp.items():
            if a <= 1.0: 
                continue
            eps_candidates.append(v + math.log(1.0 / max(delta, 1e-300)) / (a - 1.0))
        eps_rdp = min(eps_candidates) if eps_candidates else 0.0
        return float(self.eps_pure + eps_rdp)

    def delta(self, epsilon: float) -> float:
        """Return δ for a given total ε."""
        # If only pure-DP, δ=0 when epsilon >= eps_pure, else undefined (<0)
        if all(v == 0.0 for v in self.rdp.values()):
            return 0.0 if epsilon >= self.eps_pure else 1.0
        # otherwise minimize over orders
        rem = max(0.0, float(epsilon) - self.eps_pure)
        deltas = []
        for a, v in self.rdp.items():
            if a <= 1.0:
                continue
            deltas.append(math.exp((rem - v) * (a - 1.0)))
        return float(min(deltas) if deltas else 1.0)
