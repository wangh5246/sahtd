# -*- coding: utf-8 -*-
"""
dap_client.py
=============
SAHTD-X C 路用的 DAP/VDAF 客户端实现。

支持两种模式：
- mode="dryrun": 完全本地聚合，用于开发/实验，不依赖任何外部服务；
- mode="daphne"/"divviup": 通过 HTTP 调用真实 DAP/VDAF 服务（需要你按实际接口补充 URL）。

sahtd_x-2.py 中期望的接口：
    submit_reports(task_id, reports, batch_key)
    start_collect(task_id, batch_key) -> {"collect_id": ...}
    poll_collect(task_id, collect_id, timeout_s, interval_s)
    get_aggregate(task_id, collect_id, batch_key) -> {"sum": [sum_v, sum_n]}
"""

from __future__ import annotations
from typing import List, Dict, Any, Tuple
import time
import uuid

try:
    import requests
except Exception:
    requests = None


class DAPClient:
    def __init__(
        self,
        leader_url: str,
        helper_url: str = "",
        api_token: str | None = None,
        mode: str = "dryrun",
        timeout: int = 30,
    ):
        self.leader_url = leader_url.rstrip("/") if leader_url else ""
        self.helper_url = helper_url.rstrip("/") if helper_url else ""
        self.api_token = api_token
        self.mode = mode.lower()
        self.timeout = int(timeout)

        # dryrun 模式下的本地存储：(task_id, batch_key) -> list[vectors]
        self._store: Dict[Tuple[str, str], List[list[float]]] = {}

    # ------------------------------------------------------------------
    # 上报测量：sahtd_x-2.py already packs "reports" 为:
    #   {"vector": [value, 1.0], "entity_id": str(e), "slot": batch_key}
    # ------------------------------------------------------------------
    def submit_reports(self, task_id: str, reports: List[Dict[str, Any]], batch_key: str | int):
        key = (str(task_id), str(batch_key))

        if self.mode == "dryrun":
            buf = self._store.setdefault(key, [])
            for r in reports:
                vec = r.get("vector", None)
                if vec is None:
                    continue
                buf.append([float(vec[0]), float(vec[1])])
            return {"status": "ok", "mode": "dryrun", "count": len(reports)}

        # 真正 HTTP 模式
        if requests is None:
            raise RuntimeError("requests 未安装，无法使用 HTTP DAP 模式")

        payload = {
            "task_id": task_id,
            "batch_key": str(batch_key),
            "reports": reports,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        url = self.leader_url + "/upload"
        resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # 启动一次聚合收集（返回 collect_id）
    # ------------------------------------------------------------------
    def start_collect(self, task_id: str, batch_key: str | int) -> Dict[str, Any]:
        collect_id = str(uuid.uuid4())

        if self.mode == "dryrun":
            # dryrun 模式下，仅生成一个 ID 即可
            return {"collect_id": collect_id}

        if requests is None:
            raise RuntimeError("requests 未安装，无法使用 HTTP DAP 模式")

        payload = {
            "task_id": task_id,
            "batch_key": str(batch_key),
        }
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        url = self.leader_url + "/collect"
        resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        # 若服务端返回自己的 collect_id，就用它覆盖
        return {"collect_id": data.get("collect_id", collect_id)}

    # ------------------------------------------------------------------
    # 轮询等待聚合完成；dryrun 中只是 sleep 一下
    # ------------------------------------------------------------------
    def poll_collect(
        self,
        task_id: str,
        collect_id: str,
        timeout_s: float = 15.0,
        interval_s: float = 0.5,
    ):
        if self.mode == "dryrun":
            # dryrun：简单等待一小会，假装服务端在处理
            t0 = time.time()
            while time.time() - t0 < min(timeout_s, 1.0):
                time.sleep(min(interval_s, 0.1))
            return {"status": "ok", "mode": "dryrun"}

        if requests is None:
            raise RuntimeError("requests 未安装，无法使用 HTTP DAP 模式")

        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        url = self.leader_url + f"/collect/{collect_id}/status"
        t0 = time.time()
        while True:
            resp = requests.get(url, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            if data.get("done", False):
                return data
            if time.time() - t0 > timeout_s:
                raise TimeoutError("DAP poll_collect 超时")
            time.sleep(interval_s)

    # ------------------------------------------------------------------
    # 取回聚合结果：返回 {"sum": [sum_v, sum_n], ...}
    # ------------------------------------------------------------------
    def get_aggregate(self, task_id: str, collect_id: str, batch_key: str | int) -> Dict[str, Any]:
        key = (str(task_id), str(batch_key))

        if self.mode == "dryrun":
            vecs = self._store.get(key, [])
            if not vecs:
                return {"sum": [0.0, 0.0], "mode": "dryrun"}
            sum_v = float(sum(v[0] for v in vecs))
            sum_n = float(sum(v[1] for v in vecs))
            return {"sum": [sum_v, sum_n], "mode": "dryrun"}

        if requests is None:
            raise RuntimeError("requests 未安装，无法使用 HTTP DAP 模式")

        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        url = self.leader_url + f"/collect/{collect_id}/aggregate"
        resp = requests.get(url, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()