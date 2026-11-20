# -*- coding: utf-8 -*-
"""
common.py —— 合并项目公共库

为两套管线（同事 OULAD 主链 + up_ 统一管线）提供共享设施：
  - 路径常量（02_数据与数据集 / 04_运行结果 / raw/）
  - 日志（控制台 + JSONL 审计日志）
  - GPU 加速引擎（--device auto|cpu|gpu：RAPIDS cuML → PyTorch CUDA → CPU 回退）
    KMeans / 轮廓系数 / 最近质心分配的统一 ClusterEngine
  - 中文字体配置、调色板、Markdown 表格降级导出

设备选择优先级：环境变量 WB_DEVICE > 默认 auto。
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ----------------------------------------------------------------------
# 路径常量（相对本文件 03_源代码/ 定位交付包结构）
# ----------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(HERE, "..", "02_数据与数据集"))
RAW_DIR = os.path.join(DATA_DIR, "raw")
OUT_DIR = os.path.abspath(os.path.join(HERE, "..", "04_运行结果"))
PKG_DIR = os.path.abspath(os.path.join(HERE, ".."))
FIG_DIR = os.path.join(PKG_DIR, "05_图表与看板", "figures")

SEED = 42
PALETTE = ["#4C72B0", "#55A868", "#C44E52", "#CCB974", "#8172B2", "#64B5CD"]
# 报告插图调色板（figures.py / multimodal_figures.py 共用，深蓝系适合 A4 印刷）
FIG_PALETTE = ["#1F4E79", "#2E75B6", "#C55A11", "#548235", "#7F7F7F", "#8FAADC"]

log_handle = None


def log(msg: str) -> None:
    """同时输出到控制台与 JSONL 审计日志。"""
    global log_handle
    print(msg)
    if log_handle:
        log_handle.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"),
                                     "msg": msg}, ensure_ascii=False) + "\n")
        log_handle.flush()


def open_log(out_dir: str) -> None:
    """打开审计日志文件（各阶段脚本共用追加模式）。"""
    global log_handle
    os.makedirs(out_dir, exist_ok=True)
    log_handle = open(os.path.join(out_dir, "up_pipeline_log.jsonl"), "a",
                      encoding="utf-8")


def close_log() -> None:
    global log_handle
    if log_handle:
        log_handle.close()
        log_handle = None


def setup_cjk_font() -> str | None:
    """探测并配置中文字体，避免图表中文显示为方块。"""
    candidates = ["Microsoft YaHei", "SimHei", "PingFang SC", "Hiragino Sans GB",
                  "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Micro Hei",
                  "SimSun", "KaiTi"]
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            log(f"  中文字体已配置: {name}")
            return name
    log("  警告: 未找到可用中文字体（可安装 微软雅黑/SimHei/Noto Sans CJK SC），"
        "图表中文可能显示为方块")
    return None


def df_markdown(df) -> list[str]:
    """DataFrame 转 Markdown 表格；缺少 tabulate 时降级为纯文本表格。"""
    try:
        return df.to_markdown(index=False).splitlines()
    except ImportError:
        return df.to_string(index=False).splitlines()


def apply_fig_style(dpi: int = 200) -> None:
    """报告插图统一 matplotlib 样式（figures.py / multimodal_figures.py /
    profile_modeling.py 共用，参数与各文件原配置逐项一致）。"""
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 9
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["figure.dpi"] = dpi
    plt.rcParams["savefig.dpi"] = dpi
    plt.rcParams["savefig.bbox"] = "tight"


def clean_axes(ax, grid_axis: str = "y") -> None:
    """报告插图坐标轴清理：去顶/右边框 + 浅灰网格（原 figures._clean）。"""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color="#DDDDDD", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)


# ======================================================================
# 计算加速：可选 GPU 后端（--device auto|cpu|gpu）
# ======================================================================
# 探测顺序：RAPIDS cuML（Linux/WSL，KMeans 全 GPU）→ PyTorch CUDA（Windows 可用，
# GPU 加速 KMeans / 轮廓系数 / 最近质心分配）→ CPU（sklearn + RF 多核并行）。
# 显式指定 --device gpu 但无可用后端时自动回退 CPU 并告警，不中断管线。
# 环境变量 WB_DEVICE 可在子进程编排（run_all.py）中贯通设备选择。
_DEVICE_PREF = os.environ.get("WB_DEVICE", "auto")
_ENGINE_INFO: dict = {}


def detect_device(pref: str = "auto") -> dict:
    """探测可用 GPU 计算后端；pref=cpu 强制 CPU。"""
    global _ENGINE_INFO
    info = {"pref": pref, "backend": "cpu", "device_name": None}
    if pref != "cpu":
        try:
            import cuml  # noqa: F401
            info["backend"] = "cuml"
        except Exception:
            pass
        if info["backend"] == "cpu":
            try:
                import torch
                if torch.cuda.is_available():
                    info["backend"] = "torch"
                    info["device_name"] = torch.cuda.get_device_name(0)
            except Exception:
                pass
    _ENGINE_INFO = info
    return info


def ensure_device() -> dict:
    """按环境变量 WB_DEVICE 探测一次设备并缓存（供所有阶段脚本调用）。"""
    if not _ENGINE_INFO:
        detect_device(_DEVICE_PREF)
    return _ENGINE_INFO


def device_banner() -> None:
    """打印当前计算设备信息。"""
    info = ensure_device()
    if info["backend"] == "cpu":
        log("  计算设备: CPU（sklearn 多核并行，--device gpu 可启用 GPU 加速）")
        if info.get("pref") == "gpu":
            log("  警告: 未检测到可用 GPU 后端（需 PyTorch+CUDA 或 RAPIDS cuML），已回退 CPU")
    else:
        log(f"  计算设备: GPU[{info.get('device_name') or 'CUDA'}]，"
            f"KMeans / 轮廓系数 / 质心分配使用 {info['backend']} 后端加速")


def _torch_device():
    import torch
    return torch.device("cuda")


def torch_kmeans(X_np: np.ndarray, k: int, seed: int = SEED, n_init: int = 3,
                 max_iter: int = 100, tol: float = 1e-5):
    """GPU KMeans（PyTorch Lloyd 迭代，多次初始化取最优）。"""
    import torch
    dev = _torch_device()
    torch.manual_seed(seed)
    X = torch.as_tensor(np.ascontiguousarray(X_np, dtype=np.float32), device=dev)
    n = X.shape[0]
    best = None
    for _ in range(max(1, n_init)):
        centers = X[torch.randperm(n, device=dev)[:k]].clone()
        for _ in range(max_iter):
            labels = torch.cdist(X, centers).argmin(1)
            sums = torch.zeros_like(centers)
            sums.index_add_(0, labels, X)
            counts = torch.zeros(k, device=dev)
            counts.index_add_(0, labels, torch.ones(n, device=dev))
            counts = counts.clamp(min=1)
            new_centers = sums / counts.unsqueeze(1)
            shift = float((new_centers - centers).norm(dim=1).max())
            centers = new_centers
            if shift < tol:
                break
        inertia = float(torch.cdist(X, centers).min(dim=1).values.pow(2).sum())
        if best is None or inertia < best[0]:
            best = (inertia, centers, labels)
    inertia, centers, labels = best
    return centers.cpu().numpy(), labels.cpu().numpy(), inertia


def torch_silhouette(X_np: np.ndarray, labels: np.ndarray, chunk: int = 4096) -> float:
    """GPU 轮廓系数（分块计算成对距离，控制显存占用）。"""
    import torch
    dev = _torch_device()
    X = torch.as_tensor(np.ascontiguousarray(X_np, dtype=np.float32), device=dev)
    y = torch.as_tensor(np.asarray(labels, dtype=np.int64), device=dev)
    n = X.shape[0]
    uniq, y_inv = torch.unique(y, return_inverse=True)
    k = len(uniq)
    sizes = torch.bincount(y_inv, minlength=k).float()
    s = torch.zeros(n, device=dev)
    for st in range(0, n, chunk):
        ed = min(st + chunk, n)
        d = torch.cdist(X[st:ed], X)
        same = y_inv[st:ed].unsqueeze(1) == y_inv.unsqueeze(0)
        a = (d * same).sum(1) / (sizes[y_inv[st:ed]] - 1).clamp(min=1)
        sums = torch.zeros((ed - st, k), device=dev)
        sums.scatter_add_(1, y_inv.unsqueeze(0).expand(ed - st, -1), d)
        mean = sums / sizes.unsqueeze(0)
        mean.scatter_(1, y_inv[st:ed].unsqueeze(1), float("inf"))   # 屏蔽自身簇
        b = mean.min(dim=1).values
        s[st:ed] = (b - a) / torch.maximum(a, b).clamp(min=1e-12)
    s[sizes[y_inv] <= 1] = 0            # 单点簇轮廓系数记 0（与 sklearn 一致）
    return float(s.mean())


def torch_nearest_centers(X_np: np.ndarray, centers_np: np.ndarray,
                          chunk: int = 16384) -> np.ndarray:
    """GPU 最近质心分配（全量样本标签，分块控显存）。"""
    import torch
    dev = _torch_device()
    C = torch.as_tensor(np.ascontiguousarray(centers_np, dtype=np.float32), device=dev)
    X = torch.as_tensor(np.ascontiguousarray(X_np, dtype=np.float32), device=dev)
    labels = torch.empty(X.shape[0], dtype=torch.long, device=dev)
    for st in range(0, X.shape[0], chunk):
        ed = min(st + chunk, X.shape[0])
        labels[st:ed] = torch.cdist(X[st:ed], C).argmin(1)
    return labels.cpu().numpy()


class ClusterEngine:
    """KMeans + 轮廓系数统一引擎，按 detect_device 探测的后端自动加速。"""

    def __init__(self, info: dict | None = None):
        self.backend = (info or _ENGINE_INFO or {"backend": "cpu"})["backend"]

    def fit(self, X_fit: np.ndarray, k: int, seed: int = SEED, n_init: int = 10) -> dict:
        if self.backend == "torch":
            centers, labels, inertia = torch_kmeans(X_fit, k, seed=seed, n_init=3)
            sil = torch_silhouette(X_fit, labels)
            return {"labels": labels, "centers": centers, "inertia": inertia, "sil": sil}
        if self.backend == "cuml":
            from cuml.cluster import KMeans as cuKMeans
            km = cuKMeans(n_clusters=k, random_state=seed,
                          n_init=3).fit(X_fit.astype(np.float32))
            labels = np.asarray(km.labels_).astype(int)
            try:
                sil = float(silhouette_score(X_fit, labels))
            except ValueError:
                sil = np.nan
            return {"labels": labels, "centers": np.asarray(km.cluster_centers_),
                    "inertia": float(km.inertia_), "sil": sil}
        km = KMeans(n_clusters=k, random_state=seed, n_init=n_init).fit(X_fit)
        try:
            sil = float(silhouette_score(X_fit, km.labels_))
        except ValueError:
            sil = np.nan
        return {"labels": km.labels_, "centers": km.cluster_centers_,
                "inertia": km.inertia_, "sil": sil}

    def predict(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        if self.backend == "torch":
            return torch_nearest_centers(X, centers)
        d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        return d.argmin(1).astype(int)


def knee_distances(ks: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """肘部得分：inertia 下降曲线上每个 k 到首尾连线的垂直距离（kneedle 思路）。"""
    x0, y0, x1, y1 = ks[0], ys[0], ks[-1], ys[-1]
    denom = np.hypot(x1 - x0, y1 - y0) + 1e-12
    return np.abs((y1 - y0) * ks - (x1 - x0) * ys + x1 * y0 - y1 * x0) / denom
