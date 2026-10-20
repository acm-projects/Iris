"""Load our own pose sign classifiers (ml/two_word) by name for live inference.

A model is picked by name, so swapping models needs no code change:

  meeting100                    -> ml/two_word/runs/meeting100/best.pt
  pretrain_asl2000_e100         -> ml/two_word/pretrained/pretrain_asl2000_e100.pt
  asl100_15_ft_3seeds           -> ml/two_word/runs/asl100_15_ft_3seeds/seed0/best.pt
  path/to/any/best.pt           -> that file

Each checkpoint carries its own word list (label order), feature settings and
frame count, so whatever model is chosen is rebuilt exactly as it was trained.
The input features are computed by the same `build_features` training uses,
including the shared normalization in landmark_normalization.py.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
TWO_WORD_DIR = REPO_ROOT / "ml" / "two_word"
PRETRAINED_DIR = TWO_WORD_DIR / "pretrained"
RUNS_DIR = TWO_WORD_DIR / "runs"


def _load_module(name: str, path: Path):
    """Import a training module by file path under a unique name (avoids clashing with other 'data' modules)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_data = _load_module("two_word_data", TWO_WORD_DIR / "data.py")
_model = _load_module("two_word_model", TWO_WORD_DIR / "model.py")


def resolve_model(name: str) -> Path:
    """Model name (or path) -> checkpoint file. Raises FileNotFoundError listing what's available."""
    candidates = [
        Path(name),
        PRETRAINED_DIR / f"{name}.pt",
        RUNS_DIR / name / "best.pt",
    ]
    candidates += sorted((RUNS_DIR / name).glob("seed*/best.pt"))  # multi-seed run: first seed
    for path in candidates:
        if path.is_file():
            return path
    available = ", ".join(list_models()) or "none"
    raise FileNotFoundError(f"No model named {name!r}. Available: {available}")


def list_models() -> list[str]:
    """Names usable with resolve_model: local training runs + committed pretrained models."""
    names = []
    if RUNS_DIR.is_dir():
        for run in sorted(RUNS_DIR.iterdir()):
            if (run / "best.pt").is_file() or any(run.glob("seed*/best.pt")):
                names.append(run.name)
    names += [p.stem for p in sorted(PRETRAINED_DIR.glob("*.pt"))]
    return names


class PoseModel:
    """One trained classifier, ready to score recorded MediaPipe keypoints."""

    def __init__(self, name: str, device: str | None = None):
        self.path = resolve_model(name)
        self.name = name
        ckpt = torch.load(self.path, map_location="cpu", weights_only=True)
        cfg = ckpt["config"]
        self.labels: list[str] = list(ckpt["glosses"])  # label number -> word, saved at training time
        self.components = _data.parse_features(cfg.get("features", "legacy"))
        if self.components == ["legacy"]:
            raise ValueError(f"{name} uses 'legacy' features, which live inference doesn't support; "
                             "retrain it with the default --features.")
        self.num_frames = int(cfg["num_frames"])
        self.feature_dim = _data.feature_dim(self.components)
        self.trained_val_acc = float(ckpt.get("val_acc", float("nan")))
        self.trained_epoch = int(ckpt.get("epoch", -1))

        self.device = torch.device(device or "cpu")
        self.model = _model.PoseGRU(self.feature_dim, len(self.labels), cfg["hidden_size"],
                                    num_layers=cfg.get("num_layers", 1), dropout=cfg["dropout"])
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device).eval()

    def features(self, frames: np.ndarray) -> np.ndarray:
        """(T, 55, 2) pixel keypoints with NaN for undetected joints -> (num_frames, feature_dim).

        Mirrors evaluation in training: pick num_frames evenly spaced frames
        across the whole recording, then build the same features.
        """
        frames = np.asarray(frames, dtype=np.float32)
        if frames.ndim != 3 or frames.shape[1:] != (55, 2) or len(frames) == 0:
            raise ValueError(f"Expected frames shaped (T, 55, 2); got {frames.shape}.")
        detected = np.isfinite(frames).all(axis=-1)
        # Training input format: x, y, confidence (0 = not detected).
        raw = np.zeros((len(frames), 55, 3), dtype=np.float32)
        raw[..., :2] = np.where(detected[..., None], frames, 0.0)
        raw[..., 2] = detected.astype(np.float32)
        idx = _data.sample_frames(len(raw), self.num_frames, False, np.random.default_rng(0))
        return _data.build_features(raw[idx], self.components)

    def predict(self, frames: np.ndarray, top_k: int = 5) -> list[dict]:
        """Score a recording -> [{"gloss", "confidence"}, ...], most likely first."""
        x = torch.from_numpy(self.features(frames))[None].to(self.device)
        with torch.inference_mode():
            probs = self.model(x).softmax(1)[0]
        if not torch.isfinite(probs).all():
            raise ValueError("Model produced non-finite scores.")
        conf, idx = probs.topk(min(top_k, len(self.labels)))
        return [{"gloss": self.labels[i], "confidence": float(c)}
                for i, c in zip(idx.cpu().tolist(), conf.cpu().tolist())]
