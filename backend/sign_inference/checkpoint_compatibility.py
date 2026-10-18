"""Reviewed checkpoint contracts used by Iris sign inference.

The contract records the exact, revision-pinned checkpoint and the matching
public WLASL preprocessing/label behavior.  Shape agreement alone is not
treated as proof: changes to any of these inputs require a new audit.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PreprocessingAudit:
    status: str
    reason: str
    preprocessing: str | None = None
    label_map: str | None = None


ASL100_AUDIT = PreprocessingAudit(
    status="compatible",
    reason=(
        "Reviewed 2026-10-18: the revision-pinned ASL100 checkpoint and strict "
        "state dict match the upstream WLASL TGCN architecture. Iris reproduces "
        "upstream code/TGCN/sign_dataset.py xy preprocessing (2*(pixel/256-.5)), "
        "the 55-joint BODY_25-plus-hands order, and its alphabetically sorted "
        "LabelEncoder mapping. The deterministic static-keypoint regression "
        "fixture locks this contract; see PREPROCESSING_STATUS.md."
    ),
    preprocessing="wlasl_affine_256_v1",
    label_map="wlasl_asl100_sorted_gloss_v1",
)


def require_compatible(audit):
    """Fail closed for unknown, incompatible, missing, or malformed decisions."""
    if not isinstance(audit, PreprocessingAudit) or audit.status != "compatible":
        status = audit.status if isinstance(audit, PreprocessingAudit) else "unknown"
        reason = audit.reason if isinstance(audit, PreprocessingAudit) else "No reviewed preprocessing audit."
        raise RuntimeError(f"Preprocessing {status}: predictions blocked. {reason}")
