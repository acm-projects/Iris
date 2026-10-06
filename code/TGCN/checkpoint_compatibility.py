"""Repository audit decisions, not inferred from tensor shapes or loader agreement.

Only a reviewed training provenance record can justify a compatible decision.
No user-editable sidecar or checkpoint field is treated as proof automatically.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PreprocessingAudit:
    status: str
    reason: str


ASL100_AUDIT = PreprocessingAudit(
    "unknown",
    "HF ASL100 download hash is verified, but its publisher revision lacks training preprocessing and label-map "
    "provenance. Historical affine code differs from selected pivot_v1; "
    "see PREPROCESSING_STATUS.md. Supply original training provenance or retrain and audit.",
)


def require_compatible(audit):
    """Fail closed for unknown, incompatible, missing, or malformed decisions."""
    if not isinstance(audit, PreprocessingAudit) or audit.status != "compatible":
        status = audit.status if isinstance(audit, PreprocessingAudit) else "unknown"
        reason = audit.reason if isinstance(audit, PreprocessingAudit) else "No reviewed preprocessing audit."
        raise RuntimeError(f"Preprocessing {status}: predictions blocked. {reason}")
