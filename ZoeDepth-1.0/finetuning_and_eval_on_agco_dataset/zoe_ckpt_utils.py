from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, Iterable
from collections import OrderedDict
import torch


def _is_state_dict(obj: Any) -> bool:
    """Heuristic: a state_dict is a mapping whose values are mostly tensors."""
    if not isinstance(obj, (dict, OrderedDict)):
        return False
    if len(obj) == 0:
        return False
    tensor_like = 0
    total = 0
    for v in obj.values():
        total += 1
        if torch.is_tensor(v):
            tensor_like += 1
    # Many checkpoints are 100% tensors; be tolerant.
    return (tensor_like / max(1, total)) >= 0.8


def _count_tensors(d: Dict[str, Any]) -> int:
    return sum(1 for v in d.values() if torch.is_tensor(v))


def extract_state_dict(ckpt_obj: Any) -> Dict[str, torch.Tensor]:
    """
    Extract the real tensor state_dict from common checkpoint formats.

    Handles:
      - plain state_dict (OrderedDict[str, Tensor])
      - dict with keys like: 'state_dict', 'model', 'model_state_dict', etc.
      - nested dict where the biggest tensor-dict is the weights
    """
    # Case 1: already a state_dict
    if _is_state_dict(ckpt_obj):
        return dict(ckpt_obj)

    # Case 2: dict wrapper
    if isinstance(ckpt_obj, dict):
        # common direct keys
        for k in ["state_dict", "model", "model_state_dict", "net", "weights"]:
            if k in ckpt_obj and _is_state_dict(ckpt_obj[k]):
                return dict(ckpt_obj[k])

        # Otherwise: pick the nested dict with most tensors
        best = None
        best_score = -1
        for k, v in ckpt_obj.items():
            if _is_state_dict(v):
                score = _count_tensors(v)
                if score > best_score:
                    best = v
                    best_score = score

        if best is not None:
            return dict(best)

    raise RuntimeError(
        "Could not find a valid state_dict inside checkpoint. "
        "Run inspect to see keys/structure."
    )


def strip_prefix(state: Dict[str, torch.Tensor], prefixes: Iterable[str]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out


def load_state_dict_strict(
    model: torch.nn.Module,
    ckpt_path: str,
    device: torch.device,
    *,
    verbose: bool = True,
) -> Tuple[int, int]:
    """
    Loads a checkpoint into model with strict=True after extracting + cleaning keys.
    Returns: (num_missing, num_unexpected)  -> should be (0,0) in the ideal case.
    """
    ckpt_obj = torch.load(ckpt_path, map_location=device)
    sd = extract_state_dict(ckpt_obj)

    # Strip common wrappers
    sd = strip_prefix(sd, prefixes=[
        "module.",
        "model.",
        "state_dict.",
        "net.",
        "zoe.",
    ])

    # Try strict load
    try:
        model.load_state_dict(sd, strict=True)
        if verbose:
            print("[CKPT] strict=True load OK ")
        return (0, 0)
    except RuntimeError as e:
        # Give a useful report and stop (don’t silently train from scratch)
        msd = model.state_dict()
        model_keys = set(msd.keys())
        ckpt_keys = set(sd.keys())

        missing = sorted(list(model_keys - ckpt_keys))
        unexpected = sorted(list(ckpt_keys - model_keys))

        # shape mismatches
        shape_bad = []
        intersect = model_keys & ckpt_keys
        for k in list(intersect)[:50000]:
            if msd[k].shape != sd[k].shape:
                shape_bad.append((k, tuple(sd[k].shape), tuple(msd[k].shape)))

        print("\n[CKPT] strict load FAILED ")
        print("Reason:", str(e).split("\n")[0])
        print(f"Missing keys    : {len(missing)}")
        print(f"Unexpected keys : {len(unexpected)}")
        print(f"Shape mismatches: {len(shape_bad)}")

        if len(missing) > 0:
            print("\nFirst 25 missing keys:")
            for k in missing[:25]:
                print("  -", k)

        if len(unexpected) > 0:
            print("\nFirst 25 unexpected keys:")
            for k in unexpected[:25]:
                print("  -", k)

        if len(shape_bad) > 0:
            print("\nFirst 10 shape mismatches (ckpt_shape -> model_shape):")
            for k, s1, s2 in shape_bad[:10]:
                print(f"  - {k}: {s1} -> {s2}")

        raise  # stop execution