"""Evaluate S1/S2 escalation policies offline from one run that scored both paths on every row.

Each policy ranks rows by how safely S1 can answer them; accepting the top fraction and
escalating the rest traces an accuracy-coverage curve between always-S2 (coverage 0) and
always-S1 (coverage 1). Policies:

- confidence: calibrated S1 top probability (temperature fit on dev S1 logits)
- confidence_rot: the same on the rotation-debiased S1 readout; accepted rows answer with the
  debiased argmax, so its S1 correctness is the debiased one (needs a --rotations run)
- probe: closed-form ridge on S1 hidden states, trained on dev to predict S1 correctness
  (needs --features); like confidence, it reads the S1 forward, so it is not a pre-router
- router_zero_shot: the same model's SemIf readout of "decidable at a glance?" from the query alone
- router_supervised: logistic regression on hashed query n-grams, trained on dev to predict S1 correctness
- router_probe: the probe on hidden states at the end of the shared prefix, before the model answers
  (needs --prefork-features); a pre-router that sees the same model's reading of the query
- <name>_confidence for every --external NAME=PATH (e.g. jev): that system's own top probability,
  with a dev-fit temperature; accepted rows answer with its argmax and escalated rows go to our S2
- combined: logistic regression on the logits of confidence_rot and router_zero_shot, trained on
  dev to predict the debiased S1's correctness (dev scores cross-fitted by group); accepted rows
  answer with the debiased argmax (needs --rotations and --router; added after the release runs were read, so exploratory)
- oracle: escalate exactly the rows where S1 is wrong and S2 is right

Operating points are chosen on dev (target = dev always-S2 accuracy minus a tolerance) and
applied unchanged to test.

With anytime readouts (`thinkjev-s2 readout --checkpoints`), the report adds `anytime_test`: S2
accuracy and calibration at each thought length t (the answer under a think budget of t), and the
anytime policy, which reads S1, then S2 at successive checkpoints, and stops at the first answer
whose calibrated confidence reaches tau (each checkpoint has its own dev temperature). It is
compared with the gate on accuracy against mean thought tokens. numpy only; no model is loaded.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from thinkjev import calibration

POLICIES = ("confidence", "confidence_rot", "probe", "router_probe", "router_zero_shot", "router_supervised",
            "combined", "oracle")
# Which S1 answer a policy serves when it accepts a row.
S1_KEY = {"confidence_rot": "s1_rot_correct", "combined": "s1_rot_correct"}
COMBINED_FEATURES = ("confidence_rot", "router_zero_shot")
TOLERANCES = (0.0, 0.01, 0.02)
RISKS = (0.01, 0.02, 0.05, 0.10)
DELTA = 0.05  # Learn-then-Test failure probability
FEATURES = 1 << 13
PRIOR_STRENGTH = 0.75  # AnyJev's batch-calibration default; 1.0 divides out the full prior
RIDGE_SCALES = (0.01, 0.1, 1.0, 10.0, 100.0)  # multiples of the feature dimension


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_features(path: Path) -> list[np.ndarray]:
    """Every np.save record the scorer appended, in output-row order."""
    records = []
    with path.open("rb") as stream:
        while stream.peek(1):
            records.append(np.load(stream, allow_pickle=False))
    return records


def load(rows_path: Path, cascade_path: Path, router_path: Path | None, features_path: Path | None = None,
         prefork_path: Path | None = None, externals: dict[str, Path] | None = None) -> list[dict]:
    gold = {row["id"]: row for row in _jsonl(rows_path)}
    external = {name: {pred["id"]: pred for pred in _jsonl(path)} for name, path in (externals or {}).items()}
    router = {}
    if router_path is not None:
        for pred in _jsonl(router_path):
            router[pred["id"].removeprefix("router/")] = pred
    predictions = _jsonl(cascade_path)
    features = read_features(features_path) if features_path is not None else None
    if features is not None and len(features) != len(predictions):
        raise ValueError(f"{len(features)} feature records for {len(predictions)} predictions")
    prefork = read_features(prefork_path) if prefork_path is not None else None
    if prefork is not None and len(prefork) != len(predictions):
        raise ValueError(f"{len(prefork)} pre-fork feature records for {len(predictions)} predictions")
    items = []
    for position, pred in enumerate(predictions):
        row = gold.get(pred["id"])
        if row is None:
            raise ValueError(f"{pred['id']}: prediction has no gold row")
        if "s2_option_logits" not in pred:
            raise ValueError(f"{pred['id']}: evaluation needs S2 on every row (run with --s2 always)")
        if pred["option_ids"] != [option["id"] for option in row["options"]]:
            raise ValueError(f"{pred['id']}: option order differs from gold")
        truth = row["label"]
        s1 = np.asarray(pred["option_logits"], dtype=float)
        s2 = np.asarray(pred["s2_option_logits"], dtype=float)
        item = {
            "id": row["id"],
            "split": row["cascade_split"],
            "source": row["cascade_source"],
            "family": row["family"],
            "lang": row.get("lang", "en"),
            "difficulty": row.get("difficulty"),
            "group": f"{row['cascade_source']}/{row['group_id']}",
            "text": " ".join([json.dumps(row["state"], ensure_ascii=False), row["question"],
                              *[option["description"] for option in row["options"]]]),
            "logits": s1,
            "s2_logits": s2,
            "true_index": truth,
            "s1_correct": int(s1.argmax() == truth),
            "s2_correct": int(s2.argmax() == truth),
            "think_tokens": pred["s2_think_tokens"],
            "closed": pred["s2_closed_naturally"],
            "prefill": pred["prefill_seconds"],
            "copy": pred["copy_seconds"],
            "s1_seconds": pred["s1_seconds"],
            # None for two-stage batch S2; latency then comes from a separate timing run.
            "s2_seconds": pred.get("s2_seconds"),
        }
        if "rotation_option_logits" in pred:
            rotated = np.asarray(pred["rotation_option_logits"], dtype=float)
            item["logits_rot"] = np.asarray(pred["debiased_logits"], dtype=float)
            item["s1_rot_correct"] = int(item["logits_rot"].argmax() == truth)
            item["order_flip"] = int(len(set(rotated.argmax(axis=1).tolist())) > 1)
            item["rotation_seconds"] = pred["rotation_seconds"]
            item["option_key"] = (row["cascade_source"], row["family"], tuple(pred["option_ids"]))
        if features is not None:
            item["features"] = features[position].astype(float)
        if prefork is not None:
            item["prefork_features"] = prefork[position].astype(float)
        if "s2_anytime" in pred:
            item["anytime"] = {read["t"]: np.asarray(read["option_logits"], dtype=float) for read in pred["s2_anytime"]}
        for name, preds in external.items():
            pred_x = preds.get(row["id"])
            if pred_x is None:
                raise ValueError(f"{row['id']}: missing from external {name}")
            if pred_x["option_ids"] != pred["option_ids"]:
                raise ValueError(f"{row['id']}: external {name} option order differs from gold")
            item[f"ext_{name}_logits"] = np.asarray(pred_x["option_logits"], dtype=float)
            item[f"{name}_correct"] = int(item[f"ext_{name}_logits"].argmax() == truth)
        if router:
            glance = router[row["id"]]
            item["router_glance"] = glance["probabilities"][glance["option_ids"].index("glance")]
            item["router_seconds"] = glance["total_seconds"]
        items.append(item)
    if not items:
        raise ValueError("No cascade predictions")
    return items


def _features(text: str) -> np.ndarray:
    tokens = re.findall(r"\w+", text.lower())
    vector = np.zeros(FEATURES)
    for gram in tokens + [a + " " + b for a, b in zip(tokens, tokens[1:])]:
        vector[zlib.crc32(gram.encode()) % (FEATURES - 1)] = 1.0
    # Unit norm keeps gradient descent inside its stability bound however many n-grams a row has.
    vector /= max(np.linalg.norm(vector), 1.0)
    vector[-1] = np.log1p(len(tokens)) / 10  # length; last bucket is reserved for it
    return vector


def fit_logistic(x: np.ndarray, y: np.ndarray, l2=1e-2, steps=500, rate=1.0) -> np.ndarray:
    weights = np.zeros(x.shape[1] + 1)
    design = np.hstack([x, np.ones((len(x), 1))])
    # Apple Accelerate raises spurious FP flags in matmul even for finite results; scores are
    # checked for finiteness before use instead.
    with np.errstate(all="ignore"):
        for _ in range(steps):
            p = 1 / (1 + np.exp(-design @ weights))
            grad = design.T @ (p - y) / len(y) + l2 * np.r_[weights[:-1], 0.0]
            weights -= rate * grad
    return weights


def predict_logistic(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        return 1 / (1 + np.exp(-(np.hstack([x, np.ones((len(x), 1))]) @ weights)))


def fit_ridge(x: np.ndarray, y: np.ndarray, penalty: float):
    """Closed-form ridge in dual form (rows << features) on standardized features."""
    mean, scale = x.mean(0), x.std(0) + 1e-6
    z = (x - mean) / scale
    offset = y.mean()
    with np.errstate(all="ignore"):  # spurious Accelerate FP flags, as in fit_logistic
        alpha = np.linalg.solve(z @ z.T + penalty * np.eye(len(z)), y - offset)
        weights = z.T @ alpha
    if not np.all(np.isfinite(weights)):
        raise ValueError("Ridge probe produced non-finite weights")
    return mean, scale, weights, offset


def predict_ridge(model, x: np.ndarray) -> np.ndarray:
    mean, scale, weights, offset = model
    with np.errstate(all="ignore"):
        return (x - mean) / scale @ weights + offset


def probe_scores(dev: list[dict], test: list[dict], folds=5, key: str = "features", policy: str = "probe") -> dict:
    """Pick (layer, penalty) by group-CV AUROC on dev; cross-fitted dev scores, full-dev fit for test.

    The selection sees the same dev folds whose scores it returns, so dev scores are mildly
    optimistic; test rows never influence the fit or the selection.
    """
    y = np.array([item["s1_correct"] for item in dev], dtype=float)
    fold_of = calibration.fold_map([{"group": item["group"]} for item in dev], folds)
    fold = np.array([fold_of[item["group"]] for item in dev])
    layers, width = dev[0][key].shape
    best = None
    for layer in range(layers):
        x = np.stack([item[key][layer] for item in dev])
        for multiple in RIDGE_SCALES:
            held = np.full(len(dev), np.nan)
            for index in range(folds):
                part = fold == index
                if part.any() and (~part).any():
                    held[part] = predict_ridge(fit_ridge(x[~part], y[~part], multiple * width), x[part])
            area = auroc(held, y) if np.isfinite(held).all() else None
            if area is not None and (best is None or area > best["dev_cv_auroc"]):
                best = {"layer": layer, "penalty_multiple": multiple, "dev_cv_auroc": area, "held": held}
    if best is None:
        raise ValueError("Probe needs both correct and wrong S1 rows in every dev fold's training part")
    for item, score in zip(dev, best.pop("held")):
        item[policy] = float(score)
    if test:
        x = np.stack([item[key][best["layer"]] for item in dev])
        model = fit_ridge(x, y, best["penalty_multiple"] * width)
        for item, score in zip(test, predict_ridge(model, np.stack([item[key][best["layer"]] for item in test]))):
            item[policy] = float(score)
    return best


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def combined_scores(dev: list[dict], test: list[dict], folds=5, policy: str = "combined") -> dict:
    """Logistic regression on logit(confidence_rot) and logit(router_zero_shot) predicting the debiased
    S1's correctness; features standardized on the training part. Dev scores are cross-fitted by
    group, test scores come from the full-dev fit (the router_supervised recipe)."""
    features = lambda part: np.column_stack([_logit(np.array([item[key] for item in part])) for key in COMBINED_FEATURES])
    x_dev, x_test = features(dev), features(test) if test else None
    y = np.array([item["s1_rot_correct"] for item in dev], dtype=float)
    if y.min() == y.max():
        raise ValueError("combined needs both correct and wrong debiased S1 rows on dev")

    def fit(x, labels):
        mean, scale = x.mean(0), x.std(0) + 1e-9
        return mean, scale, fit_logistic((x - mean) / scale, labels, steps=2000)

    def predict(model, x):
        mean, scale, weights = model
        return predict_logistic(weights, (x - mean) / scale)

    fold_of = calibration.fold_map([{"group": item["group"]} for item in dev], folds)
    fold = np.array([fold_of[item["group"]] for item in dev])
    held = np.full(len(dev), np.nan)
    for index in range(folds):
        part = fold == index
        if part.any() and (~part).any():
            held[part] = predict(fit(x_dev[~part], y[~part]), x_dev[part])
    if not np.isfinite(held).all():
        raise ValueError("combined: a dev fold has no training rows")
    for item, score in zip(dev, held):
        item[policy] = float(score)
    mean, scale, weights = fit(x_dev, y)
    if test is not None and len(test):
        for item, score in zip(test, predict((mean, scale, weights), x_test)):
            item[policy] = float(score)
    return {"features": list(COMBINED_FEATURES), "label": "s1_rot_correct",
            "standardized_weights": dict(zip(COMBINED_FEATURES, weights[:-1].tolist())), "intercept": float(weights[-1]),
            "dev_cv_auroc": auroc(held, y)}


def rot_view(items: list[dict], logits_key: str = "logits_rot") -> list[dict]:
    return [{**item, "logits": item[logits_key]} for item in items]


def batch_prior(dev: list[dict], strength: float = PRIOR_STRENGTH) -> dict:
    """Per (source, family, option IDs) mean debiased distribution on dev, raised to `strength`."""
    grouped = defaultdict(list)
    for item in dev:
        grouped[item["option_key"]].append(calibration.softmax(item["logits_rot"]))
    return {key: strength * np.log(np.mean(values, axis=0)) for key, values in grouped.items()}


def assign_scores(items: list[dict], temperature: float, folds=5, temperature_rot: float | None = None,
                  external_temperatures: dict[str, float] | None = None) -> dict:
    """Attach every policy's accept-S1 score. Dev supervised scores are cross-fitted by group."""
    for name, value in (external_temperatures or {}).items():
        for item in items:
            item[f"{name}_confidence"] = float(calibration.softmax(item[f"ext_{name}_logits"] / value).max())
    for item in items:
        item["confidence"] = float(calibration.softmax(item["logits"] / temperature).max())
        rescued = (1 - item["s1_correct"]) * item["s2_correct"]
        item["oracle"] = (1 - rescued) + 1e-3 * item["confidence"]
        if "router_glance" in item:
            item["router_zero_shot"] = item["router_glance"]
        if temperature_rot is not None and "logits_rot" in item:
            item["confidence_rot"] = float(calibration.softmax(item["logits_rot"] / temperature_rot).max())
    dev = [item for item in items if item["split"] == "dev"]
    test = [item for item in items if item["split"] == "test"]
    selection = {}
    for key, policy in (("features", "probe"), ("prefork_features", "router_probe")):
        if all(key in item for item in items):
            try:
                selection[policy] = probe_scores(dev, test, folds, key=key, policy=policy)
            except ValueError as error:  # e.g. S1 all right or all wrong in a fold; the policy is skipped
                selection[policy] = {"skipped": str(error)}
                for item in items:
                    item.pop(policy, None)
    x_dev = np.stack([_features(item["text"]) for item in dev])
    y_dev = np.array([item["s1_correct"] for item in dev], dtype=float)
    fold_of = calibration.fold_map([{"group": item["group"]} for item in dev], folds)
    fold = np.array([fold_of[item["group"]] for item in dev])
    for index in range(folds):
        held = fold == index
        if held.any() and (~held).any():
            weights = fit_logistic(x_dev[~held], y_dev[~held])
            for item, score in zip([d for d, h in zip(dev, held) if h], predict_logistic(weights, x_dev[held])):
                item["router_supervised"] = float(score)
    if test:
        weights = fit_logistic(x_dev, y_dev)
        x_test = np.stack([_features(item["text"]) for item in test])
        for item, score in zip(test, predict_logistic(weights, x_test)):
            item["router_supervised"] = float(score)
    if all(all(key in item for key in COMBINED_FEATURES) for item in items):
        try:
            selection["combined"] = combined_scores(dev, test, folds)
        except ValueError as error:
            selection["combined"] = {"skipped": str(error)}
            for item in items:
                item.pop("combined", None)
    return selection


def curve(scores: np.ndarray, s1: np.ndarray, s2: np.ndarray, weights: np.ndarray | None = None):
    """Coverage and system accuracy at every distinct threshold (ties accepted together)."""
    weights = np.ones(len(scores)) if weights is None else weights
    order = np.argsort(-scores, kind="stable")
    ordered = scores[order]
    ends = np.r_[np.nonzero(np.diff(ordered))[0], len(ordered) - 1]
    total = weights.sum()
    gain = np.cumsum(((s1 - s2) * weights)[order])[ends]
    coverage = np.r_[0.0, np.cumsum(weights[order])[ends] / total]
    accuracy = np.r_[0.0, gain] / total + (s2 * weights).sum() / total
    thresholds = np.r_[np.inf, ordered[ends]]
    return coverage, accuracy, thresholds


def cost_curve(scores: np.ndarray, s1: np.ndarray, s2: np.ndarray, tokens: np.ndarray):
    """Accuracy and mean thought tokens per row at every distinct threshold; escalated rows pay their thought."""
    order = np.argsort(-scores, kind="stable")
    ordered = scores[order]
    ends = np.r_[np.nonzero(np.diff(ordered))[0], len(ordered) - 1]
    n = len(scores)
    accepted_s1, accepted_s2, saved = (np.cumsum(values[order])[ends] for values in (s1, s2, tokens))
    accuracy = np.r_[s2.sum(), s2.sum() - accepted_s2 + accepted_s1] / n
    spent = np.r_[tokens.sum(), tokens.sum() - saved] / n
    return accuracy, spent


def max_coverage(coverage, accuracy, target) -> float:
    ok = accuracy >= target - 1e-12
    return float(coverage[ok].max()) if ok.any() else 0.0


def _s1_key(policy: str) -> str:
    if policy in S1_KEY:
        return S1_KEY[policy]
    if policy.endswith("_confidence") and policy != "confidence":
        return policy.removesuffix("_confidence") + "_correct"
    return "s1_correct"


def _arrays(items: list[dict], policy: str):
    s1_key = _s1_key(policy)
    return (np.array([item[policy] for item in items]), np.array([item[s1_key] for item in items], float),
            np.array([item["s2_correct"] for item in items], float))


def latency(items: list[dict], accept: np.ndarray, policy: str) -> dict:
    if any(item["s2_seconds"] is None for item in items) or policy.endswith("_confidence") and policy != "confidence":
        # External systems' latency (network included) is reported by their own runner.
        return {}
    prefill = np.array([item["prefill"] for item in items])
    # confidence_rot pays the extra shifts; combined pays them and the router read on the same prefix.
    s1 = np.array([item["s1_seconds"] + (item["rotation_seconds"] if policy in ("confidence_rot", "combined") else 0.0)
                   + (item.get("router_seconds", 0.0) if policy == "combined" else 0.0) for item in items])
    s2 = np.array([item["s2_seconds"] for item in items])
    copy = np.array([item["copy"] for item in items])
    escalate = ~accept
    if policy.startswith("router"):
        router = np.array([item.get("router_seconds", 0.0) for item in items])
        routed = router + prefill + np.where(accept, s1, s2)
        return {"mean_seconds": float(routed.mean())}
    return {
        "mean_seconds_shared_prefix": float((prefill + copy + s1 + escalate * s2).mean()),
        "mean_seconds_no_reuse": float((prefill + s1 + escalate * (prefill + s2)).mean()),
    }


def operating_points(dev: list[dict], test: list[dict], policies: list[str]) -> list[dict]:
    s2_dev = np.mean([item["s2_correct"] for item in dev])
    points = []
    for tolerance in TOLERANCES:
        target = s2_dev - tolerance
        for policy in policies:
            scores, s1, s2 = _arrays(dev, policy)
            coverage, accuracy, thresholds = curve(scores, s1, s2)
            ok = np.nonzero(accuracy >= target - 1e-12)[0]
            threshold = float(thresholds[ok[np.argmax(coverage[ok])]]) if len(ok) else float("inf")
            scores, s1, s2 = _arrays(test, policy)
            accept = scores >= threshold
            points.append({
                "tolerance": tolerance,
                "dev_target_accuracy": float(target),
                "policy": policy,
                "threshold": threshold,
                "test_coverage": float(accept.mean()),
                "test_accuracy": float(np.where(accept, s1, s2).mean()),
                "test_mean_think_tokens": float(np.mean([(not a) * item["think_tokens"] for a, item in zip(accept, test)])),
                **latency(test, accept, policy),
            })
    return points


def matched_coverage(test: list[dict], policies: list[str], samples: int, seed=217) -> list[dict]:
    """Test-curve coverage at always-S2 accuracy minus a tolerance, with group-bootstrap CIs."""
    groups = sorted({item["group"] for item in test})
    index = {group: position for position, group in enumerate(groups)}
    member = np.array([index[item["group"]] for item in test])
    arrays = {policy: _arrays(test, policy) for policy in policies}
    s2 = arrays[policies[0]][2]
    rng = np.random.default_rng(seed)
    draws = [np.bincount(rng.integers(len(groups), size=len(groups)), minlength=len(groups))[member].astype(float)
             for _ in range(samples)]
    result = []
    for tolerance in TOLERANCES:
        entry = {"tolerance": tolerance, "target_accuracy": float(s2.mean() - tolerance), "coverage": {}, "difference_vs_confidence": {}}
        boot = {}
        for policy in policies:
            scores, s1, s2p = arrays[policy]
            coverage, accuracy, _ = curve(scores, s1, s2p)
            entry["coverage"][policy] = max_coverage(coverage, accuracy, s2.mean() - tolerance)
            values = []
            for weights in draws:
                target = (s2p * weights).sum() / weights.sum() - tolerance
                coverage, accuracy, _ = curve(scores, s1, s2p, weights)
                values.append(max_coverage(coverage, accuracy, target))
            boot[policy] = np.array(values)
        for policy in policies:
            if policy != "confidence":
                delta = boot["confidence"] - boot[policy]
                entry["difference_vs_confidence"][policy] = {
                    "confidence_minus_policy": entry["coverage"]["confidence"] - entry["coverage"][policy],
                    "ci95": [float(np.quantile(delta, 0.025)), float(np.quantile(delta, 0.975))],
                }
        result.append(entry)
    return result


def normalized_area(scores: np.ndarray, s1: np.ndarray, s2: np.ndarray, weights: np.ndarray | None = None) -> float | None:
    """(policy - random) / (oracle - random) area under the accuracy-coverage curve.

    Random and oracle use the same S1 answers as the policy, so a policy that serves a different S1
    answer (debiased, a draft, another system) is compared with its own endpoints.
    """
    weights = np.ones(len(scores)) if weights is None else weights
    area = lambda values: float(np.trapezoid(*curve(values, s1, s2, weights)[1::-1]))
    total = weights.sum()
    random = ((s1 * weights).sum() + (s2 * weights).sum()) / (2 * total)
    best = area(s1 - s2)
    return None if best - random <= 1e-12 else (area(scores) - random) / (best - random)


def normalized_areas(test: list[dict], policies: list[str], samples: int, primary: str, seed=217) -> dict:
    """Normalized areas on test, with group-bootstrap CIs of each policy's difference from `primary`."""
    groups = sorted({item["group"] for item in test})
    index = {group: position for position, group in enumerate(groups)}
    member = np.array([index[item["group"]] for item in test])
    rng = np.random.default_rng(seed)
    draws = [np.bincount(rng.integers(len(groups), size=len(groups)), minlength=len(groups))[member].astype(float)
             for _ in range(samples)]
    arrays = {policy: _arrays(test, policy) for policy in policies if policy != "oracle"}
    result = {"primary": primary, "area": {}, "difference_vs_primary": {}}
    boot = {}
    for policy, (scores, s1, s2) in arrays.items():
        result["area"][policy] = normalized_area(scores, s1, s2)
        boot[policy] = np.array([normalized_area(scores, s1, s2, weights) or np.nan for weights in draws])
    for policy in arrays:
        if policy != primary and primary in boot:
            delta = boot[primary] - boot[policy]
            delta = delta[np.isfinite(delta)]
            result["difference_vs_primary"][policy] = {
                "primary_minus_policy": (None if result["area"][primary] is None or result["area"][policy] is None
                                         else result["area"][primary] - result["area"][policy]),
                "ci95": [float(np.quantile(delta, 0.025)), float(np.quantile(delta, 0.975))] if len(delta) else None,
            }
    return result


def auroc(scores: np.ndarray, positive: np.ndarray) -> float | None:
    positives, negatives = scores[positive == 1], scores[positive == 0]
    if not len(positives) or not len(negatives):
        return None
    wins = (positives[:, None] > negatives[None, :]).sum() + 0.5 * (positives[:, None] == negatives[None, :]).sum()
    return float(wins / (len(positives) * len(negatives)))


def risk_threshold(scores: np.ndarray, correct: np.ndarray, risk: float) -> float:
    """Lowest confidence threshold whose accepted rows have error at most `risk` (ties together)."""
    order = np.argsort(-scores, kind="stable")
    ordered = scores[order]
    ends = np.r_[np.nonzero(np.diff(ordered))[0], len(ordered) - 1]
    errors = np.cumsum(1 - correct[order])[ends] / (ends + 1)
    ok = np.nonzero(errors <= risk + 1e-12)[0]
    return float(ordered[ends[ok[-1]]]) if len(ok) else float("inf")


def binomial_cdf(k: int, n: int, p: float) -> float:
    if p <= 0:
        return 1.0
    if p >= 1:
        return float(k >= n)
    log_p, log_q = math.log(p), math.log1p(-p)
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * log_p + (n - i) * log_q
             for i in range(k + 1)]
    top = max(terms)
    return min(1.0, math.exp(top) * sum(math.exp(term - top) for term in terms))


def clopper_pearson_upper(errors: int, n: int, delta: float) -> float:
    """One-sided (1 - delta) upper confidence bound on an error rate."""
    if n == 0 or errors >= n:
        return 1.0
    low, high = errors / n, 1.0
    for _ in range(50):
        middle = (low + high) / 2
        if binomial_cdf(errors, n, middle) > delta:
            low = middle
        else:
            high = middle
    return high


def ltt_threshold(scores: np.ndarray, correct: np.ndarray, groups: list[str], risk: float,
                  delta: float = DELTA) -> float:
    """Learn-then-Test with fixed-sequence testing: loosen the threshold while the bound holds.

    Candidate thresholds are tested from strictest to loosest and testing stops at the first
    failure, so the whole sequence keeps a (1 - delta) guarantee without a multiplicity correction.
    Rows sharing a group are not independent, so the effective sample size is the number of
    distinct groups accepted, with the accepted error rate scaled to it (rounded up). The sequence
    starts at the first threshold with enough groups that zero errors could pass; that start
    depends only on counts, not labels, so the guarantee is unaffected.
    """
    minimum = math.log(delta) / math.log1p(-risk)
    order = np.argsort(-scores, kind="stable")
    ordered = scores[order]
    ends = np.r_[np.nonzero(np.diff(ordered))[0], len(ordered) - 1]
    threshold, seen, errors, start = float("inf"), set(), 0, 0
    for end in ends:
        for index in order[start:end + 1]:
            seen.add(groups[index])
            errors += int(1 - correct[index])
        start = end + 1
        effective = len(seen)
        if effective < minimum:
            continue
        scaled = math.ceil(errors / (end + 1) * effective - 1e-9)
        if clopper_pearson_upper(scaled, effective, delta) > risk:
            break
        threshold = float(ordered[end])
    return threshold


def s1_readouts(dev: list[dict], test: list[dict]) -> dict:
    """S1 alone as a selective classifier: raw, rotation-debiased, debiased with a batch prior, and
    every external system (e.g. Jev).

    Each readout fits its own temperature on dev; coverage-at-risk thresholds are chosen on dev
    (point estimate, and Learn-then-Test with a (1 - DELTA) guarantee) and applied unchanged to test.
    """
    views = {"raw": (dev, test)}
    for key in sorted(key for key in dev[0] if key.startswith("ext_") and key.endswith("_logits")):
        external = lambda part, key=key: [{**item, "logits": item[key]} for item in part]
        views[key.removeprefix("ext_").removesuffix("_logits")] = (external(dev), external(test))
    if all("logits_rot" in item for item in dev + test):
        views["rotation"] = (rot_view(dev), rot_view(test))
        prior = batch_prior(dev)
        corrected = lambda part: [{**item, "logits": item["logits_rot"] - prior.get(item["option_key"], 0.0)}
                                  for item in part]
        views["rotation_batch_prior"] = (corrected(dev), corrected(test))
    report = {}
    for name, (dev_view, test_view) in views.items():
        temperature = calibration.fit_temperature(dev_view)
        dev_rows, test_rows = calibration.scored(dev_view, temperature), calibration.scored(test_view, temperature)
        entry = {"temperature": temperature, "accuracy": float(np.mean([row["correct"] for row in test_rows])),
                 "ece": calibration.ece(test_rows), "ece_t1": calibration.ece(calibration.scored(test_view, 1.0)),
                 "coverage_at_risk": [], "accuracy_by_source": {}}
        dev_scores = np.array([row["confidence"] for row in dev_rows])
        dev_correct = np.array([row["correct"] for row in dev_rows], float)
        correct = np.array([row["correct"] for row in test_rows], float)
        for risk in RISKS:
            threshold = risk_threshold(dev_scores, dev_correct, risk)
            guaranteed = ltt_threshold(dev_scores, dev_correct, [row["group"] for row in dev_rows], risk)
            accept = np.array([row["confidence"] >= threshold for row in test_rows])
            safe = np.array([row["confidence"] >= guaranteed for row in test_rows])
            entry["coverage_at_risk"].append({
                "risk": risk, "dev_threshold": threshold, "test_coverage": float(accept.mean()),
                "test_risk": float(1 - correct[accept].mean()) if accept.any() else None,
                "ltt_delta": DELTA, "ltt_dev_threshold": guaranteed, "ltt_test_coverage": float(safe.mean()),
                "ltt_test_risk": float(1 - correct[safe].mean()) if safe.any() else None,
            })
        for field in ("source", "lang"):
            parts = defaultdict(list)
            for item, row in zip(test_view, test_rows):
                parts[item[field]].append(row["correct"])
            entry[f"accuracy_by_{field}"] = {key: float(np.mean(values)) for key, values in sorted(parts.items())}
        report[name] = entry
    return report


def checkpoint_logits(item: dict, t: int) -> np.ndarray:
    """S2 readout under a think budget of t: the checkpoint read, or the final read once the thought ended."""
    if t >= item["think_tokens"]:
        return item["s2_logits"]
    if t not in item["anytime"]:
        raise ValueError(f"{item['id']}: no anytime readout at t={t} for a {item['think_tokens']}-token thought")
    return item["anytime"][t]


def anytime_outcomes(items: list[dict], gate: str, stages: list[tuple[int, float]], tau: float):
    """Correctness and thought tokens per row: S1 if the gate reaches tau, else the first checkpoint
    (t, temperature) whose calibrated confidence reaches tau, else the full thought."""
    s1_key = _s1_key(gate)
    correct, tokens = [], []
    for item in items:
        if item[gate] >= tau:
            correct.append(item[s1_key])
            tokens.append(0)
            continue
        answer = (item["s2_correct"], item["think_tokens"])
        for t, temperature in stages:
            if t >= item["think_tokens"]:
                break
            probs = calibration.softmax(item["anytime"][t] / temperature)
            if probs.max() >= tau:
                answer = (int(probs.argmax() == item["true_index"]), t)
                break
        correct.append(answer[0])
        tokens.append(answer[1])
    return np.array(correct, float), np.array(tokens, float)


def gate_outcomes(items: list[dict], gate: str, tau: float):
    s1_key = _s1_key(gate)
    accept = np.array([item[gate] >= tau for item in items])
    correct = np.where(accept, [item[s1_key] for item in items], [item["s2_correct"] for item in items])
    tokens = np.where(accept, 0, [item["think_tokens"] for item in items])
    return correct.astype(float), tokens.astype(float)


def anytime_report(dev: list[dict], test: list[dict], gate: str, grid: int = 60) -> dict:
    """S2 at every checkpoint, and the anytime policy against the gate in thought-token space."""
    checkpoints = sorted(set().union(*(item["anytime"] for item in dev + test)))
    per_checkpoint, stages = [], []
    for t in checkpoints:
        view = lambda part: [{**item, "logits": checkpoint_logits(item, t)} for item in part]
        temperature = calibration.fit_temperature(view(dev))
        rows = calibration.scored(view(test), temperature)
        stages.append((t, temperature))
        per_checkpoint.append({
            "t": t, "temperature": temperature,
            "accuracy": float(np.mean([row["correct"] for row in rows])),
            "ece": calibration.ece(rows),
            "mean_think_tokens": float(np.mean([min(t, item["think_tokens"]) for item in test])),
        })
    s2_temperature = calibration.fit_temperature(rot_view(dev, "s2_logits"))
    rows = calibration.scored(rot_view(test, "s2_logits"), s2_temperature)
    final = {"temperature": s2_temperature, "accuracy": float(np.mean([row["correct"] for row in rows])),
             "ece": calibration.ece(rows),
             "mean_think_tokens": float(np.mean([item["think_tokens"] for item in test]))}
    taus = np.unique(np.r_[np.quantile([item[gate] for item in dev + test], np.linspace(0, 1, grid)), np.inf])
    curves = {}
    for name, outcome in (("anytime", lambda part, tau: anytime_outcomes(part, gate, stages, tau)),
                          ("gate", lambda part, tau: gate_outcomes(part, gate, tau))):
        accuracy, spent = zip(*((correct.mean(), tokens.mean()) for correct, tokens in
                                (outcome(test, tau) for tau in taus)))
        curves[name] = {"tau": [float(tau) for tau in taus], "accuracy": [float(a) for a in accuracy],
                        "mean_think_tokens": [float(s) for s in spent]}
    s2_dev = np.mean([item["s2_correct"] for item in dev])
    points = []
    for tolerance in TOLERANCES:
        for name, outcome in (("anytime", lambda part, tau: anytime_outcomes(part, gate, stages, tau)),
                              ("gate", lambda part, tau: gate_outcomes(part, gate, tau))):
            on_dev = [(outcome(dev, tau), tau) for tau in taus]
            feasible = [(tokens.mean(), tau) for (correct, tokens), tau in on_dev
                        if correct.mean() >= s2_dev - tolerance - 1e-12]
            tau = min(feasible)[1] if feasible else float("inf")
            correct, tokens = outcome(test, tau)
            points.append({"tolerance": tolerance, "policy": name, "tau": float(tau),
                           "test_accuracy": float(correct.mean()), "test_mean_think_tokens": float(tokens.mean())})
    return {"gate": gate, "checkpoints": checkpoints, "per_checkpoint": per_checkpoint, "full_thought": final,
            "curves": curves, "dev_selected_operating_points": points}


def summarize(items: list[dict], policies: list[str], temperature: float, samples: int) -> dict:
    dev = [item for item in items if item["split"] == "dev"]
    test = [item for item in items if item["split"] == "test"]
    families = defaultdict(list)
    for item in test:
        families[(item["source"], item["family"], item["lang"], item["difficulty"])].append(item)
    points = operating_points(dev, test, policies)
    gate = "confidence_rot" if "confidence_rot" in policies else "confidence"
    gate_threshold = next(point["threshold"] for point in points if point["policy"] == gate and point["tolerance"] == 0.01)
    curves = {}
    for policy in policies:
        coverage, accuracy, _ = curve(*_arrays(test, policy))
        curves[policy] = {"coverage": coverage.round(6).tolist(), "accuracy": accuracy.round(6).tolist(),
                          "area": float(np.trapezoid(accuracy, coverage))}
    return {
        "rows": {"dev": len(dev), "test": len(test)},
        "temperature": temperature,
        "baselines_test": {
            "always_s1_accuracy": float(np.mean([item["s1_correct"] for item in test])),
            "always_s2_accuracy": float(np.mean([item["s2_correct"] for item in test])),
            "oracle_best_accuracy": float(np.mean([max(item["s1_correct"], item["s2_correct"]) for item in test])),
            "s1_wrong_s2_right": int(sum((1 - item["s1_correct"]) * item["s2_correct"] for item in test)),
            "s1_right_s2_wrong": int(sum(item["s1_correct"] * (1 - item["s2_correct"]) for item in test)),
            "s2_closed_naturally_rate": float(np.mean([item["closed"] for item in test])),
            "s2_mean_think_tokens": float(np.mean([item["think_tokens"] for item in test])),
            **{f"always_s1_{key}": value for key, value in latency(test, np.ones(len(test), bool), "confidence").items()},
            "always_s2_mean_seconds": (float(np.mean([item["prefill"] + item["s2_seconds"] for item in test]))
                                       if all(item["s2_seconds"] is not None for item in test) else None),
        },
        "s1_calibration_test": {
            "ece_t1": calibration.ece(calibration.scored(test, 1.0)),
            "ece_fitted": calibration.ece(calibration.scored(test, temperature)),
        },
        "auroc_s1_correct_test": {policy: auroc(*_arrays(test, policy)[:2]) for policy in policies if policy != "oracle"},
        "s1_readouts_test": s1_readouts(dev, test),
        "order_flip_rate_test": (
            {"all": float(np.mean([item["order_flip"] for item in test])),
             **{source: float(np.mean([item["order_flip"] for item in test if item["source"] == source]))
                for source in sorted({item["source"] for item in test})}}
            if all("order_flip" in item for item in test) else None
        ),
        "matched_accuracy_coverage_test": matched_coverage(test, policies, samples),
        "normalized_area_test": normalized_areas(test, policies, samples, gate),
        "disagreement_test": {
            "by_source": {
                source: {"s1_wrong_s2_right": int(sum((1 - i["s1_correct"]) * i["s2_correct"] for i in test if i["source"] == source)),
                         "s1_right_s2_wrong": int(sum(i["s1_correct"] * (1 - i["s2_correct"]) for i in test if i["source"] == source))}
                for source in sorted({item["source"] for item in test})
            },
        },
        "dev_selected_operating_points": points,
        "per_family_test_gate": {"policy": gate, "tolerance": 0.01, "dev_threshold": gate_threshold},
        "per_family_test": [
            {
                "source": source,
                "family": family,
                "lang": lang,
                "difficulty": difficulty,
                "rows": len(part),
                "escalation_rate": float(np.mean([item[gate] < gate_threshold for item in part])),
                "system_accuracy": float(np.mean([item[_s1_key(gate)] if item[gate] >= gate_threshold else item["s2_correct"]
                                                  for item in part])),
                "s1_accuracy": float(np.mean([item["s1_correct"] for item in part])),
                "s2_accuracy": float(np.mean([item["s2_correct"] for item in part])),
                "mean_confidence": float(np.mean([item["confidence"] for item in part])),
                "mean_think_tokens": float(np.mean([item["think_tokens"] for item in part])),
            }
            for (source, family, lang, difficulty), part in sorted(families.items(), key=lambda pair: str(pair[0]))
        ],
        "curves_test": curves,
        "cost_curves_test": {
            policy: dict(zip(("accuracy", "mean_think_tokens"),
                             (values.round(6).tolist() for values in cost_curve(
                                 *_arrays(test, policy), np.array([item["think_tokens"] for item in test], float)))))
            for policy in policies
        },
        "anytime_test": anytime_report(dev, test, gate) if all("anytime" in item for item in items) else None,
        "row_scores": [
            {"id": item["id"], "split": item["split"], **{policy: item[policy] for policy in policies}}
            for item in items
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, required=True, help="Assembled cascade rows with gold labels and splits")
    parser.add_argument("--cascade", type=Path, required=True, help="thinkjev-score --s2 always output")
    parser.add_argument("--router", type=Path, help="SemIf semif-score --mode direct output on the router rows")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--features", type=Path, help="thinkjev-score --features output of the same cascade run")
    parser.add_argument("--prefork-features", type=Path, help="thinkjev-score --prefork-features output, same run")
    parser.add_argument("--external", action="append", default=[], metavar="NAME=PATH",
                        help="Another system's option probabilities for the same rows, e.g. jev=build/jev.jsonl")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Report must be new")
    externals = dict(item.split("=", 1) for item in args.external)
    if any(not re.fullmatch(r"[a-z][a-z0-9]*", name) for name in externals):
        parser.error("External names must be lowercase alphanumeric")
    items = load(args.rows, args.cascade, args.router, args.features, args.prefork_features,
                 {name: Path(path) for name, path in externals.items()})
    dev = [item for item in items if item["split"] == "dev"]
    if not dev or len(dev) == len(items):
        parser.error("Need both dev and test rows")
    temperature = calibration.fit_temperature(dev)
    temperature_rot = calibration.fit_temperature(rot_view(dev)) if all("logits_rot" in item for item in items) else None
    external_temperatures = {name: calibration.fit_temperature(rot_view(dev, f"ext_{name}_logits")) for name in externals}
    selection = assign_scores(items, temperature, temperature_rot=temperature_rot,
                              external_temperatures=external_temperatures)
    policies = [policy for policy in (*POLICIES[:-1], *(f"{name}_confidence" for name in externals), POLICIES[-1])
                if all(policy in item for item in items)]
    for policy in policies:
        if not all(np.isfinite(item[policy]) for item in items):
            raise ValueError(f"{policy}: non-finite scores")
    report = {
        "schema": "thinkjev-eval-v3",
        "inputs": {"rows": str(args.rows), "cascade": str(args.cascade), "router": str(args.router) if args.router else None,
                   "features": str(args.features) if args.features else None,
                   "prefork_features": str(args.prefork_features) if args.prefork_features else None,
                   "externals": externals},
        "policies": policies,
        "temperature_rot": temperature_rot,
        "s2_temperature": calibration.fit_temperature(rot_view(dev, "s2_logits")),
        "external_temperatures": external_temperatures,
        "probe_selection": selection.get("probe"),
        "router_probe_selection": selection.get("router_probe"),
        "combined_fit": selection.get("combined"),
        **summarize(items, policies, temperature, args.bootstrap),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("rows", "temperature", "baselines_test")}, indent=2))
    for entry in report["matched_accuracy_coverage_test"]:
        print(f"tolerance {entry['tolerance']:.2f}: " + ", ".join(f"{k}={v:.3f}" for k, v in entry["coverage"].items()))


if __name__ == "__main__":
    main()
