from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"
TRACES = DEMO / "data" / "traces"
N_PASSES = 8
CHECK_STATES = {"on", "wait", "off"}
FEATURE_CATEGORIES = {"syntax", "structure", "semantics"}


def fail(message: str) -> None:
    raise SystemExit(f"demo validation failed: {message}")


def read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        fail(f"{path.relative_to(ROOT)} is invalid JSON: {exc}")


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def require_string(value: object, label: str) -> str:
    require(isinstance(value, str) and bool(value), f"{label} must be a non-empty string")
    return value


def validate_trace(slug: str, path: Path) -> None:
    trace = read_json(path)
    require(isinstance(trace, dict), f"{path.relative_to(ROOT)} must be an object")
    require(trace.get("slug") == slug, f"{path.relative_to(ROOT)} slug must match filename")
    require_string(trace.get("prefix"), f"{slug}.prefix")
    require_string(trace.get("caption"), f"{slug}.caption")
    require_string(trace.get("target_smiles"), f"{slug}.target_smiles")

    passes = trace.get("passes")
    require(isinstance(passes, list) and len(passes) == N_PASSES, f"{slug}.passes must contain {N_PASSES} passes")
    for idx, item in enumerate(passes):
        require(isinstance(item, dict), f"{slug}.passes[{idx}] must be an object")
        require(item.get("pass_idx") == idx, f"{slug}.passes[{idx}].pass_idx must equal {idx}")
        require_string(item.get("caption"), f"{slug}.passes[{idx}].caption")
        require_string(item.get("argmax_token"), f"{slug}.passes[{idx}].argmax_token")
        require_string(item.get("completion"), f"{slug}.passes[{idx}].completion")
        require(isinstance(item.get("completion_valid"), bool), f"{slug}.passes[{idx}].completion_valid must be boolean")

        checks = item.get("checks")
        require(isinstance(checks, list) and checks, f"{slug}.passes[{idx}].checks must be a non-empty list")
        for check_idx, check in enumerate(checks):
            require(isinstance(check, dict), f"{slug}.passes[{idx}].checks[{check_idx}] must be an object")
            require_string(check.get("label"), f"{slug}.passes[{idx}].checks[{check_idx}].label")
            require(check.get("state") in CHECK_STATES, f"{slug}.passes[{idx}].checks[{check_idx}].state is unknown")

        logits = item.get("top_logits")
        require(isinstance(logits, list) and logits, f"{slug}.passes[{idx}].top_logits must be a non-empty list")
        for logit_idx, logit in enumerate(logits):
            require(isinstance(logit, dict), f"{slug}.passes[{idx}].top_logits[{logit_idx}] must be an object")
            require("token" in logit, f"{slug}.passes[{idx}].top_logits[{logit_idx}].token is required")
            prob = logit.get("prob")
            require(isinstance(prob, (int, float)) and math.isfinite(prob) and prob >= 0, f"{slug}.passes[{idx}].top_logits[{logit_idx}].prob must be finite and non-negative")

    features = trace.get("features")
    require(isinstance(features, list) and features, f"{slug}.features must be a non-empty list")
    for idx, feature in enumerate(features):
        require(isinstance(feature, dict), f"{slug}.features[{idx}] must be an object")
        require_string(feature.get("slug"), f"{slug}.features[{idx}].slug")
        require_string(feature.get("label"), f"{slug}.features[{idx}].label")
        require(feature.get("category") in FEATURE_CATEGORIES, f"{slug}.features[{idx}].category is unknown")
        active_passes = feature.get("active_passes")
        require(isinstance(active_passes, list), f"{slug}.features[{idx}].active_passes must be a list")
        require(all(isinstance(pass_idx, int) and 0 <= pass_idx < N_PASSES for pass_idx in active_passes), f"{slug}.features[{idx}].active_passes has an out-of-range pass")
        strengths = feature.get("strengths")
        require(isinstance(strengths, list) and len(strengths) == N_PASSES, f"{slug}.features[{idx}].strengths must contain {N_PASSES} values")
        require(all(isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1 for value in strengths), f"{slug}.features[{idx}].strengths must be finite values in [0, 1]")


def main() -> None:
    require((DEMO / "index.html").is_file(), "demo/index.html is missing")
    require((DEMO / ".nojekyll").is_file(), "demo/.nojekyll is missing")
    require((DEMO / "vendor" / "rdkit" / "RDKit_minimal.js").is_file(), "vendored RDKit JS is missing")
    require((DEMO / "vendor" / "rdkit" / "RDKit_minimal.wasm").is_file(), "vendored RDKit WASM is missing")
    require(not (DEMO / "data" / "manifest.json").exists(), "demo/data/manifest.json must not be published")
    require(not (DEMO / "data" / "sae_labels.json").exists(), "demo/data/sae_labels.json must not be published")

    index = read_json(TRACES / "index.json")
    require(isinstance(index, dict), "trace index must be an object")
    examples = index.get("examples")
    require(isinstance(examples, list) and examples, "trace index must contain examples")
    seen: set[str] = set()
    for idx, example in enumerate(examples):
        require(isinstance(example, dict), f"index.examples[{idx}] must be an object")
        slug = require_string(example.get("slug"), f"index.examples[{idx}].slug")
        require(slug not in seen, f"duplicate trace slug {slug}")
        seen.add(slug)
        require_string(example.get("prefix"), f"index.examples[{idx}].prefix")
        require_string(example.get("caption"), f"index.examples[{idx}].caption")
        validate_trace(slug, TRACES / f"{slug}.json")

    trace_files = {path.stem for path in TRACES.glob("*.json") if path.name != "index.json"}
    require(trace_files == seen, f"trace files {sorted(trace_files)} do not match index slugs {sorted(seen)}")
    print(f"validated {len(seen)} demo traces")


if __name__ == "__main__":
    main()
