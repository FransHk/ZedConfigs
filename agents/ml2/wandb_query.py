from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import socket
import sys
import time
from pathlib import Path
from typing import Any

import wandb

from common.wandb.wandb_constants import WANDB_ENTITY_NAME, WANDB_PROJECT_NAME

SEGMENT_SUMMARY_METRICS: tuple[str, str, str] = (
    "segment PPK-AUC @ TopK=500 (validation)",
    "segment mTPR in [0,0.001] FPR (validation)",
    "segment PPK-AUC @ TopK=250 (validation)",
)
SUMMARY_METRICS: tuple[str, ...] = SEGMENT_SUMMARY_METRICS + tuple(
    metric.replace("segment ", "event ", 1) for metric in SEGMENT_SUMMARY_METRICS
)
PERF_SCORE_METRIC = "perf_score (validation)"
CACHE_FILE = Path(".wandb_query_cache.json")
CACHE_TTL_SEC = 12 * 60 * 60
AGENT_SELF_TEST_FILE = Path(".wandb_query_agent_self_test.json")
TERMINAL_BAR_WIDTH = 24
AGENT_SELF_TEST_VERSION = 2
SELF_TEST_RUN_SUFFIX = "4045"
SELF_TEST_MODE = "best"
SELF_TEST_EXPECTED_SELECTION = {
    "metric": PERF_SCORE_METRIC,
    "value": 0.8196026886095766,
    "epoch": 8,
    "step": 76691,
    "source": "history_best",
}
SELF_TEST_EXPECTED: tuple[dict[str, Any], ...] = (
    {
        "metric": "segment PPK-AUC @ TopK=500 (validation)",
        "value": 0.853988983827029,
        "epoch": 8,
        "step": 76689,
    },
    {
        "metric": "segment mTPR in [0,0.001] FPR (validation)",
        "value": 0.3661830723285675,
        "epoch": 8,
        "step": 76675,
    },
    {
        "metric": "segment PPK-AUC @ TopK=250 (validation)",
        "value": 0.9693943008361571,
        "epoch": 8,
        "step": 76687,
    },
    {
        "metric": "event PPK-AUC @ TopK=500 (validation)",
        "value": 0.493184733725263,
        "epoch": 8,
        "step": 76688,
    },
    {
        "metric": "event mTPR in [0,0.001] FPR (validation)",
        "value": 0.4217942953109741,
        "epoch": 8,
        "step": 76674,
    },
    {
        "metric": "event PPK-AUC @ TopK=250 (validation)",
        "value": 0.6960279472381714,
        "epoch": 8,
        "step": 76686,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast W&B run lookup by numeric suffix in run name."
    )
    parser.add_argument(
        "--entity",
        default=WANDB_ENTITY_NAME,
        help=f"W&B entity (default: {WANDB_ENTITY_NAME})",
    )
    parser.add_argument(
        "--project",
        default=WANDB_PROJECT_NAME,
        help=f"W&B project (default: {WANDB_PROJECT_NAME})",
    )
    parser.add_argument(
        "--run-suffix",
        required=True,
        help="Numeric suffix after the final '-' in run name (e.g. 4072).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    config_parser = subparsers.add_parser("config", help="Print run config as JSON.")
    config_parser.add_argument(
        "--include-private",
        action="store_true",
        help="Include private config keys that start with '_'.",
    )

    metric_parser = subparsers.add_parser(
        "metric", help="Print a metric value from summary or history."
    )
    metric_parser.add_argument(
        "--name",
        required=True,
        help="Metric name, e.g. eval/auroc",
    )
    metric_parser.add_argument(
        "--history",
        action="store_true",
        help="Read the latest value from history instead of summary.",
    )
    metric_parser.add_argument(
        "--best",
        action="store_true",
        help="Read the maximum value from history and return its epoch/step.",
    )

    summary_parser = subparsers.add_parser(
        "metric-summary",
        help="Return the fixed 6-metric validation summary (3 segment + 3 event).",
    )
    summary_parser.add_argument(
        "--mode",
        choices=("best", "latest", "summary"),
        default="best",
        help="How to choose the perf_score reference epoch: best/latest; summary is kept as an alias of best.",
    )

    compare_parser = subparsers.add_parser(
        "metric-summary-compare",
        help="Compare the fixed 6-metric validation summary between two or more runs.",
    )
    compare_parser.add_argument(
        "--other-run-suffix",
        required=True,
        nargs="+",
        help="One or more additional numeric run suffixes to compare against.",
    )
    compare_parser.add_argument(
        "--mode",
        choices=("best", "latest", "summary"),
        default="best",
        help="How to choose the perf_score reference epoch: best/latest; summary is kept as an alias of best.",
    )
    compare_parser.add_argument(
        "--format",
        choices=("terminal", "json"),
        default="terminal",
        help="Output format for comparisons (default: terminal).",
    )

    return parser.parse_args()


def load_cache() -> dict[str, Any]:
    if not CACHE_FILE.exists():
        return {}
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_cache(cache: dict[str, Any]) -> None:
    try:
        CACHE_FILE.write_text(json.dumps(cache, sort_keys=True), encoding="utf-8")
    except OSError:
        return


def load_agent_self_test_state() -> dict[str, Any]:
    if not AGENT_SELF_TEST_FILE.exists():
        return {}
    try:
        return json.loads(AGENT_SELF_TEST_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_agent_self_test_state(state: dict[str, Any]) -> None:
    try:
        AGENT_SELF_TEST_FILE.write_text(
            json.dumps(state, sort_keys=True), encoding="utf-8"
        )
    except OSError:
        return


def get_agent_identity() -> str:
    # Prefer Codex thread id when present so the test runs once per agent/thread.
    return (
        os.getenv("CODEX_THREAD_ID")
        or os.getenv("AGENT_ID")
        or os.getenv("HOSTNAME")
        or socket.gethostname()
        or "unknown-agent"
    )


def get_run_by_suffix(
    api: wandb.Api, *, entity: str, project: str, run_suffix: str
) -> Any:
    path = f"{entity}/{project}"
    suffix = str(run_suffix)
    regex = f"-{re.escape(suffix)}$"
    cache_key = f"{entity}/{project}:{suffix}"

    cache = load_cache()
    suffix_cache = cache.get("suffix_to_run", {})
    cached = suffix_cache.get(cache_key)
    if isinstance(cached, dict):
        cached_at = float(cached.get("cached_at", 0))
        run_path = str(cached.get("run_path", ""))
        if run_path and (time.time() - cached_at) < CACHE_TTL_SEC:
            try:
                return api.run(run_path)
            except Exception:
                pass

    # Fast path: server-side display_name regex filter.
    runs = list(
        api.runs(
            path=path,
            filters={"display_name": {"$regex": regex}},
            order="-created_at",
            per_page=50,
        )
    )

    if not runs:
        raise LookupError(f"no run found in {path} with display name suffix '-{suffix}'")

    # Keep exact suffix matches only, then take newest.
    exact = [run for run in runs if str(getattr(run, "name", "")).endswith(f"-{suffix}")]
    candidates = exact or runs
    candidates.sort(key=lambda run: str(getattr(run, "created_at", "")), reverse=True)
    run = candidates[0]

    suffix_cache[cache_key] = {
        "run_path": f"{run.entity}/{run.project}/{run.id}",
        "run_name": run.name,
        "cached_at": time.time(),
    }
    cache["suffix_to_run"] = suffix_cache
    save_cache(cache)
    return run


def get_latest_history_metric(run: Any, metric_name: str) -> Any:
    latest: Any = None
    seen = False
    for row in run.scan_history(keys=[metric_name]):
        if metric_name in row:
            latest = row[metric_name]
            seen = True
    if not seen:
        raise KeyError(f"metric '{metric_name}' not found in run history")
    return latest


def normalize_epoch(value: Any) -> int | None:
    if value is None:
        return None
    try:
        float_value = float(value)
    except (TypeError, ValueError):
        return None
    if not float_value.is_integer():
        return None
    return int(float_value)


def get_latest_history_metric_with_metadata(run: Any, metric_name: str) -> dict[str, Any]:
    latest: dict[str, Any] | None = None
    for row in run.scan_history(keys=[metric_name, "epoch", "_step"]):
        if metric_name not in row:
            continue
        value = row.get(metric_name)
        if value is None:
            continue
        latest = {
            "value": value,
            "epoch": row.get("epoch"),
            "step": row.get("_step"),
        }
    if latest is None:
        raise KeyError(f"metric '{metric_name}' not found in run history")
    return latest


def get_best_history_metric(run: Any, metric_name: str) -> dict[str, Any]:
    best: dict[str, Any] | None = None

    # Fast path: sampled history fetch is typically much faster than full scan_history.
    rows = run.history(keys=[metric_name, "epoch", "_step"], pandas=False, samples=5000)
    if rows is None:
        rows = []

    for row in rows:
        if metric_name not in row:
            continue
        value = row[metric_name]
        if value is None:
            continue
        try:
            float_value = float(value)
        except (TypeError, ValueError):
            continue
        rec = {
            "value": value,
            "epoch": row.get("epoch"),
            "step": row.get("_step"),
            "_float_value": float_value,
        }
        if best is None or float_value > best["_float_value"]:
            best = rec

    if best is None:
        # Fallback when sampled history misses the key.
        for row in run.scan_history(keys=[metric_name, "epoch", "_step"]):
            if metric_name not in row:
                continue
            value = row[metric_name]
            if value is None:
                continue
            try:
                float_value = float(value)
            except (TypeError, ValueError):
                continue
            rec = {
                "value": value,
                "epoch": row.get("epoch"),
                "step": row.get("_step"),
                "_float_value": float_value,
            }
            if best is None or float_value > best["_float_value"]:
                best = rec

    if best is None:
        raise KeyError(f"metric '{metric_name}' not found in run history")
    best.pop("_float_value", None)
    return best


def get_best_history_metric_fast(run: Any, metric_name: str) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    rows = run.history(keys=[metric_name, "epoch", "_step"], pandas=False, samples=2000)
    if rows is None:
        raise KeyError(f"metric '{metric_name}' not found in sampled run history")

    for row in rows:
        if metric_name not in row:
            continue
        value = row.get(metric_name)
        if value is None:
            continue
        try:
            float_value = float(value)
        except (TypeError, ValueError):
            continue
        rec = {
            "value": value,
            "epoch": row.get("epoch"),
            "step": row.get("_step"),
            "_float_value": float_value,
        }
        if best is None or float_value > best["_float_value"]:
            best = rec

    if best is None:
        raise KeyError(f"metric '{metric_name}' not found in sampled run history")
    best.pop("_float_value", None)
    return best


def get_latest_history_metric_fast(run: Any, metric_name: str) -> dict[str, Any]:
    latest: dict[str, Any] | None = None
    rows = run.history(keys=[metric_name, "epoch", "_step"], pandas=False, samples=2000)
    if rows is None:
        raise KeyError(f"metric '{metric_name}' not found in sampled run history")

    for row in rows:
        if metric_name not in row:
            continue
        value = row.get(metric_name)
        if value is None:
            continue
        latest = {
            "value": value,
            "epoch": row.get("epoch"),
            "step": row.get("_step"),
        }

    if latest is None:
        raise KeyError(f"metric '{metric_name}' not found in sampled run history")
    return latest


def get_metric_for_epoch(run: Any, metric_name: str, target_epoch: int) -> dict[str, Any]:
    latest: dict[str, Any] | None = None
    for row in run.scan_history(keys=[metric_name, "epoch", "_step"]):
        if normalize_epoch(row.get("epoch")) != target_epoch:
            continue
        if metric_name not in row:
            continue
        value = row.get(metric_name)
        if value is None:
            continue
        latest = {
            "value": value,
            "epoch": target_epoch,
            "step": row.get("_step"),
        }
    if latest is None:
        raise KeyError(f"metric '{metric_name}' not found in run history for epoch {target_epoch}")
    return latest


def get_metric_for_epoch_fast(run: Any, metric_name: str, target_epoch: int) -> dict[str, Any]:
    latest: dict[str, Any] | None = None
    rows = run.history(keys=[metric_name, "epoch", "_step"], pandas=False, samples=2000)
    if rows is None:
        rows = []

    for row in rows:
        if normalize_epoch(row.get("epoch")) != target_epoch:
            continue
        if metric_name not in row:
            continue
        value = row.get(metric_name)
        if value is None:
            continue
        latest = {
            "value": value,
            "epoch": target_epoch,
            "step": row.get("_step"),
        }

    if latest is not None:
        return latest
    return get_metric_for_epoch(run, metric_name, target_epoch)


def get_perf_score_reference(run: Any, *, mode: str) -> dict[str, Any]:
    selector_mode = "latest" if mode == "latest" else "best"
    try:
        rec = (
            get_latest_history_metric_fast(run, PERF_SCORE_METRIC)
            if selector_mode == "latest"
            else get_best_history_metric_fast(run, PERF_SCORE_METRIC)
        )
    except KeyError:
        rec = (
            get_latest_history_metric_with_metadata(run, PERF_SCORE_METRIC)
            if selector_mode == "latest"
            else get_best_history_metric(run, PERF_SCORE_METRIC)
        )

    epoch = normalize_epoch(rec.get("epoch"))
    if epoch is None:
        raise KeyError(
            f"metric '{PERF_SCORE_METRIC}' is missing a usable integer epoch in run '{run.name}'"
        )

    return {
        "metric": PERF_SCORE_METRIC,
        "value": rec.get("value"),
        "epoch": epoch,
        "step": rec.get("step"),
        "source": f"history_{selector_mode}",
    }


def build_metric_summary_payload(run: Any, *, mode: str) -> dict[str, Any]:
    metric_names = list(SUMMARY_METRICS)
    reference = get_perf_score_reference(run, mode=mode)
    selected_epoch = int(reference["epoch"])
    payload = {
        "run_path": f"{run.entity}/{run.project}/{run.id}",
        "run_name": run.name,
        "mode": mode,
        "selection_metric": reference["metric"],
        "selection_epoch": selected_epoch,
        "selection_step": reference["step"],
        "selection_value": reference["value"],
        "selection_source": reference["source"],
        "metrics": [],
    }

    history_values: dict[str, dict[str, Any] | None] = {metric: None for metric in metric_names}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(metric_names)) as pool:
        future_map = {
            pool.submit(get_metric_for_epoch_fast, run, metric, selected_epoch): metric
            for metric in metric_names
        }
        for future in concurrent.futures.as_completed(future_map):
            metric = future_map[future]
            try:
                history_values[metric] = future.result()
            except KeyError:
                history_values[metric] = None

    for metric in metric_names:
        rec = history_values.get(metric)
        if rec is None:
            payload["metrics"].append(
                {
                    "metric": metric,
                    "value": None,
                    "epoch": selected_epoch,
                    "step": None,
                    "source": "missing_at_selected_epoch",
                }
            )
            continue
        payload["metrics"].append(
            {
                "metric": metric,
                "value": rec.get("value"),
                "epoch": selected_epoch,
                "step": rec.get("step"),
                "source": "history_selected_epoch",
            }
        )

    return payload


def build_metric_summary_comparison_payload(
    run_payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    if not run_payloads:
        raise ValueError("run_payloads must not be empty")

    baseline = run_payloads[0]
    baseline_suffix = str(baseline["run_suffix"])
    metrics_per_run = {
        str(run_payload["run_suffix"]): {
            rec["metric"]: rec for rec in run_payload.get("metrics", []) if "metric" in rec
        }
        for run_payload in run_payloads
    }

    comparisons: list[dict[str, Any]] = []
    wins_by_run_suffix = {str(run_payload["run_suffix"]): 0 for run_payload in run_payloads}
    ties = 0
    missing = 0

    for metric in SUMMARY_METRICS:
        metric_runs: list[dict[str, Any]] = []
        baseline_rec = metrics_per_run[baseline_suffix].get(metric, {})
        baseline_value = baseline_rec.get("value")
        delta_vs_baseline: dict[str, float | None] = {}
        better_vs_baseline: dict[str, str | None] = {}
        numeric_values: list[tuple[str, float]] = []
        has_missing = False

        for run_payload in run_payloads:
            run_suffix = str(run_payload["run_suffix"])
            rec = metrics_per_run[run_suffix].get(metric, {})
            value = rec.get("value")
            metric_runs.append(
                {
                    "run_suffix": run_suffix,
                    "run_name": run_payload.get("run_name"),
                    "value": value,
                    "epoch": rec.get("epoch"),
                    "step": rec.get("step"),
                    "source": rec.get("source"),
                }
            )

            if value is None:
                has_missing = True
                continue
            numeric_values.append((run_suffix, float(value)))

        if has_missing:
            missing += 1

        for metric_run in metric_runs[1:]:
            run_suffix = str(metric_run["run_suffix"])
            value = metric_run.get("value")
            if baseline_value is None or value is None:
                delta_vs_baseline[run_suffix] = None
                better_vs_baseline[run_suffix] = None
                continue

            delta = float(value) - float(baseline_value)
            delta_vs_baseline[run_suffix] = delta
            if delta > 0:
                better_vs_baseline[run_suffix] = run_suffix
            elif delta < 0:
                better_vs_baseline[run_suffix] = baseline_suffix
            else:
                better_vs_baseline[run_suffix] = "tie"

        if len(numeric_values) == len(run_payloads):
            max_value = max(value for _, value in numeric_values)
            best_suffixes = [suffix for suffix, value in numeric_values if value == max_value]
            if len(best_suffixes) == 1:
                wins_by_run_suffix[best_suffixes[0]] += 1
            else:
                ties += 1
        else:
            best_suffixes = []

        comparisons.append(
            {
                "metric": metric,
                "runs": metric_runs,
                "best_run_suffixes": best_suffixes,
                "delta_vs_baseline": delta_vs_baseline,
                "better_vs_baseline": better_vs_baseline,
            }
        )

    return {
        "mode": baseline.get("mode"),
        "baseline_run_suffix": baseline_suffix,
        "runs": run_payloads,
        "comparisons": comparisons,
        "summary": {
            "metrics_compared": len(comparisons),
            "run_count": len(run_payloads),
            "wins_by_run_suffix": wins_by_run_suffix,
            "ties": ties,
            "missing": missing,
        },
    }


def _format_terminal_metric_value(value: Any) -> str:
    if value is None:
        return "    n/a"
    return f"{float(value):>7.3f}"


def _render_terminal_bar(value: Any, *, width: int = TERMINAL_BAR_WIDTH) -> str:
    if value is None:
        return "?" * width
    bounded = max(0.0, min(1.0, float(value)))
    filled = round(bounded * width)
    return ("#" * filled) + ("." * (width - filled))


def _format_terminal_delta(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.3f}"


def render_metric_summary_comparison_terminal(payload: dict[str, Any]) -> str:
    runs = payload.get("runs", [])
    baseline_run_suffix = str(payload["baseline_run_suffix"])
    mode = payload.get("mode")
    suffix_width = max(len(str(run["run_suffix"])) for run in runs)

    lines = [f"Runs compared (mode={mode}, baseline={baseline_run_suffix}):"]
    for run in runs:
        lines.append(f"  {run['run_suffix']}: {run['run_name']}")
        lines.append(
            "        "
            f"{run['selection_metric']}: "
            f"epoch={run['selection_epoch']} "
            f"step={run['selection_step']} "
            f"value={float(run['selection_value']):.3f}"
        )
    lines.append("")

    for comparison in payload.get("comparisons", []):
        lines.append(str(comparison["metric"]))
        delta_vs_baseline = comparison.get("delta_vs_baseline", {})
        better_vs_baseline = comparison.get("better_vs_baseline", {})
        for metric_run in comparison.get("runs", []):
            run_suffix = str(metric_run["run_suffix"])
            line = (
                f" {run_suffix:>{suffix_width}} "
                f"{_format_terminal_metric_value(metric_run.get('value'))} "
                f"|{_render_terminal_bar(metric_run.get('value'))}|"
            )
            if run_suffix != baseline_run_suffix:
                delta = delta_vs_baseline.get(run_suffix)
                better = better_vs_baseline.get(run_suffix)
                better_text = better if better is not None else "n/a"
                line += (
                    f"  delta={_format_terminal_delta(delta)}"
                    f"  better={better_text}"
                )
            lines.append(line)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def verify_metric_summary_payload(payload: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    for key, expected in SELF_TEST_EXPECTED_SELECTION.items():
        got = payload.get(f"selection_{key}" if key != "metric" else "selection_metric")
        if key == "value":
            if got is None or not math.isclose(
                float(got), float(expected), rel_tol=0.0, abs_tol=1e-12
            ):
                problems.append(
                    f"selection {key} mismatch: expected {expected!r}, got {got!r}"
                )
        elif got != expected:
            problems.append(
                f"selection {key} mismatch: expected {expected!r}, got {got!r}"
            )

    got_metrics = payload.get("metrics")
    if not isinstance(got_metrics, list):
        return ["payload.metrics is missing or not a list"]
    if len(got_metrics) != len(SELF_TEST_EXPECTED):
        problems.append(
            f"expected {len(SELF_TEST_EXPECTED)} metrics, got {len(got_metrics)}"
        )

    for idx, expected in enumerate(SELF_TEST_EXPECTED):
        if idx >= len(got_metrics):
            break
        got = got_metrics[idx]
        for key in ("metric", "epoch", "step"):
            if got.get(key) != expected[key]:
                problems.append(
                    f"index {idx} {key} mismatch: expected {expected[key]!r}, got {got.get(key)!r}"
                )
        got_value = got.get("value")
        if got_value is None or not math.isclose(
            float(got_value), float(expected["value"]), rel_tol=0.0, abs_tol=1e-12
        ):
            problems.append(
                f"index {idx} value mismatch: expected {expected['value']!r}, got {got_value!r}"
            )
        if got.get("source") != "history_selected_epoch":
            problems.append(
                "index "
                f"{idx} source mismatch: expected 'history_selected_epoch', got {got.get('source')!r}"
            )
    return problems


def run_agent_self_test_once(api: wandb.Api, *, entity: str, project: str) -> str:
    agent_id = get_agent_identity()
    state = load_agent_self_test_state()
    key = f"{agent_id}:{entity}/{project}"
    cached = state.get(key, {})
    if (
        cached.get("status") == "passed"
        and cached.get("version") == AGENT_SELF_TEST_VERSION
    ):
        return "already_passed"

    run = get_run_by_suffix(
        api,
        entity=entity,
        project=project,
        run_suffix=SELF_TEST_RUN_SUFFIX,
    )
    payload = build_metric_summary_payload(run, mode=SELF_TEST_MODE)
    problems = verify_metric_summary_payload(payload)
    if problems:
        raise RuntimeError(
            "agent self-test failed for run suffix "
            f"{SELF_TEST_RUN_SUFFIX}: " + "; ".join(problems)
        )

    state[key] = {
        "status": "passed",
        "version": AGENT_SELF_TEST_VERSION,
        "checked_at": time.time(),
        "run_path": payload["run_path"],
        "mode": SELF_TEST_MODE,
    }
    save_agent_self_test_state(state)
    return "passed_now"


def main() -> int:
    args = parse_args()
    api = wandb.Api(timeout=45, api_key=os.getenv("WANDB_API_KEY"))

    try:
        self_test_status = run_agent_self_test_once(api, entity=args.entity, project=args.project)
    except (LookupError, RuntimeError, KeyError) as err:
        print(
            "W&B agent self-test status: FAILED. Agent is not ready for metric retrieval. "
            f"Details: {err}",
            file=sys.stderr,
        )
        return 4
    if self_test_status == "passed_now":
        print(
            "W&B agent self-test status: PASSED (verified now). "
            "Agent is ready for further metric retrieval.",
            file=sys.stderr,
        )
    else:
        print(
            "W&B agent self-test status: PASSED (already verified). "
            "Agent is ready for further metric retrieval.",
            file=sys.stderr,
        )

    if args.command == "config":
        try:
            run = get_run_by_suffix(
                api,
                entity=args.entity,
                project=args.project,
                run_suffix=args.run_suffix,
            )
        except LookupError as err:
            print(str(err), file=sys.stderr)
            return 2
        config = dict(getattr(run, "config", {}) or {})
        if not args.include_private:
            config = {key: value for key, value in config.items() if not str(key).startswith("_")}
        print(json.dumps(config, indent=2, sort_keys=True, default=str))
        return 0

    if args.command == "metric":
        try:
            run = get_run_by_suffix(
                api,
                entity=args.entity,
                project=args.project,
                run_suffix=args.run_suffix,
            )
        except LookupError as err:
            print(str(err), file=sys.stderr)
            return 2
        metric_name = args.name
        if args.best:
            try:
                best = get_best_history_metric(run, metric_name)
                payload = {
                    "run_path": f"{run.entity}/{run.project}/{run.id}",
                    "run_name": run.name,
                    "metric": metric_name,
                    "value": best["value"],
                    "epoch": best["epoch"],
                    "step": best["step"],
                    "source": "history_best",
                }
                print(json.dumps(payload, indent=2, sort_keys=True, default=str))
                return 0
            except KeyError as err:
                print(str(err), file=sys.stderr)
                return 3

        if args.history:
            try:
                value = get_latest_history_metric(run, metric_name)
                payload = {
                    "run_path": f"{run.entity}/{run.project}/{run.id}",
                    "run_name": run.name,
                    "metric": metric_name,
                    "value": value,
                    "source": "history_latest",
                }
                print(json.dumps(payload, indent=2, sort_keys=True, default=str))
                return 0
            except KeyError as err:
                print(str(err), file=sys.stderr)
                return 3

        summary = dict(getattr(run, "summary", {}) or {})
        if metric_name in summary:
            payload = {
                "run_path": f"{run.entity}/{run.project}/{run.id}",
                "run_name": run.name,
                "metric": metric_name,
                "value": summary[metric_name],
                "source": "summary",
            }
            print(json.dumps(payload, indent=2, sort_keys=True, default=str))
            return 0

        try:
            value = get_latest_history_metric(run, metric_name)
            payload = {
                "run_path": f"{run.entity}/{run.project}/{run.id}",
                "run_name": run.name,
                "metric": metric_name,
                "value": value,
                "source": "history_latest_fallback",
            }
            print(json.dumps(payload, indent=2, sort_keys=True, default=str))
            return 0
        except KeyError:
            print(
                f"metric '{metric_name}' not found in summary or history for run '{run.name}'",
                file=sys.stderr,
            )
            return 3

    if args.command == "metric-summary":
        try:
            run = get_run_by_suffix(
                api,
                entity=args.entity,
                project=args.project,
                run_suffix=args.run_suffix,
            )
        except LookupError as err:
            print(str(err), file=sys.stderr)
            return 2
        payload = build_metric_summary_payload(run, mode=args.mode)
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0

    if args.command == "metric-summary-compare":
        run_suffixes = [str(args.run_suffix), *[str(value) for value in args.other_run_suffix]]
        if len(set(run_suffixes)) != len(run_suffixes):
            print("duplicate run suffixes are not allowed in metric-summary-compare", file=sys.stderr)
            return 1

        try:
            runs_by_suffix: dict[str, Any] = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(run_suffixes)) as pool:
                future_map = {
                    pool.submit(
                        get_run_by_suffix,
                        api,
                        entity=args.entity,
                        project=args.project,
                        run_suffix=run_suffix,
                    ): run_suffix
                    for run_suffix in run_suffixes
                }
                for future in concurrent.futures.as_completed(future_map):
                    run_suffix = future_map[future]
                    runs_by_suffix[run_suffix] = future.result()
        except LookupError as err:
            print(str(err), file=sys.stderr)
            return 2

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(run_suffixes)) as pool:
            future_map = {
                pool.submit(build_metric_summary_payload, runs_by_suffix[run_suffix], mode=args.mode): run_suffix
                for run_suffix in run_suffixes
            }
            run_payloads: dict[str, dict[str, Any]] = {}
            for future in concurrent.futures.as_completed(future_map):
                run_suffix = future_map[future]
                run_payloads[run_suffix] = {"run_suffix": run_suffix, **future.result()}

        payload = build_metric_summary_comparison_payload(
            [run_payloads[run_suffix] for run_suffix in run_suffixes]
        )
        if args.format == "terminal":
            print(render_metric_summary_comparison_terminal(payload), end="")
        else:
            print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0

    print(f"unknown command '{args.command}'", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
