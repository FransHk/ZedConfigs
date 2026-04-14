from wandb_query import (
    SUMMARY_METRICS,
    build_metric_summary_comparison_payload,
    render_metric_summary_comparison_terminal,
)


def test_build_metric_summary_comparison_payload() -> None:
    payload_a = {
        "run_suffix": "3743",
        "run_path": "entity/project/run-a",
        "run_name": "model-a-3743",
        "mode": "best",
        "selection_metric": "perf_score (validation)",
        "selection_epoch": 4,
        "selection_step": 40,
        "selection_value": 0.95,
        "selection_source": "history_best",
        "metrics": [
            {
                "metric": metric,
                "value": float(index),
                "epoch": 4,
                "step": index * 10,
                "source": "history_selected_epoch",
            }
            for index, metric in enumerate(SUMMARY_METRICS, start=1)
        ],
    }
    payload_b = {
        "run_suffix": "4045",
        "run_path": "entity/project/run-b",
        "run_name": "model-b-4045",
        "mode": "best",
        "selection_metric": "perf_score (validation)",
        "selection_epoch": 8,
        "selection_step": 80,
        "selection_value": 0.99,
        "selection_source": "history_best",
        "metrics": [
            {
                "metric": metric,
                "value": None if index == 3 else float(index) + 0.5,
                "epoch": 8,
                "step": index * 100,
                "source": "history_selected_epoch" if index != 3 else "missing_at_selected_epoch",
            }
            for index, metric in enumerate(SUMMARY_METRICS, start=1)
        ],
    }
    payload_c = {
        "run_suffix": "5000",
        "run_path": "entity/project/run-c",
        "run_name": "model-c-5000",
        "mode": "best",
        "selection_metric": "perf_score (validation)",
        "selection_epoch": 9,
        "selection_step": 90,
        "selection_value": 0.75,
        "selection_source": "history_best",
        "metrics": [
            {
                "metric": metric,
                "value": float(index) - 0.25,
                "epoch": 9,
                "step": index * 1000,
                "source": "history_selected_epoch",
            }
            for index, metric in enumerate(SUMMARY_METRICS, start=1)
        ],
    }

    comparison = build_metric_summary_comparison_payload([payload_a, payload_b, payload_c])

    assert comparison["mode"] == "best"
    assert comparison["baseline_run_suffix"] == "3743"
    assert [run["run_suffix"] for run in comparison["runs"]] == ["3743", "4045", "5000"]
    assert comparison["summary"] == {
        "metrics_compared": 6,
        "run_count": 3,
        "wins_by_run_suffix": {"3743": 0, "4045": 5, "5000": 0},
        "ties": 0,
        "missing": 1,
    }

    first = comparison["comparisons"][0]
    assert first["metric"] == SUMMARY_METRICS[0]
    assert first["delta_vs_baseline"] == {"4045": 0.5, "5000": -0.25}
    assert first["better_vs_baseline"] == {"4045": "4045", "5000": "3743"}
    assert first["runs"][0]["step"] == 10
    assert first["runs"][1]["step"] == 100
    assert first["runs"][2]["step"] == 1000

    missing = comparison["comparisons"][2]
    assert missing["metric"] == SUMMARY_METRICS[2]
    assert missing["delta_vs_baseline"]["4045"] is None
    assert missing["better_vs_baseline"]["4045"] is None


def test_render_metric_summary_comparison_terminal() -> None:
    payload = build_metric_summary_comparison_payload(
        [
            {
                "run_suffix": "3743",
                "run_path": "entity/project/run-a",
                "run_name": "oddity-training_1-movinet_a5_base-3708-random-erasing-dawn-3743",
                "mode": "best",
                "selection_metric": "perf_score (validation)",
                "selection_epoch": 6,
                "selection_step": 111,
                "selection_value": 0.91234,
                "selection_source": "history_best",
                "metrics": [
                    {
                        "metric": metric,
                        "value": 1.0 if index == 0 else 0.5,
                        "epoch": 6,
                        "step": 10,
                        "source": "history_selected_epoch",
                    }
                    for index, metric in enumerate(SUMMARY_METRICS)
                ],
            },
            {
                "run_suffix": "4045",
                "run_path": "entity/project/run-b",
                "run_name": "oddity-training_1-ymamba_mobilenetv4_conv_small-4037_new_batch_mix_sampler-4045",
                "mode": "best",
                "selection_metric": "perf_score (validation)",
                "selection_epoch": 8,
                "selection_step": 222,
                "selection_value": 0.81234,
                "selection_source": "history_best",
                "metrics": [
                    {
                        "metric": metric,
                        "value": 0.75 if index == 0 else 0.25,
                        "epoch": 8,
                        "step": 20,
                        "source": "history_selected_epoch",
                    }
                    for index, metric in enumerate(SUMMARY_METRICS)
                ],
            },
        ]
    )

    rendered = render_metric_summary_comparison_terminal(payload)

    assert "Runs compared (mode=best, baseline=3743):" in rendered
    assert "  3743: oddity-training_1-movinet_a5_base-3708-random-erasing-dawn-3743" in rendered
    assert "perf_score (validation): epoch=6 step=111 value=0.912" in rendered
    assert "  4045: oddity-training_1-ymamba_mobilenetv4_conv_small-4037_new_batch_mix_sampler-4045" in rendered
    assert "perf_score (validation): epoch=8 step=222 value=0.812" in rendered
    assert "segment PPK-AUC @ TopK=500 (validation)" in rendered
    assert "event mTPR in [0,0.001] FPR (validation)" in rendered
    assert "delta=-0.250  better=3743" in rendered
