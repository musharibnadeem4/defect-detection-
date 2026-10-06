"""Threshold selection (validation-only rule), metric arithmetic, bootstrap and model ranking."""
import numpy as np
import pytest

from defect_detection.metrics import (bootstrap_ci, ranking_metrics, select_threshold, selection_key,
                                      threshold_metrics)


def test_threshold_metrics_known_confusion():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([.9, .8, .2, .7, .1, .1, .1, .1])          # at 0.5: tp=2 fn=1 fp=1 tn=4
    m = threshold_metrics(y, p, 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 4)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)
    assert m["precision_neg"] == pytest.approx(4 / 5) and m["recall_neg"] == pytest.approx(4 / 5)
    assert m["accuracy"] == pytest.approx(6 / 8)


def test_threshold_rule_is_greater_equal():
    assert threshold_metrics(np.array([1, 0]), np.array([0.5, 0.2]), 0.5)["tp"] == 1


def test_selected_threshold_meets_recall_target_with_best_precision():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(40), np.zeros(160)].astype(int)
    p = np.clip(np.r_[rng.normal(.75, .2, 40), rng.normal(.3, .2, 160)], 0, 1)
    for target in (0.8, 0.95, 1.0):
        t = select_threshold(y, p, target)
        m = threshold_metrics(y, p, t)
        assert m["recall"] >= target
        # no other threshold reaching the target has higher precision
        best = max(threshold_metrics(y, p, c)["precision"] for c in np.unique(p)
                   if threshold_metrics(y, p, c)["recall"] >= target)
        assert m["precision"] >= best - 1e-9


def test_perfectly_separable_threshold_sits_inside_the_gap_not_on_the_edge():
    y = np.r_[np.ones(10), np.zeros(30)].astype(int)
    p = np.r_[np.linspace(.90, .99, 10), np.linspace(.01, .20, 30)]
    t = select_threshold(y, p, 0.95)
    assert p[y == 0].max() < t < p[y == 1].min()           # strictly between the classes
    m = threshold_metrics(y, p, t)
    assert m["recall"] == 1.0 and m["precision"] == 1.0


def test_recall_target_forces_catching_the_hard_positive():
    # 19 positives: 18/19 = 0.947 < 0.95, so the single low-scored positive must be caught
    y = np.r_[np.ones(19), np.zeros(81)].astype(int)
    p = np.r_[np.full(18, .95), [.30], np.full(81, .05)]
    p[19:23] = [.40, .35, .1, .1]                           # a few negatives above/near the hard positive
    t = select_threshold(y, p, 0.95)
    assert threshold_metrics(y, p, t)["recall"] == 1.0 and t <= .30


def test_ranking_metrics_perfect_and_degenerate():
    y = np.array([0, 0, 1, 1])
    assert ranking_metrics(y, np.array([.1, .2, .8, .9])) == {"pr_auc": 1.0, "roc_auc": 1.0}
    assert np.isnan(ranking_metrics(np.zeros(4), np.random.rand(4))["pr_auc"])


def test_bootstrap_ci_deterministic_and_brackets_point_estimate():
    rng = np.random.default_rng(1)
    y = (rng.random(300) < .2).astype(int)
    p = np.clip(y * .6 + rng.normal(.2, .15, 300), 0, 1)
    a, b = bootstrap_ci(y, p, .5, 200, seed=3), bootstrap_ci(y, p, .5, 200, seed=3)
    assert a == b
    pt = threshold_metrics(y, p, .5)
    for k in ("precision", "recall", "f1"):
        assert a[k][0] <= pt[k] <= a[k][1] and a[k][0] < a[k][1]


def test_selection_key_prefers_pr_auc_then_recall_then_precision_then_val_loss():
    def run(pr, rec, prec, vloss):
        return {"val": {"pr_auc": pr, "at_0.5": {"recall": rec}, "at_tuned": {"precision": prec}},
                "best_epoch": 1, "history": [{"val_loss": vloss}]}
    assert selection_key(run(.99, 0, 0, 9)) > selection_key(run(.98, 1, 1, 0))
    assert selection_key(run(1, 1, 1, .5)) > selection_key(run(1, .9, 1, 0))
    assert selection_key(run(1, 1, 1, .01)) > selection_key(run(1, 1, 1, .20))   # tie -> lower val loss
    assert selection_key(run(1.0, 1, 1, .1)) == selection_key(run(1.0004, 1, 1, .1))  # <0.001 is noise


def test_clopper_pearson_known_values_and_edges():
    from defect_detection.metrics import clopper_pearson
    lo, hi = clopper_pearson(5, 10)
    assert (lo, hi) == pytest.approx((0.18709, 0.81291), abs=1e-4)           # textbook value
    assert clopper_pearson(19, 19) == pytest.approx((0.82353, 1.0), abs=1e-4)  # perfect score: wide lower bound
    assert clopper_pearson(0, 10)[0] == 0.0 and clopper_pearson(0, 10)[1] == pytest.approx(0.30850, abs=1e-4)
    assert np.isnan(clopper_pearson(0, 0)[0])
    for k in range(0, 21):                                                   # always contains the point estimate
        lo, hi = clopper_pearson(k, 20)
        assert lo <= k / 20 <= hi


def test_bootstrap_scalar_flags_degenerate_perfect_scores():
    from defect_detection.metrics import bootstrap_scalar_ci
    y = np.r_[np.ones(10), np.zeros(40)].astype(int)
    perfect = np.r_[np.full(10, .99), np.full(40, .01)]
    r = bootstrap_scalar_ci(y, perfect, lambda a, b: ranking_metrics(a, b)["pr_auc"], 100, seed=0)
    assert r["degenerate"] and r["ci"] == [1.0, 1.0]
    noisy = np.clip(perfect + np.random.default_rng(0).normal(0, .6, 50), 0, 1)
    r2 = bootstrap_scalar_ci(y, noisy, lambda a, b: ranking_metrics(a, b)["roc_auc"], 100, seed=0)
    assert not r2["degenerate"] and r2["ci"][0] < r2["ci"][1]
