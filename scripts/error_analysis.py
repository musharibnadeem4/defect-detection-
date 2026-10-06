"""Step 3 (OOF only, never the test set): error inventory, per-fold score analysis, image-property
analysis, thresholds/calibration, exact confidence intervals.

Needs the per-fold models from scripts/cv_save_folds.py. Writes figures to reports/figures/, tables to
reports/, and all numbers to reports/error_analysis_results.json.
Usage: python scripts/error_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import brier_score_loss, precision_recall_curve, roc_curve  # noqa: E402

from defect_detection import error_analysis as ea  # noqa: E402
from defect_detection.data import load_splits, make_loader  # noqa: E402
from defect_detection.metrics import (bootstrap_scalar_ci, proportion_report, ranking_metrics,  # noqa: E402
                                      threshold_metrics)
from defect_detection.model import build_model  # noqa: E402
from defect_detection.utils import ROOT, configure_threads, load_config, resolve, save_json  # noqa: E402

COL = {0: "#4C78A8", 1: "#E45756"}


def sigmoid(z):
    return 1 / (1 + np.exp(-np.asarray(z, float)))


def main() -> None:
    cfg = load_config()
    configure_threads(cfg)
    X = cfg["error_analysis"]
    figs, reports = resolve(cfg, "figures_dir"), resolve(cfg, "reports_dir")
    figs.mkdir(parents=True, exist_ok=True)
    classes = cfg["classes"]
    R: dict = {}

    oof = ea.load_oof(cfg)
    thr = ea.candidate_thresholds(cfg, oof)
    T_VAL = thr["val_tuned"]
    print(f"OOF rows: {len(oof)} (train+val only), positives {int(oof.label.sum())}; thresholds {thr}")
    R["n_oof"], R["thresholds"] = len(oof), thr

    # ---- logits from the exact fold models; consistency with the Step 2 OOF file
    oof["z"] = ea.oof_logits(cfg, oof)
    diff = np.abs(sigmoid(oof["z"]) - oof["prob"])
    R["fold_model_vs_oof_file"] = {"max_abs_prob_diff": float(diff.max()), "mean_abs_prob_diff": float(diff.mean())}
    print("fold models vs oof_predictions.csv: max |dp| = %.2e" % diff.max())
    oof.to_csv(reports / "oof_logits.csv", index=False)

    # ================= 1. error inventory
    errs = ea.error_table(oof, {"0.5": 0.5, "val_tuned": T_VAL}).sort_values(["threshold_name", "error_type", "prob"])
    errs[["filename", "fold", "prob", "label", "error_type", "threshold_name", "threshold"]].assign(
        true_label=lambda d: d["label"].map(dict(enumerate(classes)))).to_csv(reports / "errors.csv", index=False)
    summ = errs.groupby(["threshold_name", "error_type"]).size().unstack(fill_value=0)
    print("\nerrors (FP/FN):\n", summ.to_string())
    print(errs.groupby(["threshold_name", "error_type", "fold"]).size().unstack(fill_value=0).to_string())
    R["error_counts"] = {f"{a}_{b}": int(v) for (a, b), v in errs.groupby(["threshold_name", "error_type"]).size().items()}
    R["errors_by_fold"] = {f"{a}_{b}": {int(k): int(v) for k, v in g.fold.value_counts().items()}
                           for (a, b), g in errs.groupby(["threshold_name", "error_type"])}

    fn = oof[(oof.label == 1) & (oof.prob < T_VAL)].sort_values("prob")
    ea.gallery(cfg, fn, figs / "gallery_false_negatives.png", f"False negatives at the tuned threshold {T_VAL:.3f} "
               "(all; * = also missed at 0.5)", 4,
               lambda r: f"{r['filename']} fold {int(r['fold'])}\np={r['prob']:.4f}" + (" *" if r["prob"] < .5 else ""))
    fp = oof[oof.label == 0].sort_values("prob", ascending=False).head(X["n_confident_fp"])
    ea.gallery(cfg, fp, figs / "gallery_confident_false_positives.png",
               f"{len(fp)} most confident false positives (highest-scored normals)", 4,
               lambda r: f"{r['filename']} fold {int(r['fold'])}\np={r['prob']:.4f}")
    tp = oof[(oof.label == 1) & (oof.prob >= T_VAL)].sort_values("prob").head(X["n_hard_tp"])
    ea.gallery(cfg, tp, figs / "gallery_hardest_true_positives.png",
               f"Hardest true positives (defects just above {T_VAL:.3f})", 4,
               lambda r: f"{r['filename']} fold {int(r['fold'])}\np={r['prob']:.4f}")
    tn = oof[(oof.label == 0) & (oof.prob < T_VAL)].sort_values("prob", ascending=False).head(X["n_hard_tn"])
    ea.gallery(cfg, tn, figs / "gallery_hardest_true_negatives.png",
               f"Hardest true negatives (normals just below {T_VAL:.3f})", 4,
               lambda r: f"{r['filename']} fold {int(r['fold'])}\np={r['prob']:.4f}")

    folds = sorted(oof.fold.unique())
    fig, axes = plt.subplots(2, len(folds), figsize=(2.6 * len(folds), 5.2))
    for row, (name, t) in enumerate((("0.5", 0.5), (f"tuned {T_VAL:.3f}", T_VAL))):
        cb = ea.confusion_by_fold(oof, t)
        for ax, (_, r) in zip(axes[row], cb.iterrows()):
            cm = np.array([[r.tn, r.fp], [r.fn, r.tp]])
            ax.imshow(cm, cmap="Blues", vmax=max(cm.max(), 1))
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
            ax.set_xticks([0, 1], ["normal", "defect"], fontsize=7)
            ax.set_yticks([0, 1], ["normal", "defect"], fontsize=7)
            ax.set_title(f"fold {int(r.fold)} @ {name}", fontsize=8)
    fig.suptitle("Per-fold confusion matrices (rows = true, cols = predicted)")
    fig.tight_layout()
    fig.savefig(figs / "per_fold_confusion.png", dpi=130)
    plt.close(fig)

    zt = float(ea.logit(T_VAL))
    fig, axes = plt.subplots(2, len(folds), figsize=(3 * len(folds), 5.6), sharex="row")
    for j, k in enumerate(folds):
        g = oof[oof.fold == k]
        for c in (0, 1):
            axes[0, j].hist(g.loc[g.label == c, "prob"], bins=np.linspace(0, 1, 26), alpha=.7, color=COL[c],
                            label=classes[c])
            axes[1, j].hist(g.loc[g.label == c, "z"], bins=np.linspace(oof.z.min(), oof.z.max(), 30), alpha=.7,
                            color=COL[c])
        axes[0, j].set_yscale("log")
        axes[0, j].axvline(.5, color="k", ls="--", lw=.8)
        axes[0, j].axvline(T_VAL, color="g", ls="--", lw=.8)
        axes[1, j].axvline(0, color="k", ls="--", lw=.8)
        axes[1, j].axvline(zt, color="g", ls="--", lw=.8)
        axes[0, j].set_title(f"fold {k}", fontsize=9)
        axes[1, j].set_xlabel("logit margin z = l_defect - l_normal")
    axes[0, 0].set_ylabel("count (log) vs probability")
    axes[1, 0].set_ylabel("count vs logit")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("OOF score distributions per fold (dashed black: 0.5, green: tuned)")
    fig.tight_layout()
    fig.savefig(figs / "per_fold_score_distributions.png", dpi=130)
    plt.close(fig)

    # ---- per-fold table + fold 2 investigation
    pool = load_splits(cfg)
    pool = pool[pool.cv_fold >= 0]
    rows = []
    for k in folds:
        g = oof[oof.fold == k]
        zn, zd = g.loc[g.label == 0, "z"], g.loc[g.label == 1, "z"]
        m5, mt = (threshold_metrics(g.label, g.prob, t) for t in (0.5, T_VAL))
        rk = ranking_metrics(g.label, g.prob)
        rows.append({"fold": int(k), "n_normal": len(zn), "n_defect": len(zd), "pr_auc": rk["pr_auc"],
                     "roc_auc": rk["roc_auc"], "z_normal_median": zn.median(), "z_normal_q95": zn.quantile(.95),
                     "z_normal_max": zn.max(), "z_defect_median": zd.median(), "z_defect_min": zd.min(),
                     "gap_z": zd.min() - zn.max(), "ok_thr_lo": float(sigmoid(zn.max())),
                     "ok_thr_hi": float(sigmoid(zd.min())), "fp_0.5": m5["fp"], "fn_0.5": m5["fn"],
                     "fp_tuned": mt["fp"], "fn_tuned": mt["fn"], "brier": brier_score_loss(g.label, g.prob),
                     "train_normals": int(((pool.cv_fold != k) & (pool.label == "normal")).sum()),
                     "train_defects": int(((pool.cv_fold != k) & (pool.label == "defective")).sum())})
    ft = pd.DataFrame(rows)
    ft.to_csv(reports / "per_fold_table.csv", index=False)
    print("\nper-fold table:\n", ft.round(3).T.to_string())
    R["per_fold"] = ft.round(5).to_dict(orient="records")

    # fold 2: is the whole distribution shifted?
    F = 2
    tests = {}
    for cls, nm in ((0, "normals"), (1, "defects")):
        a, b = oof[(oof.fold == F) & (oof.label == cls)].z, oof[(oof.fold != F) & (oof.label == cls)].z
        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        tests[f"fold{F}_vs_rest_{nm}_z"] = {"median_fold": float(a.median()), "median_rest": float(b.median()),
                                            "shift": float(a.median() - b.median()), "mannwhitney_p": float(u.pvalue),
                                            "rank_biserial": float(2 * u.statistic / (len(a) * len(b)) - 1)}
    kw = stats.kruskal(*[oof[(oof.fold == k) & (oof.label == 0)].z for k in folds])
    tests["kruskal_normals_z_across_folds_p"] = float(kw.pvalue)
    kwd = stats.kruskal(*[oof[(oof.fold == k) & (oof.label == 1)].z for k in folds])
    tests["kruskal_defects_z_across_folds_p"] = float(kwd.pvalue)
    # fold-2 vs rest, normals only, by quantile: shift is uniform or tail-only?
    qs = [.1, .25, .5, .75, .9, .95]
    a, b = oof[(oof.fold == F) & (oof.label == 0)].z, oof[(oof.fold != F) & (oof.label == 0)].z
    tests["normals_z_quantiles_fold2"] = {str(q): float(a.quantile(q)) for q in qs}
    tests["normals_z_quantiles_rest"] = {str(q): float(b.quantile(q)) for q in qs}
    # does the fold-2 MODEL itself look shifted on the data it was trained on?
    train_z = {}
    with torch.no_grad():
        for k in folds:
            model, size = ea.load_fold_model(cfg, int(k))
            tr = pool[pool.cv_fold != k]
            ds = ea.eval_dataset(cfg, tr[["filename", "label"]], size)
            out = []
            for x, _ in make_loader(ds, cfg["training"]["batch_size"], False):
                lg = model(x).float()
                out.append((lg[:, 1] - lg[:, 0]).numpy())
            z = np.concatenate(out)
            y = ds.targets
            train_z[int(k)] = {"train_normal_median_z": float(np.median(z[y == 0])),
                               "train_normal_q95_z": float(np.quantile(z[y == 0], .95)),
                               "train_defect_median_z": float(np.median(z[y == 1])),
                               "train_fp_at_0.5": int(((z > 0) & (y == 0)).sum()),
                               "train_fn_at_0.5": int(((z <= 0) & (y == 1)).sum())}
    tests["train_set_scores_per_fold_model"] = train_z
    R["fold2_investigation"] = tests
    print("\nfold-2 investigation:", pd.json_normalize(tests, sep=".").T.round(4).to_string())

    # ================= 3. image-property analysis
    raw = resolve(cfg, "raw_dir")
    feats = pd.DataFrame([{**ea.image_features(raw / classes[int(r.label)] / r.filename),
                           "filename": r.filename} for r in oof.itertuples()])
    D = oof.merge(feats, on="filename")
    D["outcome_0.5"] = np.select([(D.label == 0) & (D.prob < .5), (D.label == 0), (D.label == 1) & (D.prob >= .5)],
                                 ["TN", "FP", "TP"], "FN")
    D["outcome_tuned"] = np.select([(D.label == 0) & (D.prob < T_VAL), (D.label == 0), (D.label == 1) & (D.prob >= T_VAL)],
                                   ["TN", "FP", "TP"], "FN")
    D.to_csv(reports / "image_features.csv", index=False)
    fcols = ["brightness", "contrast", "sharpness", "background", "hole_radius", "hole_offset"]
    res = []
    for oc in ("outcome_0.5", "outcome_tuned"):
        for bad, good in (("FP", "TN"), ("FN", "TP")):
            A, B = D[D[oc] == bad], D[D[oc] == good]
            for f in fcols:
                if len(A) >= 2 and len(B) >= 2:
                    u = stats.mannwhitneyu(A[f], B[f], alternative="two-sided")
                    res.append({"threshold": oc.split("_")[1], "comparison": f"{bad} vs {good}", "feature": f,
                                "n_bad": len(A), "n_good": len(B), "median_bad": A[f].median(),
                                "median_good": B[f].median(), "p": u.pvalue,
                                "rank_biserial": 2 * u.statistic / (len(A) * len(B)) - 1})
    cmp = pd.DataFrame(res)
    cmp["p_bh"] = np.nan
    for key, g in cmp.groupby(["threshold", "comparison"]):   # Benjamini-Hochberg within each comparison
        p = g["p"].to_numpy(); order = np.argsort(p); ranks = np.empty(len(p)); ranks[order] = np.arange(1, len(p) + 1)
        adj = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
        cmp.loc[g.index[order], "p_bh"] = np.minimum(adj, 1)
    cmp.to_csv(reports / "error_vs_correct_features.csv", index=False)
    print("\nerrors vs correct (Mann-Whitney):\n", cmp.round(4).to_string())
    # continuous association of the score with each feature (more power than the error groups)
    sp = []
    for cls, nm in ((0, "normals"), (1, "defects")):
        for sub, mask in (("all folds", D.label == cls), ("excluding fold 2", (D.label == cls) & (D.fold != 2))):
            for f in fcols:
                rho, p = stats.spearmanr(D.loc[mask, f], D.loc[mask, "z"], nan_policy="omit")
                sp.append({"class": nm, "subset": sub, "feature": f, "n": int(mask.sum()), "spearman_rho": rho, "p": p})
    spd = pd.DataFrame(sp)
    spd.to_csv(reports / "score_vs_feature_spearman.csv", index=False)
    print("\nSpearman(z, feature):\n", spd.round(4).to_string())
    # fold-2 normals vs other normals on features
    fo = []
    for f in fcols:
        a, b = D[(D.fold == 2) & (D.label == 0)][f], D[(D.fold != 2) & (D.label == 0)][f]
        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        fo.append({"feature": f, "median_fold2": a.median(), "median_rest": b.median(), "p": u.pvalue})
    fo = pd.DataFrame(fo)
    fo.to_csv(reports / "fold2_normals_features.csv", index=False)
    print("\nfold-2 normals vs other normals (features):\n", fo.round(4).to_string())
    R["feature_tests_n"] = {"FP@0.5": int((D["outcome_0.5"] == "FP").sum()), "FN@0.5": int((D["outcome_0.5"] == "FN").sum()),
                            "FP@tuned": int((D["outcome_tuned"] == "FP").sum()),
                            "FN@tuned": int((D["outcome_tuned"] == "FN").sum())}

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    order = ["TN", "FP", "TP", "FN"]
    for ax, f in zip(axes.ravel(), fcols):
        data = [D.loc[D["outcome_0.5"] == o, f].dropna() for o in order]
        ax.boxplot(data, tick_labels=[f"{o}\n(n={len(d)})" for o, d in zip(order, data)], showfliers=False)
        for i, d in enumerate(data):
            ax.scatter(np.random.default_rng(0).normal(i + 1, .05, len(d)), d, s=6, alpha=.4,
                       color={"TN": COL[0], "FP": "#F58518", "TP": COL[1], "FN": "k"}[order[i]])
        ax.set_title(f, fontsize=9)
    fig.suptitle("Image properties by outcome at threshold 0.5 (OOF)")
    fig.tight_layout()
    fig.savefig(figs / "error_features.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, f in zip(axes.ravel(), fcols):
        n = D[D.label == 0]
        sc = ax.scatter(n[f], n["z"], c=n["fold"], cmap="tab10", s=10, vmin=0, vmax=9)
        ax.axhline(0, color="k", ls="--", lw=.7)
        ax.set_xlabel(f)
        ax.set_ylabel("z (normals)")
    fig.suptitle("Normal images: score vs image property, coloured by fold (fold 2 = green)")
    fig.tight_layout()
    fig.savefig(figs / "score_vs_features_normals.png", dpi=130)
    plt.close(fig)

    # ================= 4. thresholds & calibration (OOF only)
    y, p = oof.label.to_numpy(), oof.prob.to_numpy()
    prec, rec, tpr_thr = precision_recall_curve(y, p)
    fpr, tpr, _ = roc_curve(y, p)
    ts = np.unique(np.r_[0, np.sort(p), 1])
    pr_t = [threshold_metrics(y, p, t) for t in ts]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(rec, prec, color="k", lw=2, label="pooled OOF")
    axes[1].plot(fpr, tpr, color="k", lw=2, label="pooled OOF")
    for k in folds:
        g = oof[oof.fold == k]
        pp, rr, _ = precision_recall_curve(g.label, g.prob)
        ff, tt, _ = roc_curve(g.label, g.prob)
        axes[0].plot(rr, pp, lw=.9, alpha=.7, label=f"fold {k}")
        axes[1].plot(ff, tt, lw=.9, alpha=.7)
    for name, t in thr.items():
        m = threshold_metrics(y, p, t)
        axes[0].scatter(m["recall"], m["precision"], s=50, zorder=5, label=f"{name} ({t:.3f})")
        axes[1].scatter(1 - m["recall_neg"], m["recall"], s=50, zorder=5)
    axes[0].set(xlabel="recall", ylabel="precision", title=f"PR curve (AP={ranking_metrics(y, p)['pr_auc']:.3f})")
    axes[1].set(xlabel="false positive rate", ylabel="recall", title=f"ROC (AUC={ranking_metrics(y, p)['roc_auc']:.3f})")
    axes[0].legend(fontsize=7)
    axes[2].plot(ts, [m["precision"] for m in pr_t], label="precision")
    axes[2].plot(ts, [m["recall"] for m in pr_t], label="recall")
    for name, t in thr.items():
        axes[2].axvline(t, ls="--", lw=.8, color="gray")
        axes[2].text(t, .02, name, rotation=90, fontsize=7, va="bottom")
    axes[2].set(xlabel="threshold", title="precision / recall vs threshold (pooled OOF)")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "pr_roc_threshold_curves.png", dpi=130)
    plt.close(fig)

    # reliability + Brier
    rel, ece = ea.reliability(y, p)
    cal = {"pooled": {"brier": ea.brier(y, p), "ece": ece}}
    fig, axes = plt.subplots(1, len(folds) + 1, figsize=(3.2 * (len(folds) + 1), 3.4), sharey=True)
    for ax, (nm, g) in zip(axes, [("pooled", oof)] + [(f"fold {k}", oof[oof.fold == k]) for k in folds]):
        t, e = ea.reliability(g.label.to_numpy(), g.prob.to_numpy())
        ax.plot([0, 1], [0, 1], "k--", lw=.8)
        ax.plot(t.mean_pred, t.frac_pos, "o-", color="#E45756")
        for _, r in t.iterrows():
            ax.annotate(int(r.n), (r.mean_pred, r.frac_pos), fontsize=6, xytext=(2, 3), textcoords="offset points")
        ax.set_title(f"{nm}: Brier {ea.brier(g.label, g.prob):.3f}, ECE {e:.3f}", fontsize=8)
        ax.set_xlabel("mean predicted p")
        if nm != "pooled":
            cal[nm] = {"brier": ea.brier(g.label, g.prob), "ece": e}
    axes[0].set_ylabel("fraction defective")
    fig.suptitle("Reliability diagrams (numbers = images per bin)", fontsize=9)
    fig.tight_layout()
    fig.savefig(figs / "reliability.png", dpi=130)
    plt.close(fig)
    R["calibration"] = cal
    print("\ncalibration:", {k: {a: round(b, 4) for a, b in v.items()} for k, v in cal.items()})

    # candidate thresholds
    alpha = X["ci_alpha"]
    lofo = ea.lofo_predictions(cfg, oof)
    cand = {}
    for name, t in thr.items():
        m = threshold_metrics(y, p, t)
        cand[name] = {"threshold": t, **proportion_report(m, alpha), "f1": m["f1"]}
    m = threshold_metrics(y, lofo.astype(float), 0.5)
    cand["oof_leave_one_fold_out"] = {"threshold": float("nan"), **proportion_report(m, alpha), "f1": m["f1"]}
    for name, c in cand.items():
        c["cost"] = {str(r): r * c["fn"] + c["fp"] for r in X["cost_ratios"]}
    R["candidate_thresholds"] = cand
    ct = pd.DataFrame({k: {"thr": v["threshold"], "TP": v["tp"], "FP": v["fp"], "FN": v["fn"], "TN": v["tn"],
                           "recall": v["recall"], "recall_95CI": "%.3f-%.3f" % v["recall_ci"],
                           "precision": v["precision"], "precision_95CI": "%.3f-%.3f" % v["precision_ci"],
                           **{f"cost@{r}:1": v["cost"][str(r)] for r in X["cost_ratios"]}} for k, v in cand.items()}).T
    ct.to_csv(reports / "candidate_thresholds.csv")
    print("\ncandidate thresholds (pooled OOF, exact CIs):\n", ct.to_string())
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for r in X["cost_ratios"]:
        c = np.array([r * m_["fn"] + m_["fp"] for m_ in pr_t])
        ax.plot(ts, c / c.max(), label=f"FN:FP cost = {r}:1 (min at t={ts[c.argmin()]:.3f})")
    ax.set(xlabel="threshold", ylabel="relative total cost (OOF)", title="Cost vs threshold (illustrative ratios)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figs / "cost_vs_threshold.png", dpi=130)
    plt.close(fig)
    R["cost_minimising_threshold"] = {str(r): float(ts[np.argmin([r * m_["fn"] + m_["fp"] for m_ in pr_t])]) for r in X["cost_ratios"]}

    # optional: temperature scaling fitted on the validation split only (main model)
    ck = torch.load(resolve(cfg, "artifacts_dir") / "best.pt", map_location="cpu")
    main = build_model(ck["model"], len(classes), pretrained=False)
    main.load_state_dict(ck["state_dict"])
    main.eval()
    sp_ = load_splits(cfg)
    val = sp_[sp_.split == "val"]
    ds = ea.eval_dataset(cfg, val[["filename", "label"]], ck["size"] if "size" in ck else 224)
    zv = []
    with torch.no_grad():
        for x, _ in make_loader(ds, cfg["training"]["batch_size"], False):
            lg = main(x).float()
            zv.append((lg[:, 1] - lg[:, 0]).numpy())
    zv = np.concatenate(zv)
    ts_fit = ea.fit_temperature(zv, ds.targets)
    T = ts_fit["T"]
    ts_res = {"fit_on_validation": ts_fit, "oof": {}}
    for nm, zz in (("raw", oof.z.to_numpy()), ("scaled", oof.z.to_numpy() / T)):
        pp = sigmoid(zz)
        d = {"brier": ea.brier(y, pp), "ece": ea.reliability(y, pp)[1],
             "fp_at_0.5": int(((pp >= .5) & (y == 0)).sum()), "fn_at_0.5": int(((pp < .5) & (y == 1)).sum()),
             "fp_at_val_tuned_p": int(((pp >= T_VAL) & (y == 0)).sum()), "fn_at_val_tuned_p": int(((pp < T_VAL) & (y == 1)).sum()),
             "fold2_fp_at_0.5": int(((pp >= .5) & (y == 0) & (oof.fold == 2)).sum()),
             "fold2_brier": ea.brier(y[oof.fold == 2], pp[(oof.fold == 2).to_numpy()]),
             "pr_auc": ranking_metrics(y, zz)["pr_auc"],     # on logits: monotone, so unaffected by sigmoid saturation
             "n_saturated_probs": int(((pp >= 1 - 1e-12) | (pp <= 1e-12)).sum())}
        ts_res["oof"][nm] = d
    R["temperature_scaling"] = ts_res
    print("\ntemperature scaling (fit on validation only):", ts_fit)
    print(pd.DataFrame(ts_res["oof"]).round(4).to_string())

    # ================= 5. statistics (pooled OOF)
    from sklearn.metrics import average_precision_score, roc_auc_score
    n_b = cfg["evaluation"]["bootstrap_resamples"]
    seed = cfg["project"]["seed"]
    st = {"f1_bootstrap": {}, "pr_auc_bootstrap": bootstrap_scalar_ci(y, p, average_precision_score, n_b, seed),
          "roc_auc_bootstrap": bootstrap_scalar_ci(y, p, roc_auc_score, n_b, seed)}
    from sklearn.metrics import f1_score
    for name, t in thr.items():
        st["f1_bootstrap"][name] = bootstrap_scalar_ci(y, p, lambda a, b, t=t: f1_score(a, b >= t, zero_division=0), n_b, seed)
    R["statistics_oof"] = st
    print("\nbootstrap (F1/AUC) on pooled OOF:", st)

    save_json(R, reports / "error_analysis_results.json")
    print("\nsaved reports/error_analysis_results.json and figures to", figs)


if __name__ == "__main__":
    main()
