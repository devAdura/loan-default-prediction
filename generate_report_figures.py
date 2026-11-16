"""Generate report figures from the Nigerian OneFi model and saved metrics."""
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve, roc_auc_score
from explanations import explain
from train_nigeria_model import score_predictions
import hashlib

from nigeria_features import DISPLAY_NAMES, build_training_frame

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data" / "onefi"
ARTIFACT_DIR = ROOT / "artifacts"
FIGURE_DIR = ARTIFACT_DIR / "report_figures"
MODEL_PATH = ARTIFACT_DIR / "credit_risk_nigeria_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "nigeria_metrics.json"

COLORS = {
    "ink": "#18313F",
    "blue": "#2C7FB8",
    "teal": "#32A6A0",
    "gold": "#E7A83E",
    "coral": "#D96C5F",
    "mist": "#EAF2F5",
    "green": "#39856B",
    "red": "#C65A55",
}


def save_figure(fig, filename):
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Created {path.relative_to(ROOT)}")


def add_box(ax, xy, width, height, title, detail, color):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.5, edgecolor=color, facecolor="white",
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height * 0.68, title, ha="center", va="center",
            fontsize=10, weight="bold", color=COLORS["ink"])
    ax.text(x + width / 2, y + height * 0.34, detail, ha="center", va="center",
            fontsize=8, color=COLORS["ink"], linespacing=1.35)


def add_arrow(ax, start, end, color=None, connectionstyle="arc3"):
    arrow = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=13,
        linewidth=1.5, color=color or COLORS["ink"],
        connectionstyle=connectionstyle,
    )
    ax.add_patch(arrow)


def draw_architecture(metrics):
    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("System architecture of the Nigerian credit-risk prototype",
                 loc="left", fontsize=16, weight="bold", color=COLORS["ink"], pad=20)

    ax.text(0.02, 0.83, "MODEL DEVELOPMENT", fontsize=9, weight="bold", color=COLORS["blue"])
    ax.text(0.02, 0.38, "APPLICATION AND INFERENCE", fontsize=9, weight="bold", color=COLORS["teal"])

    add_box(ax, (0.02, 0.57), 0.17, 0.20, "OneFi source files",
            "Performance\nDemographics\nPrevious loans", COLORS["blue"])
    add_box(ax, (0.25, 0.57), 0.19, 0.20, "Feature engineering",
            "Temporal loan filtering\n18 predictors\nBad outcome label", COLORS["blue"])
    add_box(ax, (0.50, 0.57), 0.20, 0.20, "Model comparison",
            "Customer-grouped split\nLogistic Regression\nRandom Forest / Gradient Boosting", COLORS["blue"])
    add_box(ax, (0.76, 0.57), 0.21, 0.20, "Selected artifacts",
            f"{metrics['deployed_model']} pipeline\nThreshold {metrics['decision_threshold']:.2f}\nMetrics JSON", COLORS["blue"])

    add_arrow(ax, (0.19, 0.67), (0.25, 0.67))
    add_arrow(ax, (0.44, 0.67), (0.50, 0.67))
    add_arrow(ax, (0.70, 0.67), (0.76, 0.67))

    add_box(ax, (0.02, 0.12), 0.17, 0.20, "Officer inputs",
            "Applicant and loan\nPrevious-loan summary", COLORS["teal"])
    add_box(ax, (0.25, 0.12), 0.19, 0.20, "Prediction frame",
            "Feature schema alignment\nFitted preprocessing", COLORS["teal"])
    add_box(ax, (0.50, 0.12), 0.20, 0.20, "Model and explanation",
            "Bad-class probability\nThreshold classification\nLocal SHAP", COLORS["teal"])
    add_box(ax, (0.76, 0.12), 0.21, 0.20, "Streamlit results",
            "Risk estimate and factors\nWarnings\nSQLite assessment history", COLORS["teal"])

    add_arrow(ax, (0.19, 0.22), (0.25, 0.22))
    add_arrow(ax, (0.44, 0.22), (0.50, 0.22))
    add_arrow(ax, (0.70, 0.22), (0.76, 0.22))
    add_arrow(ax, (0.86, 0.57), (0.61, 0.32), color=COLORS["gold"], connectionstyle="arc3,rad=-0.1")
    ax.text(0.77, 0.42, "saved model", fontsize=8, color=COLORS["ink"], rotation=-24)

    save_figure(fig, "figure_3_1_system_architecture.png")


def load_project_data():
    demographics = pd.read_csv(DATA_DIR / "traindemographics.csv")
    performance = pd.read_csv(DATA_DIR / "trainperf.csv")
    previous = pd.read_csv(DATA_DIR / "trainprevloans.csv")
    X, y = build_training_frame(demographics, performance, previous)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    model = joblib.load(MODEL_PATH)

    for name, digest in metrics["source_sha256"].items():
        if hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Source data differs from training run")
    X_test = X.iloc[metrics["split_positions"]["test"]]
    y_test = y.iloc[metrics["split_positions"]["test"]]
    probabilities = model.predict_proba(X_test)[:, 1]
    actual = score_predictions(y_test, probabilities, metrics["decision_threshold"])
    if actual != metrics["test_metrics"]:
        raise ValueError("Saved model predictions do not reproduce the test metrics")

    return X, y, X_test, y_test, probabilities, metrics, model


def draw_class_balance(y):
    counts = y.value_counts().reindex([0, 1], fill_value=0)
    labels = ["Not Bad", "Bad"]
    colors = [COLORS["blue"], COLORS["coral"]]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars = ax.bar(labels, counts.values, color=colors, width=0.58)
    ax.set_title("Outcome distribution in labeled training records", loc="left",
                 fontsize=13, weight="bold", color=COLORS["ink"])
    ax.set_ylabel("Applications")
    ax.set_ylim(0, max(counts.values) * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{count:,} ({count / len(y):.1%})", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    save_figure(fig, "figure_4_1_class_distribution.png")


def draw_validation_comparison(metrics):
    rows = metrics["validation_model_comparison"]
    model_names = [f"{row['model']}\nthreshold {row['threshold']:.2f}" for row in rows]
    metric_names = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    display_names = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    palette = [COLORS["blue"], COLORS["teal"], COLORS["gold"], COLORS["coral"], COLORS["green"]]
    values = np.array([[row[name] for name in metric_names] for row in rows])
    x = np.arange(len(rows))
    width = 0.14

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    for index, (name, color) in enumerate(zip(display_names, palette)):
        ax.bar(x + (index - 2) * width, values[:, index], width, label=name, color=color)
    ax.set_title("Validation performance by candidate model", loc="left",
                 fontsize=14, weight="bold", color=COLORS["ink"])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x, model_names)
    ax.legend(ncols=3, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, "figure_4_2_validation_model_comparison.png")


def draw_confusion_matrix(y_test, probabilities, metrics):
    threshold = float(metrics["decision_threshold"])
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(f"{metrics['deployed_model']} test confusion matrix (threshold {threshold:.2f})",
                 loc="left", fontsize=12, weight="bold", color=COLORS["ink"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_xticks([0, 1], ["Not Bad", "Bad"])
    ax.set_yticks([0, 1], ["Not Bad", "Bad"])
    midpoint = matrix.max() / 2
    for row in range(2):
        for column in range(2):
            ax.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center",
                    fontsize=14, weight="bold",
                    color="white" if matrix[row, column] > midpoint else COLORS["ink"])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Applications")
    fig.tight_layout()
    save_figure(fig, "figure_4_3_test_confusion_matrix.png")


def draw_test_curves(y_test, probabilities, metrics):
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    roc_area = auc(false_positive_rate, true_positive_rate)
    pr_area = auc(recall[::-1], precision[::-1])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].plot(false_positive_rate, true_positive_rate, color=COLORS["blue"], linewidth=2,
                 label=f"{metrics['deployed_model']} (AUC = {roc_area:.3f})")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#8C969B", linewidth=1)
    axes[0].set_title("ROC curve", loc="left", weight="bold", color=COLORS["ink"])
    axes[0].set_xlabel("False-positive rate")
    axes[0].set_ylabel("True-positive rate")
    axes[0].legend(frameon=False, loc="lower right")

    axes[1].plot(recall, precision, color=COLORS["coral"], linewidth=2,
                 label=f"{metrics['deployed_model']} (trapezoid PR-AUC = {pr_area:.3f})")
    axes[1].axhline(np.mean(y_test), linestyle="--", color="#8C969B", linewidth=1,
                    label=f"Positive prevalence = {np.mean(y_test):.3f}")
    axes[1].set_title("Precision-recall curve", loc="left", weight="bold", color=COLORS["ink"])
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(frameon=False, loc="upper right")

    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.2)
    fig.suptitle("Internal test-set ranking performance", x=0.06, ha="left",
                 fontsize=14, weight="bold", color=COLORS["ink"])
    fig.tight_layout()
    save_figure(fig, "figure_4_4_test_roc_precision_recall.png")


def draw_local_shap(X_test, y_test, probabilities, metrics, model):
    import json
    grouped, metadata = explain(model, X_test)
    pred = probabilities >= metrics["decision_threshold"]
    y = np.asarray(y_test)
    selections = {"threshold_adjacent": int(np.argmin(abs(probabilities-metrics["decision_threshold"])))}
    for label, mask in {"true_positive": (y==1)&pred, "true_negative": (y==0)&~pred, "false_positive": (y==0)&pred, "false_negative": (y==1)&~pred}.items():
        positions = np.flatnonzero(mask)
        if len(positions):
            selections[label] = int(positions[0])
    evidence = {"metadata": metadata, "model": metrics["deployed_model"], "run_id": metrics["run_id"], "threshold": metrics["decision_threshold"], "comparison": ">=", "top_factors": 6, "normalization": "none", "cases": {}}
    # Private row-level evidence stays outside the public commit via .gitignore.
    for label, pos in selections.items():
        evidence["cases"][label] = {"case_id": f"test-{pos:04d}", "inputs": X_test.iloc[pos].astype(object).where(X_test.iloc[pos].notna(), None).to_dict(), "actual": int(y[pos]), "probability": float(probabilities[pos]), "predicted": int(pred[pos]), "contributions": grouped.iloc[pos].to_dict(), "explained_output": metadata["explained_output"][pos]}
    (ARTIFACT_DIR / "shap_cases.private.json").write_text(json.dumps(evidence, indent=2, allow_nan=False), encoding="utf-8")
    pos = selections["threshold_adjacent"]
    values = grouped.iloc[pos].sort_values(key=abs).tail(6)
    fig, ax = plt.subplots(figsize=(9, 5.3))
    ax.barh([DISPLAY_NAMES[f] for f in values.index], values.values, color=[COLORS["coral"] if v>0 else COLORS["teal"] for v in values])
    ax.axvline(0, color=COLORS["ink"], linewidth=.9)
    ax.set_title(f"Local SHAP explanation for test-{pos:04d}", loc="left")
    ax.set_xlabel(f"SHAP attribution ({metadata['scale']}, no normalization)")
    fig.tight_layout(rect=(0,.08,1,1))
    fig.text(.99,.01,f"Actual: {y[pos]} | Score: {probabilities[pos]:.10f} | Threshold: {metrics['decision_threshold']:.2f}",ha="right",fontsize=8)
    save_figure(fig,"figure_4_5_illustrative_local_shap.png")
    global_values = grouped.abs().mean().sort_values()
    fig, ax = plt.subplots(figsize=(9,7))
    ax.barh([DISPLAY_NAMES[f] for f in global_values.index], global_values.values, color=COLORS["blue"])
    ax.set_title(f"Mean absolute SHAP over all {len(X_test)} held-out applications", loc="left")
    ax.set_xlabel(f"Mean absolute attribution ({metadata['scale']})")
    fig.tight_layout()
    save_figure(fig,"figure_4_6_global_shap.png")
    (ARTIFACT_DIR / "shap_verification.json").write_text(json.dumps({"run_id":metrics["run_id"],"sample_rows":len(X_test), **{k:v for k,v in metadata.items() if k != "explained_output"}, "representative_case_types":list(selections), "global_mean_absolute_shap":grouped.abs().mean().to_dict()}, indent=2),encoding="utf-8")


def main():
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        raise FileNotFoundError("Train the model first to create the saved model and metrics artifacts.")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    X, y, X_test, y_test, probabilities, metrics, model = load_project_data()
    draw_architecture(metrics)
    draw_class_balance(y)
    draw_validation_comparison(metrics)
    draw_confusion_matrix(y_test, probabilities, metrics)
    draw_test_curves(y_test, probabilities, metrics)
    draw_local_shap(X_test, y_test, probabilities, metrics, model)
    print(f"\nAll report figures saved to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()