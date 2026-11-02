
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    precision_recall_curve,
)


def select_threshold(y_true, probabilities, target_recall=0.70):
    """
    Select a threshold using validation data only.

    Prefer the threshold with the highest precision among those
    that achieve at least target_recall.
    If that target is not achievable, fall back to the
    threshold with the highest F1 score.
    """
    precisions, recalls, thresholds = precision_recall_curve(
        y_true, probabilities
    )

    # The final precision/recall values have no corresponding threshold.
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    f1_scores = (
        2 * precisions * recalls /
        (precisions + recalls + 1e-10)
    )

    eligible = np.where(recalls >= target_recall)[0]

    if len(eligible) > 0:
        # Among thresholds meeting the recall target,
        # choose the one with the highest precision.
        best_index = eligible[np.argmax(precisions[eligible])]
        reason = f"Highest precision while recall >= {target_recall:.2f}"
    else:
        best_index = np.argmax(f1_scores)
        reason = "Target recall not achieved; selected threshold with highest F1"

    return float(thresholds[best_index]), reason


def calculate_metrics(y_true, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(int)

    return {
        "Threshold": float(threshold),
        "Accuracy": float(accuracy_score(y_true, predictions)),
        "Precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "Recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "F1": float(f1_score(y_true, predictions, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y_true, probabilities)),
    }


def save_evaluation_plots(y_true, probabilities, predictions, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Confusion matrix
    cm = confusion_matrix(y_true, predictions, labels=[0, 1])

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not Churn", "Churn"]
    )

    display.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix - Final Test Set")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300)
    plt.show()
    plt.close()

    # 2. ROC curve
    RocCurveDisplay.from_predictions(y_true, probabilities)
    plt.title("ROC Curve - Final Test Set")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=300)
    plt.show()
    plt.close()

    # 3. Precision-Recall curve
    PrecisionRecallDisplay.from_predictions(y_true, probabilities)
    plt.title("Precision-Recall Curve - Final Test Set")
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_curve.png", dpi=300)
    plt.show()
    plt.close()

    return cm


def save_feature_importance(model, output_dir, top_n=20):
    """
    Works with a fitted sklearn Pipeline containing:
    preprocessor and classifier steps.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    if not hasattr(classifier, "feature_importances_"):
        print("This classifier does not provide feature_importances_.")
        return None

    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    importance_df.to_csv(
        output_dir / "feature_importance.csv",
        index=False
    )

    top_features = importance_df.head(top_n).sort_values(
        "Importance", ascending=True
    )

    plt.figure(figsize=(10, 8))
    plt.barh(top_features["Feature"], top_features["Importance"])
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Feature Importances - Gradient Boosting")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=300)

    return importance_df


def run_final_evaluation(
    tuned_pipeline,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    output_dir,
    target_recall=0.70,
    model_file=None
):
    """
    1. Use validation data to select the decision threshold.
    2. Refit the tuned pipeline on train + validation data.
    3. Evaluate once on the untouched test set.
    4. Save model, metrics, plots and feature importance.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get validation probabilities from a model trained on training data.
    validation_model = clone(tuned_pipeline)
    validation_model.fit(X_train, y_train)

    val_probabilities = validation_model.predict_proba(X_val)[:, 1]

    selected_threshold, threshold_reason = select_threshold(
        y_val,
        val_probabilities,
        target_recall=target_recall
    )

    print("\nSelected threshold:", selected_threshold)
    print("Selection reason:", threshold_reason)

    validation_metrics = calculate_metrics(
        y_val,
        val_probabilities,
        selected_threshold
    )

    print("\nValidation metrics at selected threshold:")
    for metric, value in validation_metrics.items():
        print(f"{metric}: {value:.4f}")

    # Refit on all available development data.
    X_train_val = pd.concat([X_train, X_val], axis=0)
    y_train_val = pd.concat([y_train, y_val], axis=0)

    final_model = clone(tuned_pipeline)
    final_model.fit(X_train_val, y_train_val)

    # The test set is used only for final evaluation.
    test_probabilities = final_model.predict_proba(X_test)[:, 1]
    test_predictions = (
        test_probabilities >= selected_threshold
    ).astype(int)

    test_metrics = calculate_metrics(
        y_test,
        test_probabilities,
        selected_threshold
    )

    print("\nFinal test metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nTest classification report:")
    print(
        classification_report(
            y_test,
            test_predictions,
            target_names=["Not Churn", "Churn"],
            zero_division=0
        )
    )

    # Save confusion matrix, ROC and PR curves.
    cm = save_evaluation_plots(
        y_test,
        test_probabilities,
        test_predictions,
        output_dir
    )

    # Save feature importance.
    importance_df = save_feature_importance(
        final_model,
        output_dir
    )

    # Save metrics and threshold information.
    results = {
        "best_threshold": selected_threshold,
        "threshold_selection_reason": threshold_reason,
        "target_recall": target_recall,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "confusion_matrix": cm.tolist()
    }

    with open(output_dir / "evaluation_metrics.json", "w") as file:
        json.dump(results, file, indent=4)

    pd.DataFrame([test_metrics]).to_csv(
        output_dir / "test_metrics.csv",
        index=False
    )

    # Save the final fitted pipeline.
    if model_file is not None:
        model_file = Path(model_file)
        model_file.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, model_file)
        print("\nFinal model saved to:", model_file)

    print("\nAll evaluation artifacts saved to:", output_dir)

    return final_model, selected_threshold, test_metrics, importance_df
