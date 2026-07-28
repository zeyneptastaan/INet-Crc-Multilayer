"""
Biomarker-Based Machine Learning Classification (Random Forest).

This module merges clinical metadata (eLife-65088 supplementary file 10)
with taxonomic abundance data (supplementary file 5), trains a Random
Forest classifier to distinguish colorectal cancer (CRC) samples from
control/healthy samples based on bacterial abundance features, evaluates
model performance via ROC-AUC and a classification report, reports the
top contributing biomarker features, and saves the ROC curve plot.
"""

import logging
import os
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, classification_report, roc_curve
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------
# Logging Configuration
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# File Paths
# ----------------------------------------------------------------------
METADATA_PATH = "elife-65088-supp10-v1.xlsx"
TAXONOMY_PATH = "elife-65088-supp5-v1.xlsx"
OUTPUT_PLOT_PATH = "crc_ml_roc_curve.png"

# Train/test split configuration.
TEST_SET_SIZE = 0.20
RANDOM_STATE = 42

# Random Forest hyperparameters.
N_ESTIMATORS = 100

# Number of top contributing biomarker features to report.
TOP_FEATURE_COUNT = 5

# Class labels for the classification report, in target order [0, 1].
CLASS_LABELS = ["Healthy", "CRC"]


def load_input_files(metadata_path: str, taxonomy_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the clinical metadata and taxonomic abundance Excel files.

    Args:
        metadata_path: Path to the clinical metadata Excel file.
        taxonomy_path: Path to the taxonomic abundance Excel file.

    Returns:
        A tuple of (metadata_df, taxonomy_df) DataFrames.
    """
    metadata_df = pd.read_excel(metadata_path)
    taxonomy_df = pd.read_excel(taxonomy_path)
    return metadata_df, taxonomy_df


def normalize_taxonomy_orientation(taxa_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the taxonomy table is oriented with samples as rows.

    If the first column of the taxonomy table does not appear to be a
    sample identifier, the table is assumed to be transposed (taxa as
    rows, samples as columns) and is transposed back accordingly.

    Args:
        taxa_df: The raw taxonomic abundance DataFrame.

    Returns:
        A DataFrame oriented with one row per sample, with a "sampleID"
        column identifying each sample.
    """
    first_column = taxa_df.columns[0]

    if "sample" not in str(first_column).lower():
        taxa_df = taxa_df.set_index(first_column).T.reset_index()
        taxa_df.rename(columns={taxa_df.columns[0]: "sampleID"}, inplace=True)

    return taxa_df


def find_sample_id_column(df: pd.DataFrame) -> str:
    """
    Locate the column representing sample identifiers.

    Args:
        df: A DataFrame expected to contain a sample ID column.

    Returns:
        The name of the first column whose name contains "sample".

    Raises:
        IndexError: If no matching column is found.
    """
    return [c for c in df.columns if "sample" in c.lower()][0]


def merge_metadata_and_taxonomy(
    metadata_df: pd.DataFrame,
    taxa_df: pd.DataFrame,
    metadata_id_col: str,
    taxa_id_col: str,
) -> pd.DataFrame:
    """
    Clean sample ID columns and merge metadata with taxonomic data.

    Args:
        metadata_df: The clinical metadata DataFrame.
        taxa_df: The taxonomic abundance DataFrame.
        metadata_id_col: Name of the sample ID column in metadata_df.
        taxa_id_col: Name of the sample ID column in taxa_df.

    Returns:
        A DataFrame containing the inner join of metadata_df and taxa_df
        on their respective sample ID columns.
    """
    metadata_df[metadata_id_col] = metadata_df[metadata_id_col].astype(str).str.strip()
    taxa_df[taxa_id_col] = taxa_df[taxa_id_col].astype(str).str.strip()

    merged_df = pd.merge(
        metadata_df,
        taxa_df,
        left_on=metadata_id_col,
        right_on=taxa_id_col,
        how="inner",
    )
    return merged_df


def build_ml_dataset(
    merged_df: pd.DataFrame,
    taxa_features: list,
    status_column: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build the feature matrix and binary target vector for classification.

    Only samples labeled as CRC, control, or healthy are retained. The
    binary target is 1 for CRC samples and 0 for control/healthy samples.

    Args:
        merged_df: The merged metadata/taxonomy DataFrame.
        taxa_features: List of taxonomic abundance column names to use
            as model features.
        status_column: Name of the column indicating disease status.

    Returns:
        A tuple of (X, y) where X is the feature DataFrame and y is the
        binary target Series.
    """
    ml_df = merged_df[
        merged_df[status_column].astype(str).str.upper().str.contains("CRC|CONTROL|HEALTHY")
    ].copy()
    ml_df["Target"] = ml_df[status_column].astype(str).str.upper().apply(
        lambda x: 1 if "CRC" in x else 0
    )

    X = ml_df[taxa_features].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = ml_df["Target"]

    return X, y


def train_random_forest_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier on the training data.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.

    Returns:
        The fitted RandomForestClassifier instance.
    """
    rf_model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE
    )
    rf_model.fit(X_train, y_train)
    return rf_model


def evaluate_model(
    rf_model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[float, str]:
    """
    Evaluate the trained model on the test set via ROC-AUC and a
    classification report.

    Args:
        rf_model: The fitted RandomForestClassifier.
        X_test: Test feature matrix.
        y_test: Test target vector.

    Returns:
        A tuple of (roc_auc, report_text, fpr, tpr) where roc_auc is the
        area under the ROC curve and report_text is the formatted
        classification report string.
    """
    y_pred = rf_model.predict(X_test)
    y_probs = rf_model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    report_text = classification_report(y_test, y_pred, target_names=CLASS_LABELS)

    return roc_auc, report_text, fpr, tpr


def log_top_feature_importances(
    rf_model: RandomForestClassifier,
    taxa_features: list,
    top_n: int,
) -> None:
    """
    Log the top contributing biomarker features by importance.

    Args:
        rf_model: The fitted RandomForestClassifier.
        taxa_features: List of feature column names, in the same order
            used to train the model.
        top_n: Number of top features to log.

    Returns:
        None. Results are written to the logger.
    """
    importances = pd.Series(
        rf_model.feature_importances_, index=taxa_features
    ).sort_values(ascending=False)

    logger.info("Top %d most influential bacterial biomarker features:", top_n)
    for species, importance in importances.head(top_n).items():
        clean_species = species.split("|")[-1]
        logger.info("  %s: %.2f%% contribution", clean_species, importance * 100)


def plot_roc_curve(
    fpr,
    tpr,
    roc_auc: float,
    output_path: str,
) -> None:
    """
    Generate and save the ROC curve for the trained classifier.

    Args:
        fpr: Array of false positive rates.
        tpr: Array of true positive rates.
        roc_auc: Area under the ROC curve.
        output_path: File path to save the resulting plot image.

    Returns:
        None. The plot is saved to output_path.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"Random Forest ROC (AUC = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("CRC Diagnostic Biomarker Model Performance (ROC Curve)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)


def run_pipeline() -> None:
    """
    Execute the full biomarker-based machine learning classification
    pipeline.

    Steps performed:
        1. Verify that both input files exist.
        2. Load and merge clinical metadata with taxonomic data.
        3. Build the classification feature matrix and target vector.
        4. Split the data into training and test sets.
        5. Train a Random Forest classifier.
        6. Evaluate the model via ROC-AUC and a classification report.
        7. Report the top contributing biomarker features.
        8. Plot and save the ROC curve.

    Returns:
        None. Progress and results are written to the logger, and a plot
        image is saved to disk.
    """
    logger.info("Starting biomarker-based machine learning classification (Random Forest).")

    if not (os.path.exists(METADATA_PATH) and os.path.exists(TAXONOMY_PATH)):
        logger.error("Required input files were not found.")
        return

    logger.info("Reading input files and preparing the machine learning dataset.")
    metadata_df, taxa_df = load_input_files(METADATA_PATH, TAXONOMY_PATH)

    taxa_df = normalize_taxonomy_orientation(taxa_df)

    metadata_id_col = find_sample_id_column(metadata_df)
    taxa_id_col = find_sample_id_column(taxa_df)

    merged_df = merge_metadata_and_taxonomy(
        metadata_df, taxa_df, metadata_id_col, taxa_id_col
    )

    status_column = (
        "study_condition" if "study_condition" in merged_df.columns else "group"
    )

    taxa_features = [c for c in taxa_df.columns if c != taxa_id_col]
    X, y = build_ml_dataset(merged_df, taxa_features, status_column)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SET_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    logger.info(
        "Training set size: %d | Test set size: %d", len(X_train), len(X_test)
    )

    rf_model = train_random_forest_classifier(X_train, y_train)

    roc_auc, report_text, fpr, tpr = evaluate_model(rf_model, X_test, y_test)

    logger.info("Model performance (ROC-AUC score): %.4f", roc_auc)
    logger.info("--- Classification Report ---")
    logger.info("\n%s", report_text)

    log_top_feature_importances(rf_model, taxa_features, TOP_FEATURE_COUNT)

    plot_roc_curve(fpr, tpr, roc_auc, OUTPUT_PLOT_PATH)
    logger.info("ROC curve plot saved as '%s'.", OUTPUT_PLOT_PATH)


if __name__ == "__main__":
    run_pipeline()
