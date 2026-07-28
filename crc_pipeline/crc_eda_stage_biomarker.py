"""
CRC Clinical Stage (Stage I-IV) Biomarker Exploratory Data Analysis.

This module merges clinical metadata (eLife-65088 supplementary file 10)
with taxonomic abundance data (supplementary file 5), robustly categorizes
patients into early-stage (I-II) and late-stage (III-IV) groups based on
TNM/stage annotations, tests a candidate biomarker taxon for a significant
abundance shift between stage groups using the Mann-Whitney U test, and
generates a comparative boxplot.
"""

import logging
import os
import re
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

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
OUTPUT_PLOT_PATH = "biomarker_stage_progression.png"

# Genus used to select the candidate biomarker taxon for stage comparison.
TARGET_GENUS = "Peptostreptococcus"

# Values treated as missing/unusable when categorizing stage.
INVALID_STAGE_VALUES = ("nan", "none", "unknown", "", "m")


def load_input_files(metadata_path: str, taxonomy_path: str) -> tuple:
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


def categorize_stage_robust(value: object) -> Optional[str]:
    """
    Categorize a raw TNM/stage value into a coarse stage group.

    Recognizes a wide range of TNM and stage notations (e.g. "M1",
    "T3", "Stage III") and classifies them into "Late Stage (III-IV)"
    or "Early Stage (I-II)". Values that cannot be confidently
    classified are treated as missing.

    Args:
        value: The raw stage/TNM value from the clinical metadata.

    Returns:
        "Late Stage (III-IV)", "Early Stage (I-II)", or np.nan if the
        value cannot be classified.
    """
    normalized_value = str(value).strip().lower()

    if normalized_value in INVALID_STAGE_VALUES:
        return np.nan

    # Metastasis and advanced-stage terms.
    if (
        "m1" in normalized_value
        or "met" in normalized_value
        or re.search(r"\b(iii|iv|3|4)\b", normalized_value)
        or "t3" in normalized_value
        or "t4" in normalized_value
    ):
        return "Late Stage (III-IV)"

    # Early-stage and in-situ terms.
    if (
        "tis" in normalized_value
        or re.search(r"\b(i|ii|0|1|2)\b", normalized_value)
        or "t1" in normalized_value
        or "t2" in normalized_value
    ):
        return "Early Stage (I-II)"

    return np.nan


def build_staged_dataframe(merged_df: pd.DataFrame, tnm_column: str) -> pd.DataFrame:
    """
    Filter samples with valid TNM/stage data and assign a coarse stage
    group to each.

    Args:
        merged_df: The merged metadata/taxonomy DataFrame.
        tnm_column: Name of the column containing raw TNM/stage values.

    Returns:
        A copy of merged_df restricted to samples with a non-null
        "Stage_Group" value, which is added as a new column.
    """
    staged_df = merged_df[merged_df[tnm_column].notnull()].copy()
    staged_df["Stage_Group"] = staged_df[tnm_column].apply(categorize_stage_robust)
    staged_df = staged_df[staged_df["Stage_Group"].notnull()]
    return staged_df


def compare_biomarker_between_stages(
    staged_df: pd.DataFrame,
    target_species: str,
) -> Optional[float]:
    """
    Compare a candidate biomarker's abundance between early- and
    late-stage groups using the Mann-Whitney U test.

    Args:
        staged_df: DataFrame with a "Stage_Group" column, as produced by
            `build_staged_dataframe`.
        target_species: Name of the taxon abundance column to test.

    Returns:
        The Mann-Whitney U test p-value, or None if either stage group
        has no valid numeric observations.
    """
    early_vals = pd.to_numeric(
        staged_df[staged_df["Stage_Group"] == "Early Stage (I-II)"][target_species],
        errors="coerce",
    ).dropna()
    late_vals = pd.to_numeric(
        staged_df[staged_df["Stage_Group"] == "Late Stage (III-IV)"][target_species],
        errors="coerce",
    ).dropna()

    if len(early_vals) == 0 or len(late_vals) == 0:
        return None

    _, p_value = mannwhitneyu(early_vals, late_vals)
    return p_value


def plot_stage_biomarker_progression(
    staged_df: pd.DataFrame,
    target_species: str,
    p_value: float,
    output_path: str,
) -> None:
    """
    Generate and save a boxplot comparing biomarker abundance across
    early- and late-stage groups.

    Args:
        staged_df: DataFrame with a "Stage_Group" column.
        target_species: Name of the taxon abundance column to plot.
        p_value: Mann-Whitney U test p-value, shown in the plot title.
        output_path: File path to save the resulting plot image.

    Returns:
        None. The plot is saved to output_path.
    """
    species_label = target_species.split("|")[-1]

    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=staged_df,
        x="Stage_Group",
        y=target_species,
        hue="Stage_Group",
        palette="Blues",
        legend=False,
    )
    plt.title(f"Early vs Late Stage Biomarker Shift\n{species_label} (p-value: {p_value:.2e})")
    plt.ylabel("Relative Abundance")
    plt.xlabel("Clinical Stage Group")
    plt.tight_layout()
    plt.savefig(output_path)


def run_analysis(metadata_path: str, taxonomy_path: str) -> None:
    """
    Execute the full clinical stage biomarker EDA pipeline.

    Steps performed:
        1. Verify that both input files exist.
        2. Load and merge clinical metadata with taxonomic data.
        3. Identify the TNM/stage column and categorize patients into
           early- and late-stage groups.
        4. Report the resulting stage group distribution.
        5. Test the target biomarker taxon for a significant abundance
           difference between stage groups.
        6. Plot and save the comparison if the test could be performed.

    Args:
        metadata_path: Path to the clinical metadata Excel file.
        taxonomy_path: Path to the taxonomic abundance Excel file.

    Returns:
        None. Progress and results are written to the logger, and a plot
        image is saved to disk when applicable.
    """
    logger.info("Starting clinical stage biomarker EDA analysis.")

    if not (os.path.exists(metadata_path) and os.path.exists(taxonomy_path)):
        logger.error("Required input files were not found.")
        return

    logger.info("Reading input files.")
    metadata_df, taxa_df = load_input_files(metadata_path, taxonomy_path)

    taxa_df = normalize_taxonomy_orientation(taxa_df)

    metadata_id_col = find_sample_id_column(metadata_df)
    taxa_id_col = find_sample_id_column(taxa_df)

    merged_df = merge_metadata_and_taxonomy(
        metadata_df, taxa_df, metadata_id_col, taxa_id_col
    )

    tnm_column = [
        c for c in merged_df.columns if "tnm" in c.lower() or "stage" in c.lower()
    ][0]

    staged_df = build_staged_dataframe(merged_df, tnm_column)
    logger.info(
        "Categorized stage group distribution:\n%s",
        staged_df["Stage_Group"].value_counts(),
    )

    target_species_columns = [c for c in taxa_df.columns if TARGET_GENUS in c]

    if not target_species_columns or len(staged_df) == 0:
        logger.warning("No target biomarker column or staged samples available.")
        return

    target_species = target_species_columns[0]
    p_value = compare_biomarker_between_stages(staged_df, target_species)

    if p_value is None:
        logger.warning(
            "Insufficient data in one or both stage groups for '%s'.",
            target_species,
        )
        return

    species_label = target_species.split("|")[-1]
    logger.info(
        "Early vs Late Stage Mann-Whitney U p-value for %s: %.4e",
        species_label,
        p_value,
    )

    plot_stage_biomarker_progression(
        staged_df, target_species, p_value, OUTPUT_PLOT_PATH
    )
    logger.info("Plot updated and saved as '%s'.", OUTPUT_PLOT_PATH)


if __name__ == "__main__":
    run_analysis(METADATA_PATH, TAXONOMY_PATH)
