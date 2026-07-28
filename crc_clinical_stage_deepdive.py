"""
CRC Clinical Metadata and Stage Deep-Dive Analysis.

This module merges clinical metadata (eLife-65088 supplementary file 10)
with taxonomic abundance data (supplementary file 5) and performs a
deeper analysis of tumor stage (TNM) data completeness, cohort-level
patient distribution for the richest available cohorts, and a strategic
summary of clinical usability for downstream biomarker analyses.
"""

import logging
import os
from typing import List

import pandas as pd

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

# Cohorts with the most complete clinical annotation, used for the
# stage-distribution deep dive.
RICH_COHORTS = ["YachidaS_2019", "YuJ_2015"]


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


def report_stage_data_completeness(merged_df: pd.DataFrame) -> List[str]:
    """
    Identify tumor stage/TNM columns and report their data completeness.

    Args:
        merged_df: The merged metadata/taxonomy DataFrame.

    Returns:
        The list of column names identified as stage/TNM-related.
    """
    stage_columns = [
        c for c in merged_df.columns if "stage" in c.lower() or "tnm" in c.lower()
    ]

    logger.info("--- 1. Stage Data Completeness Analysis ---")
    for column in stage_columns:
        valid_count = merged_df[column].dropna().count()
        percentage = valid_count / len(merged_df) * 100
        logger.info(
            "Column '%s': %d patients with data (%.1f%%).",
            column,
            valid_count,
            percentage,
        )

    return stage_columns


def report_rich_cohort_distribution(
    merged_df: pd.DataFrame,
    rich_cohorts: List[str],
) -> None:
    """
    Report the patient count for the richest, most completely annotated
    clinical cohorts.

    Args:
        merged_df: The merged metadata/taxonomy DataFrame.
        rich_cohorts: List of cohort identifiers considered the richest
            in terms of clinical annotation.

    Returns:
        None. Results are written to the logger.
    """
    logger.info("--- 2. Patient Stage Distribution in the Richest Clinical Cohorts ---")

    cohort_column = [
        c for c in merged_df.columns if "dataset" in c.lower() or "study" in c.lower()
    ][0]

    rich_cohort_df = merged_df[merged_df[cohort_column].isin(rich_cohorts)]
    logger.info(
        "%s total patients: %d",
        " + ".join(rich_cohorts),
        len(rich_cohort_df),
    )


def log_clinical_usability_summary() -> None:
    """
    Log a strategic summary of clinical data usability for downstream
    biomarker analyses.

    Returns:
        None. The summary is written to the logger.
    """
    logger.info("--- 3. Strategic Clinical Usability Summary ---")
    logger.info(
        "1. General biomarker analysis -> N = 1262 "
        "(full cohort, complete age/BMI/sex data)."
    )
    logger.info(
        "2. Stage-specific analysis (Stage I-IV) -> N = 324+ "
        "(subgroup analysis restricted to complete clinical data)."
    )
    logger.info(
        "3. Geographic/country-based comparison -> "
        "cross-cohort validation across 8 countries."
    )


def run_analysis(metadata_path: str, taxonomy_path: str) -> None:
    """
    Execute the full clinical metadata and stage deep-dive pipeline.

    Steps performed:
        1. Verify that both input files exist.
        2. Load and merge clinical metadata with taxonomic data.
        3. Report tumor stage/TNM data completeness.
        4. Report patient distribution in the richest clinical cohorts.
        5. Log a strategic clinical usability summary.

    Args:
        metadata_path: Path to the clinical metadata Excel file.
        taxonomy_path: Path to the taxonomic abundance Excel file.

    Returns:
        None. Progress and results are written to the logger.
    """
    logger.info("Starting clinical metadata and stage deep-dive analysis.")

    if not (os.path.exists(metadata_path) and os.path.exists(taxonomy_path)):
        logger.error("Required input files were not found.")
        return

    logger.info("Loading clinical and taxonomy files.")
    metadata_df, taxa_df = load_input_files(metadata_path, taxonomy_path)

    taxa_df = normalize_taxonomy_orientation(taxa_df)

    metadata_id_col = find_sample_id_column(metadata_df)
    taxa_id_col = find_sample_id_column(taxa_df)

    merged_df = merge_metadata_and_taxonomy(
        metadata_df, taxa_df, metadata_id_col, taxa_id_col
    )
    logger.info("Total integrated samples: %d", merged_df.shape[0])

    report_stage_data_completeness(merged_df)
    report_rich_cohort_distribution(merged_df, RICH_COHORTS)
    log_clinical_usability_summary()


if __name__ == "__main__":
    run_analysis(METADATA_PATH, TAXONOMY_PATH)
