"""
CRC Exploratory Data Analysis and Biomarker Discovery.

This module merges clinical metadata (eLife-65088 supplementary file 10)
with taxonomic abundance data (supplementary file 5), performs a
Mann-Whitney U test for each bacterial taxon to identify candidate
biomarkers differentiating colorectal cancer (CRC) samples from
control/healthy samples, and generates a boxplot for the most
significant candidate.
"""

import logging
import os
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
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

# Minimum number of non-missing observations required per group for a
# taxon to be included in the statistical test.
MIN_GROUP_SIZE = 5

OUTPUT_PLOT_PATH = "top_biomarker_boxplot.png"


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

    logger.info("Metadata ID sample value: %s", metadata_df[metadata_id_col].iloc[0])
    logger.info("Taxonomy ID sample value: %s", taxa_df[taxa_id_col].iloc[0])

    merged_df = pd.merge(
        metadata_df,
        taxa_df,
        left_on=metadata_id_col,
        right_on=taxa_id_col,
        how="inner",
    )
    return merged_df


def compute_biomarker_p_values(
    merged_df: pd.DataFrame,
    numeric_columns: list,
    status_column: str,
) -> pd.DataFrame:
    """
    Compute Mann-Whitney U test p-values comparing CRC vs. control samples
    for each candidate taxon.

    Args:
        merged_df: The merged metadata/taxonomy DataFrame.
        numeric_columns: List of column names representing taxon
            abundance values to test.
        status_column: Name of the column indicating disease status
            (used to split samples into CRC and control groups).

    Returns:
        A DataFrame with columns "Species" and "p_value", sorted in
        ascending order of p-value.
    """
    crc_samples = merged_df[
        merged_df[status_column].astype(str).str.upper().str.contains("CRC")
    ]
    ctrl_samples = merged_df[
        merged_df[status_column].astype(str).str.upper().str.contains("CONTROL|HEALTHY")
    ]

    p_values: Dict[str, float] = {}
    for species in numeric_columns:
        try:
            crc_vals = pd.to_numeric(crc_samples[species], errors="coerce").dropna()
            ctrl_vals = pd.to_numeric(ctrl_samples[species], errors="coerce").dropna()

            if len(crc_vals) > MIN_GROUP_SIZE and len(ctrl_vals) > MIN_GROUP_SIZE:
                if crc_vals.sum() > 0 or ctrl_vals.sum() > 0:
                    _, p_value = mannwhitneyu(crc_vals, ctrl_vals)
                    p_values[species] = p_value
        except Exception:
            continue

    p_value_df = pd.DataFrame(
        list(p_values.items()), columns=["Species", "p_value"]
    ).sort_values("p_value")

    return p_value_df


def plot_top_biomarker(
    merged_df: pd.DataFrame,
    status_column: str,
    top_species: str,
    top_p_value: float,
    output_path: str,
) -> None:
    """
    Generate and save a boxplot comparing the top candidate biomarker's
    abundance across disease status groups.

    Args:
        merged_df: The merged metadata/taxonomy DataFrame.
        status_column: Name of the column indicating disease status.
        top_species: Name of the taxon column with the lowest p-value.
        top_p_value: The p-value associated with top_species.
        output_path: File path to save the resulting plot image.

    Returns:
        None. The plot is saved to output_path.
    """
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=merged_df, x=status_column, y=top_species, palette="Set2")
    plt.title(f"CRC vs Control: {top_species}\n(p-value: {top_p_value:.2e})")
    plt.ylabel("Relative Abundance")
    plt.xlabel("Disease Status")
    plt.tight_layout()
    plt.savefig(output_path)


def run_analysis(metadata_path: str, taxonomy_path: str) -> None:
    """
    Execute the full CRC EDA and biomarker discovery pipeline.

    Steps performed:
        1. Verify that both input files exist.
        2. Load metadata and taxonomic abundance data.
        3. Normalize taxonomy table orientation and identify ID columns.
        4. Merge the two datasets on sample ID.
        5. Compute Mann-Whitney U p-values for each candidate taxon.
        6. Report the top 5 candidate biomarkers.
        7. Plot and save a boxplot for the most significant biomarker.

    Args:
        metadata_path: Path to the clinical metadata Excel file.
        taxonomy_path: Path to the taxonomic abundance Excel file.

    Returns:
        None. Progress and results are written to the logger, and a plot
        image is saved to disk when a top biomarker is identified.
    """
    logger.info("Starting CRC data integration and biomarker analysis.")

    if not (os.path.exists(metadata_path) and os.path.exists(taxonomy_path)):
        logger.error(
            "One or both input files are missing: '%s', '%s'.",
            metadata_path,
            taxonomy_path,
        )
        return

    logger.info("Reading input files.")
    metadata_df, taxa_df = load_input_files(metadata_path, taxonomy_path)

    taxa_df = normalize_taxonomy_orientation(taxa_df)

    metadata_id_col = find_sample_id_column(metadata_df)
    taxa_id_col = find_sample_id_column(taxa_df)

    merged_df = merge_metadata_and_taxonomy(
        metadata_df, taxa_df, metadata_id_col, taxa_id_col
    )
    logger.info("Merged dataset shape: %s", (merged_df.shape,))

    if merged_df.shape[0] == 0:
        logger.warning(
            "Sample IDs did not match between datasets. "
            "Please inspect the logged ID samples above."
        )
        return

    status_column = (
        "study_condition" if "study_condition" in merged_df.columns else "group"
    )

    numeric_columns = [c for c in taxa_df.columns if c != taxa_id_col]
    p_value_df = compute_biomarker_p_values(merged_df, numeric_columns, status_column)

    logger.info("--- Top 5 Candidate Biomarkers by Statistical Significance ---")
    logger.info("\n%s", p_value_df.head(5).to_string(index=False))

    if not p_value_df.empty:
        top_species = p_value_df.iloc[0]["Species"]
        top_p_value = p_value_df.iloc[0]["p_value"]
        logger.info("Generating plot for top biomarker: '%s'.", top_species)

        plot_top_biomarker(
            merged_df, status_column, top_species, top_p_value, OUTPUT_PLOT_PATH
        )
        logger.info("Plot saved as '%s'.", OUTPUT_PLOT_PATH)


if __name__ == "__main__":
    run_analysis(METADATA_PATH, TAXONOMY_PATH)
