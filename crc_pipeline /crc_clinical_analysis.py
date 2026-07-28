"""
CRC Clinical Metadata Analysis.

This module analyzes the clinical metadata associated with the eLife-65088
colorectal cancer (CRC) publication (supplementary file 10). It reports the
dataset's dimensions, per-column missing-value statistics, and the sample
distribution across study/cohort groups when such a column can be
identified.
"""

import logging
import os
from typing import List, Optional

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
METADATA_FILE = "elife-65088-supp10-v1.xlsx"

# Keywords used to auto-detect a study/cohort identifier column.
COHORT_COLUMN_KEYWORDS = ("study", "cohort", "dataset")


def load_clinical_metadata(file_path: str) -> pd.DataFrame:
    """
    Load the clinical metadata Excel file into a DataFrame.

    Args:
        file_path: Path to the metadata Excel file (.xlsx).

    Returns:
        A pandas DataFrame containing the clinical metadata.
    """
    return pd.read_excel(file_path)


def summarize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-column counts and percentages of missing values.

    Args:
        df: The clinical metadata DataFrame.

    Returns:
        A DataFrame indexed by column name, containing two columns:
            - "total_missing" (int): Number of missing values.
            - "missing_percentage" (float): Percentage of missing values
              relative to the total number of rows.
    """
    return pd.DataFrame(
        {
            "total_missing": df.isnull().sum(),
            "missing_percentage": (df.isnull().sum() / len(df)) * 100,
        }
    )


def find_cohort_column(df: pd.DataFrame) -> Optional[str]:
    """
    Identify the column most likely to represent study/cohort membership.

    Args:
        df: The clinical metadata DataFrame.

    Returns:
        The name of the first column whose name contains one of the
        cohort-related keywords ("study", "cohort", "dataset"), or None
        if no such column is found.
    """
    matching_columns: List[str] = [
        col
        for col in df.columns
        if any(keyword in col.lower() for keyword in COHORT_COLUMN_KEYWORDS)
    ]
    return matching_columns[0] if matching_columns else None


def run_analysis(file_path: str) -> None:
    """
    Execute the full CRC clinical metadata analysis pipeline.

    Steps performed:
        1. Verify that the metadata file exists.
        2. Load the metadata into a DataFrame.
        3. Report dataset dimensions.
        4. Report per-column missing-value statistics.
        5. Report sample distribution across study/cohort groups, if a
           cohort column can be identified.

    Args:
        file_path: Path to the clinical metadata Excel file.

    Returns:
        None. Progress and results are written to the logger.
    """
    logger.info("Starting CRC clinical metadata analysis (eLife-65088).")

    if not os.path.exists(file_path):
        logger.error("File '%s' was not found in the working directory.", file_path)
        logger.error(
            "Please verify that the file is located in the expected project "
            "folder and that its name matches exactly."
        )
        return

    logger.info("File '%s' found. Loading Excel data.", file_path)
    df = load_clinical_metadata(file_path)

    logger.info("Total samples (rows): %d", df.shape[0])
    logger.info("Total variables (columns): %d", df.shape[1])

    logger.info("--- Column-Wise Missing Data Summary ---")
    missing_summary = summarize_missing_values(df)
    logger.info("\n%s", missing_summary)

    logger.info("--- Sample Distribution by Study/Cohort ---")
    cohort_column = find_cohort_column(df)
    if cohort_column:
        logger.info("\n%s", df[cohort_column].value_counts())
    else:
        logger.warning(
            "No study/cohort column could be auto-detected. "
            "Manual inspection of the first rows is recommended."
        )


if __name__ == "__main__":
    run_analysis(METADATA_FILE)
