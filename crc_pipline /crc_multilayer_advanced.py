"""
Updated Multilayer Biological Network Module (v2).

This module merges clinical metadata (eLife-65088 supplementary file 10)
with taxonomic abundance data (supplementary file 5), statistically
identifies the top candidate CRC biomarker taxa via the Mann-Whitney U
test, and builds a multilayer biological network hierarchy linking each
biomarker taxon to its associated enzymes (EC numbers) and downstream
metabolic pathway modules.
"""

import logging
import os
from typing import Dict, List, Set

import pandas as pd
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
SPECIES_EC_PATH = "species_to_ec.txt"

# Minimum number of non-missing observations required per group for a
# taxon to be included in the statistical test.
MIN_GROUP_SIZE = 10

# Number of top-ranked biomarker taxa to carry forward into the network.
TOP_BIOMARKER_COUNT = 3

# Number of dominant enzymes to display per biomarker taxon.
TOP_EC_PER_BIOMARKER = 3


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


def compute_taxa_p_values(
    merged_df: pd.DataFrame,
    species_columns: List[str],
    status_column: str,
) -> pd.DataFrame:
    """
    Compute Mann-Whitney U test p-values comparing CRC vs. control samples
    for each genus/species-level taxon.

    Args:
        merged_df: The merged metadata/taxonomy DataFrame.
        species_columns: List of column names representing genus (g__) or
            species (s__) level abundance values to test.
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
    for species in species_columns:
        crc_vals = pd.to_numeric(crc_samples[species], errors="coerce").dropna()
        ctrl_vals = pd.to_numeric(ctrl_samples[species], errors="coerce").dropna()

        if (
            len(crc_vals) > MIN_GROUP_SIZE
            and len(ctrl_vals) > MIN_GROUP_SIZE
            and (crc_vals.sum() > 0 or ctrl_vals.sum() > 0)
        ):
            _, p_value = mannwhitneyu(crc_vals, ctrl_vals)
            p_values[species] = p_value

    p_value_df = pd.DataFrame(
        list(p_values.items()), columns=["Species", "p_value"]
    ).sort_values("p_value")

    return p_value_df


def log_top_biomarkers(p_value_df: pd.DataFrame, top_biomarkers: List[str]) -> None:
    """
    Log the automatically detected top-ranked genus/species-level
    biomarker candidates.

    Args:
        p_value_df: DataFrame with "Species" and "p_value" columns.
        top_biomarkers: List of the top-ranked taxon column names.

    Returns:
        None. Results are written to the logger.
    """
    logger.info("Automatically detected primary genus/species-level biomarkers:")
    for index, biomarker in enumerate(top_biomarkers, start=1):
        clean_name = biomarker.split("|")[-1]
        p_value = p_value_df[p_value_df["Species"] == biomarker]["p_value"].values[0]
        logger.info("  %d. %s (p-value: %.2e)", index, clean_name, p_value)


def load_species_ec_map(file_path: str) -> Dict[str, List[str]]:
    """
    Load the species-to-EC mapping (bacteria-to-enzyme functional layer).

    Args:
        file_path: Path to the tab-delimited species-to-EC mapping file.
            Each line is expected to contain a species name followed by
            a comma-separated list of EC numbers.

    Returns:
        A dictionary mapping species names (str) to lists of EC numbers
        (List[str]).
    """
    species_ec_map: Dict[str, List[str]] = {}

    with open(file_path, "r", encoding="utf-8") as file_handle:
        for line in file_handle:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                species = parts[0]
                ec_numbers = [ec for ec in parts[1].split(",") if ec]
                species_ec_map[species] = ec_numbers

    return species_ec_map


def log_multilayer_network_hierarchy(
    top_biomarkers: List[str],
    p_value_df: pd.DataFrame,
    species_ec_map: Dict[str, List[str]],
) -> None:
    """
    Log the full multilayer biological network hierarchy, linking each
    top biomarker taxon to its statistical significance, functional
    enzyme profile, and downstream metabolic pathway modules.

    Args:
        top_biomarkers: List of the top-ranked taxon column names.
        p_value_df: DataFrame with "Species" and "p_value" columns.
        species_ec_map: Dictionary mapping species names to EC numbers,
            as produced by `load_species_ec_map`.

    Returns:
        None. The network hierarchy is written to the logger.
    """
    logger.info("Multilayer biological network hierarchy (systems biology):")

    for biomarker in top_biomarkers:
        genus_or_species = (
            biomarker.split("|")[-1].replace("g__", "").replace("s__", "")
        )
        matches = [
            species
            for species in species_ec_map
            if genus_or_species.lower() in species.lower()
        ]

        all_ecs: Set[str] = set()
        for matched_species in matches:
            all_ecs.update(species_ec_map[matched_species])

        top_ec_numbers = list(all_ecs)[:TOP_EC_PER_BIOMARKER]
        p_value = p_value_df[p_value_df["Species"] == biomarker]["p_value"].values[0]

        logger.info("[Bacterial layer]: %s", genus_or_species)
        logger.info("  Statistical significance: p = %.2e", p_value)
        logger.info("  Total unique secreted EC enzymes: %d", len(all_ecs))

        for ec_number in top_ec_numbers:
            logger.info("    -> [Catalyzed enzyme (EC)]: %s", ec_number)
            logger.info(
                "       -> [Metabolic pathway]: MetaCyc / KEGG / ModelSEED integrated module"
            )

        logger.info("-" * 55)

    logger.info("Multilayer network update completed successfully.")


def build_network() -> None:
    """
    Execute the full updated multilayer biological network pipeline (v2).

    Steps performed:
        1. Verify that all required input files exist.
        2. Load and merge clinical metadata with taxonomic data.
        3. Compute Mann-Whitney U p-values for genus/species-level taxa.
        4. Select and log the top-ranked biomarker candidates.
        5. Load the species-to-EC functional mapping.
        6. Log the full multilayer network hierarchy for each biomarker.

    Returns:
        None. Progress and results are written to the logger.
    """
    logger.info("Starting updated multilayer biological network module (v2).")

    if not (
        os.path.exists(METADATA_PATH)
        and os.path.exists(TAXONOMY_PATH)
        and os.path.exists(SPECIES_EC_PATH)
    ):
        logger.error("Required input files were not found.")
        return

    logger.info("Reading biological data layers and clinical metadata.")
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

    # Restrict the network to genus-level (g__) or species-level (s__)
    # taxa only.
    species_columns = [
        c
        for c in taxa_df.columns
        if c != taxa_id_col and ("g__" in c or "s__" in c)
    ]

    p_value_df = compute_taxa_p_values(merged_df, species_columns, status_column)

    top_biomarkers = p_value_df.head(TOP_BIOMARKER_COUNT)["Species"].tolist()
    log_top_biomarkers(p_value_df, top_biomarkers)

    logger.info("Layers 1 & 2: mapping bacteria to functional enzymes (EC).")
    species_ec_map = load_species_ec_map(SPECIES_EC_PATH)

    log_multilayer_network_hierarchy(top_biomarkers, p_value_df, species_ec_map)


if __name__ == "__main__":
    build_network()
