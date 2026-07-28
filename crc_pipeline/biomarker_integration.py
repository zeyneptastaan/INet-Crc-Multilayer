"""
Biomarker Data Integration Pipeline.

This module integrates multi-layer microbiome biomarker data by combining:
    1. Bacterial species-to-EC (Enzyme Commission) number mappings.
    2. EC-to-MetaCyc reaction mappings.
    3. MetaCyc pathway structure data.

It also performs a data-coverage analysis to quantify how many bacterial
species have associated enzymatic annotations, and cross-checks a curated
list of clinically relevant gut microbiota species (frequently reported in
colorectal cancer and gut microbiome literature) against the integrated
dataset.
"""

import logging
from typing import Dict, List

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
SPECIES_EC_PATH = "species_to_ec.txt"
REACTIONS_PATH = "metacyc_reactions_level4ec.txt"
PATHWAYS_PATH = "metacyc_pathways_structured_v24"

# Reference list of clinically relevant gut microbiota species, frequently
# reported in colorectal cancer (CRC) and gut microbiome literature.
SAMPLE_MICROBIOTA_SPECIES = [
    "Fusobacterium_nucleatum",
    "Bacteroides_fragilis",
    "Escherichia_coli",
    "Akkermansia_muciniphila",
    "Faecalibacterium_prausnitzii",
    "Peptostreptococcus_stomatis",
]


def load_species_to_ec_mapping(file_path: str) -> Dict[str, List[str]]:
    """
    Parse the species-to-EC mapping file into a dictionary.

    The parser supports tab-delimited, colon-delimited, and
    whitespace-delimited line formats, selected automatically per line.

    Args:
        file_path: Path to the species-to-EC mapping text file. Each line
            is expected to start with a species identifier, followed by
            one or more associated EC numbers.

    Returns:
        A dictionary mapping each species name (str) to a list of its
        associated EC numbers (List[str]).
    """
    species_to_ec_map: Dict[str, List[str]] = {}

    with open(file_path, "r", encoding="utf-8") as file_handle:
        for line in file_handle:
            line = line.strip()
            if not line:
                continue

            if "\t" in line:
                parts = line.split("\t")
            elif ":" in line:
                parts = line.split(":")
            else:
                parts = line.split()

            species = parts[0].strip()
            ec_list = [ec.strip() for ec in parts[1:] if ec.strip()]
            species_to_ec_map[species] = ec_list

    return species_to_ec_map


def load_ec_to_reaction_mapping(file_path: str) -> Dict[str, List[str]]:
    """
    Parse the EC-to-reaction mapping file into a dictionary.

    Args:
        file_path: Path to the MetaCyc reactions file
            (metacyc_reactions_level4ec.txt). Each non-comment line is
            expected to contain an EC number and a reaction identifier,
            delimited by a tab or whitespace.

    Returns:
        A dictionary mapping each EC number (str) to a list of associated
        reaction identifiers (List[str]).
    """
    ec_to_reaction_map: Dict[str, List[str]] = {}

    with open(file_path, "r", encoding="utf-8", errors="ignore") as file_handle:
        for line in file_handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) >= 2:
                ec_number = parts[0].strip()
                reaction = parts[1].strip()
                ec_to_reaction_map.setdefault(ec_number, []).append(reaction)

    return ec_to_reaction_map


def count_pathway_lines(file_path: str) -> int:
    """
    Count the number of non-empty lines in the pathway structure file.

    Args:
        file_path: Path to the structured MetaCyc pathways file
            (metacyc_pathways_structured_v24).

    Returns:
        The count of non-empty lines in the file, used as a proxy for the
        number of structured pathway records.
    """
    pathway_count = 0

    with open(file_path, "r", encoding="utf-8", errors="ignore") as file_handle:
        for line in file_handle:
            if line.strip():
                pathway_count += 1

    return pathway_count


def analyze_species_ec_coverage(
    species_to_ec_map: Dict[str, List[str]]
) -> Dict[str, object]:
    """
    Analyze EC-number coverage across all bacterial species.

    Args:
        species_to_ec_map: Dictionary mapping species names to their
            associated EC numbers, as produced by
            `load_species_to_ec_mapping`.

    Returns:
        A dictionary containing:
            - "total_species" (int): Total number of species.
            - "species_with_ec" (int): Number of species with at least
              one valid EC number.
            - "species_with_no_ec" (List[str]): Names of species lacking
              EC annotations.
            - "coverage_rate" (float): Percentage of species with at
              least one valid EC number.
    """
    total_species = len(species_to_ec_map)
    species_with_no_ec: List[str] = []
    species_with_ec = 0

    for species, ec_numbers in species_to_ec_map.items():
        if len(ec_numbers) == 0 or (len(ec_numbers) == 1 and ec_numbers[0] == ""):
            species_with_no_ec.append(species)
        else:
            species_with_ec += 1

    coverage_rate = (species_with_ec / total_species) * 100 if total_species else 0.0

    return {
        "total_species": total_species,
        "species_with_ec": species_with_ec,
        "species_with_no_ec": species_with_no_ec,
        "coverage_rate": coverage_rate,
    }


def check_sample_microbiota_presence(
    species_to_ec_map: Dict[str, List[str]],
    sample_species: List[str],
) -> None:
    """
    Cross-check a reference list of microbiota species against the
    integrated species-to-EC dataset and log the result for each.

    A case-insensitive substring match is used to accommodate naming
    variants between the reference list and the dataset keys.

    Args:
        species_to_ec_map: Dictionary mapping species names to their
            associated EC numbers.
        sample_species: List of species names to check for presence in
            the dataset.

    Returns:
        None. Results are written to the logger.
    """
    logger.info("Checking reference microbiota species against dataset.")

    for species in sample_species:
        matches = [
            candidate
            for candidate in species_to_ec_map
            if species.lower() in candidate.lower()
        ]
        if matches:
            matched_species = matches[0]
            ec_count = len(species_to_ec_map[matched_species])
            logger.info(
                "Found: %s -> matched '%s' (%d associated EC numbers).",
                species,
                matched_species,
                ec_count,
            )
        else:
            logger.warning("Missing: %s -> not found in dataset.", species)


def run_pipeline() -> None:
    """
    Execute the full biomarker data integration pipeline.

    Steps performed:
        1. Load species-to-EC mappings.
        2. Load EC-to-reaction mappings.
        3. Count structured pathway records.
        4. Compute and report EC-annotation coverage statistics.
        5. Cross-check reference microbiota species against the dataset.

    Returns:
        None. Progress and results are written to the logger.
    """
    logger.info("Starting biomarker data integration pipeline.")

    species_to_ec_map = load_species_to_ec_mapping(SPECIES_EC_PATH)
    logger.info(
        "Step 1/3 complete: loaded species-to-EC data for %d species.",
        len(species_to_ec_map),
    )

    ec_to_reaction_map = load_ec_to_reaction_mapping(REACTIONS_PATH)
    logger.info(
        "Step 2/3 complete: mapped %d enzymes to reactions.",
        len(ec_to_reaction_map),
    )

    pathway_count = count_pathway_lines(PATHWAYS_PATH)
    logger.info(
        "Step 3/3 complete: parsed approximately %d pathway records.",
        pathway_count,
    )

    logger.info("All data layers integrated successfully.")

    # Data coverage analysis.
    coverage_stats = analyze_species_ec_coverage(species_to_ec_map)
    logger.info("--- Data Coverage Report ---")
    logger.info("Total species: %d", coverage_stats["total_species"])
    logger.info(
        "Species with EC annotation: %d (%.2f%%)",
        coverage_stats["species_with_ec"],
        coverage_stats["coverage_rate"],
    )
    logger.info(
        "Species missing EC annotation: %d (%.2f%%)",
        len(coverage_stats["species_with_no_ec"]),
        100 - coverage_stats["coverage_rate"],
    )

    if coverage_stats["species_with_no_ec"]:
        logger.info("Sample species missing EC annotation (first 5):")
        for species in coverage_stats["species_with_no_ec"][:5]:
            logger.info("  - %s", species)

    # Reference microbiota presence check.
    check_sample_microbiota_presence(species_to_ec_map, SAMPLE_MICROBIOTA_SPECIES)


if __name__ == "__main__":
    run_pipeline()
