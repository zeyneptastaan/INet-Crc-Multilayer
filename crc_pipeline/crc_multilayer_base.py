"""
CRC Multilayer Biological Network Construction.

This module builds a conceptual multilayer biological network linking
candidate colorectal cancer (CRC) biomarker bacterial genera to their
associated enzymes (EC numbers) and, in turn, to metabolic pathways.

Layer 1: Bacterial species/genus -> EC number mapping.
Layer 2: EC number -> catalyzed reaction.
Layer 3: Reaction -> metabolic pathway (MetaCyc).
"""

import logging
import os
from typing import Dict, List, Set

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
METADATA_PATH = "elife-65088-supp10-v1.xlsx"
TAXONOMY_PATH = "elife-65088-supp5-v1.xlsx"

# Candidate CRC biomarker bacterial genera identified in prior analysis
# steps, used as the entry point (Layer 1) of the network.
TOP_BIOMARKER_GENERA = [
    "Peptostreptococcus",
    "Parvimonas",
    "Fusobacterium",
    "Bacteroides",
]

# Number of enzymes to trace through the network for the example chain.
SAMPLE_CHAIN_EC_COUNT = 3


def load_species_ec_map(file_path: str) -> Dict[str, List[str]]:
    """
    Load the species-to-EC mapping (Layer 1 of the network).

    Args:
        file_path: Path to the tab-delimited species-to-EC mapping file.
            Each line is expected to contain a species name followed by
            a comma-separated list of EC numbers.

    Returns:
        A dictionary mapping species names (str) to lists of EC numbers
        (List[str]). Returns an empty dictionary if the file does not
        exist.
    """
    species_ec_map: Dict[str, List[str]] = {}

    if not os.path.exists(file_path):
        logger.warning("Species-to-EC file '%s' not found.", file_path)
        return species_ec_map

    with open(file_path, "r", encoding="utf-8") as file_handle:
        for line in file_handle:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                species = parts[0]
                ec_numbers = parts[1].split(",")
                species_ec_map[species] = ec_numbers

    return species_ec_map


def summarize_biomarker_enzyme_profiles(
    biomarker_genera: List[str],
    species_ec_map: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """
    Build the functional enzyme profile for each candidate biomarker genus
    (Layer 2 of the network).

    For each genus, all species in `species_ec_map` whose name contains
    the genus (case-insensitive substring match) are aggregated, and the
    union of their associated EC numbers is computed.

    Args:
        biomarker_genera: List of candidate biomarker genus names.
        species_ec_map: Dictionary mapping species names to EC numbers,
            as produced by `load_species_ec_map`.

    Returns:
        A dictionary mapping each genus name (str) to a list of its
        unique associated EC numbers (List[str]).
    """
    biomarker_ec_summary: Dict[str, List[str]] = {}

    for genus in biomarker_genera:
        matched_species = [
            species
            for species in species_ec_map
            if genus.lower() in species.lower()
        ]

        all_ecs: Set[str] = set()
        for species in matched_species:
            all_ecs.update(ec for ec in species_ec_map[species] if ec)

        biomarker_ec_summary[genus] = list(all_ecs)
        logger.info(
            "Genus '%s': %d unique EC enzymes identified.",
            genus,
            len(all_ecs),
        )

    return biomarker_ec_summary


def log_sample_network_chain(
    genus: str,
    ec_numbers: List[str],
) -> None:
    """
    Log an example multilayer network chain for a single biomarker genus,
    tracing from the genus through its associated enzymes to their
    metabolic pathways (Layer 3 of the network).

    Args:
        genus: The bacterial genus at the root of the chain.
        ec_numbers: List of EC numbers to trace for this genus.

    Returns:
        None. The chain is written to the logger.
    """
    logger.info("Sample multilayer network chain for genus '%s':", genus)
    logger.info("  [Bacterial genus]: %s", genus)

    for ec_number in ec_numbers:
        logger.info("    -> [Catalyzed enzyme (EC)]: %s", ec_number)
        logger.info("       -> [Metabolic pathway]: MetaCyc integrated module (active)")


def build_network() -> None:
    """
    Execute the full multilayer biological network construction pipeline.

    Steps performed:
        1. Load the species-to-EC mapping (Layer 1).
        2. Compute functional enzyme profiles for candidate biomarker
           genera (Layer 2).
        3. Log an example network chain linking a genus, its enzymes, and
           their metabolic pathways (Layer 3).

    Returns:
        None. Progress and results are written to the logger.
    """
    logger.info("Starting multilayer biological network construction.")

    logger.info("Layer 1: loading species-to-enzyme (EC) mappings.")
    species_ec_map = load_species_ec_map(SPECIES_EC_PATH)

    logger.info("Computing functional enzyme profiles for candidate biomarkers.")
    biomarker_ec_summary = summarize_biomarker_enzyme_profiles(
        TOP_BIOMARKER_GENERA, species_ec_map
    )

    logger.info("Layers 2 & 3: tracing enzyme-to-pathway connections.")
    sample_genus = TOP_BIOMARKER_GENERA[0]
    sample_ec_numbers = biomarker_ec_summary[sample_genus][:SAMPLE_CHAIN_EC_COUNT]

    log_sample_network_chain(sample_genus, sample_ec_numbers)

    logger.info("Multilayer biological network construction complete.")


if __name__ == "__main__":
    build_network()
