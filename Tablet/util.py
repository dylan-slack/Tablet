"""Utils for the project."""
from typing import Union

from transformers import LogitsProcessor


# Instruction types
PROTOTYPES = "prototypes"
RULESET = "ruleset"
# Instruction encodings
TEMPLATES = "templates"
SYNTHETIC = "synthetic"
NATLANGUAGE = "naturallanguage"
# Experiment names
PERF = "performance"  # aggregate performance
MISMATCHIID = "mismatchiid"  # iid performance of domain mismatch
MISMATCHSHIFT = "mistmatchshift"  # shift performance of domain mismatch
# not used currently -- perturb and precision experiments
# still figuring out how to best incorporate these
PERTURB = "perturbation"
PRECISION = "precision"


def naming_convention(instruction_type: str, instruction_encoding: str, experiment: str, num: Union[int, str]):
    """Defines the naming convention used for storage and evaluation

    :param instruction_encoding: The way the instruction is encoded into natural language, i.e., template, synthetic
    :param instruction_type: The type of instruction, e.g., rulesets
    :param experiment: The name of the experiment, e.g., performance, mismatch, robustness
    :param num: The index of the experiment, if multiple trials
    :return: Experiment name.
    """
    return f"{instruction_type}-{instruction_encoding}-{experiment}-{num}"
