"""Code for completing naturally occurring instructions for the restrictively licensed sources.

For the medical references that have permissive licensing, I've provided the corresponding instructions in the dataset.
Because some medical references have quite restrictive licensing, we cannot host the instruction text,
from my understanding. So, I've provided the links to the instructions here and code that will complete these
datasets in the benchmark once ADD_MISSING_INSTRUCTIONS_HERE is filled out.

Note, the first two instructions for each link are the `Consumer` sources and the last instruction is the `Professional`
source.

In particular, to complete the instructions fill out the missing instructions in ADD_MISSING_INSTRUCTIONS_HERE (the
ones with empty strings "") by following the link in NATURALLY_OCCURRING_INSTRUCTION_LINKS and performing the following
collection procedure:
Our procedure for collecting the instructions was as follows:
    1) We navigated to the 'Symptoms' section on the webpage
    2) We copied the text in the 'Symptoms' section
    3) We prepended INTRO and appended OUTRO (see below) to the `Symptoms` information.

If you have any questions about this process, please contact dslack@uci.edu, and I will try to answer questions and
provide guidance. Also, if you know of an easier strategy for sharing data, please let me know, and I will incorporate
it into this dataset.
"""
import json
import os
import shutil

from tqdm import tqdm

INTRO = "\nSince there are a lot of different conditions that often share similar symptoms, a differential diagnosis " \
        "lists the possible conditions that could cause the symptoms. Here are instructions for the differential " \
        "diagnosis of {condition}:\n"

OUTRO = "\nIf the patient has similar symptoms, then the answer is yes, {condition} should be included in " \
        "the differential diagnosis. Otherwise, the answer is no."

# Missing instruction Worksheet
ADD_MISSING_INSTRUCTIONS_HERE = {
    "Bronchiectasis": [
        "complete",
        "",
        ""],
    "Chagas": [
        "complete",
        "",
        ""
    ],
    "Guillain-Barré syndrome": [
        "",
        "complete",
        ""
    ],
    "Viral pharyngitis": [
        "",
        "complete",
        ""
    ],
    "Whooping cough": [
        "",
        "",
        ""
    ],
    "Ebola": [
        "complete",
        "complete",
        ""
    ],
    "Pulmonary embolism": [
        "complete",
        "",
        ""
    ],
    "Epiglottitis": [
        "",
        "",
        ""
    ],
    "Boerhaave": [
        "",
        "",
        ""
    ],
    "Myocarditis": [
        "",
        "",
        ""
    ],
}

NATURALLY_OCCURRING_INSTRUCTION_LINKS = {
    "Bronchiectasis": ["https://www.nhlbi.nih.gov/health/bronchiectasis/symptoms",
                       "https://www.mayoclinic.org/diseases-conditions/bronchiolitis/symptoms-causes/syc-20351565?p=1",
                       "https://www.merckmanuals.com/professional/pulmonary-disorders/bronchiectasis-and-atelectasis/bronchiectasis?query=Bronchiectasis#v918847"],
    "Chagas": ["https://medlineplus.gov/chagasdisease.html",
               "https://medlineplus.gov/ency/article/001372.htm",
               "https://www.merckmanuals.com/professional/infectious-diseases/extraintestinal-protozoa/chagas-disease?query=Chagas#v1016143"],
    "Guillain-Barré syndrome": [
        "https://medlineplus.gov/ency/article/000684.htm#:~:text=This%20is%20called%20ascending%20paralysis,in%20the%20arms%20and%20legs",
        "https://www.nhs.uk/conditions/guillain-barre-syndrome/symptoms/",
        "https://www.merckmanuals.com/professional/neurologic-disorders/peripheral-nervous-system-and-motor-unit-disorders/guillain-barr%C3%A9-syndrome-gbs#:~:text=Flaccid%20weakness%20predominates%20in%20most,in%20the%20arms%20or%20head."],
    "Viral pharyngitis": ["https://medlineplus.gov/ency/article/001392.htm",
                          "https://www.nhs.uk/conditions/sore-throat/",
                          "https://www.merckmanuals.com/professional/ear,-nose,-and-throat-disorders/oral-and-pharyngeal-disorders/tonsillopharyngitis#v946859"],
    "Whooping cough": ["https://medlineplus.gov/whoopingcough.html",
                       "https://kidshealth.org/en/parents/whooping-cough.html",
                       "https://www.merckmanuals.com/professional/infectious-diseases/gram-negative-bacilli/pertussis?query=whooping%20cough#v1007163"],
    "Ebola": ["https://medlineplus.gov/ebola.html",
              "https://www.cdc.gov/vhf/ebola/symptoms/",
              "https://www.merckmanuals.com/professional/infectious-diseases/arboviruses,-arenaviridae,-and-filoviridae/marburg-and-ebola-virus-infections?query=ebola%20virus#v1021049"],
    "Pulmonary embolism": ["https://medlineplus.gov/pulmonaryembolism.html",
                           "https://medlineplus.gov/ency/article/000132.htm",
                           "https://www.merckmanuals.com/professional/pulmonary-disorders/pulmonary-embolism-pe/pulmonary-embolism-pe?query=pulmonary%20embolism"],
    "Epiglottitis": ["https://medlineplus.gov/ency/article/000605.htm",
                     "https://www.mayoclinic.org/diseases-conditions/epiglottitis/symptoms-causes/syc-20372227#:~:text=Symptoms%20might%20include%3A-,Sore%20throat.,breathing%20in%2C%20known%20as%20stridor.",
                     "https://www.merckmanuals.com/professional/ear,-nose,-and-throat-disorders/oral-and-pharyngeal-disorders/epiglottitis?query=epiglottitis"],
    "Boerhaave": ["https://medlineplus.gov/ency/article/000231.htm",
                  "https://my.clevelandclinic.org/health/diseases/22898-boerhaave-syndrome#symptoms-and-causes",
                  "https://www.merckmanuals.com/professional/gastrointestinal-disorders/esophageal-and-swallowing-disorders/esophageal-rupture?query=Boerhaave"],
    "Myocarditis": ["https://medlineplus.gov/ency/article/000231.htm",
                    "https://www.mayoclinic.org/diseases-conditions/myocarditis/symptoms-causes/syc-20352539?p=1",
                    "https://www.merckmanuals.com/professional/cardiovascular-disorders/myocarditis-and-pericarditis/myocarditis?query=Myocarditis#v35600481"]
}

# Dictionary indicating if the instruction is missing
INSTRUCTION_IS_MISSING = {
    "Bronchiectasis": [False, True, True],
    "Chagas": [False, True, True],
    "Guillain-Barré syndrome": [True, False, True],
    "Viral pharyngitis": [True, False, True],
    "Whooping cough": [True, True, True],
    "Ebola": [False, False, True],
    "Pulmonary embolism": [False, True, True],
    "Epiglottitis": [True, True, True],
    "Boerhaave": [True, True, True],
    "Myocarditis": [True, True, True]
}

# Licenses
LICENSES = {
    "Bronchiectasis": ["Public Domain", None, None],
    "Chagas": ["Public Domain", None, None],
    "Guillain-Barré syndrome": [None, "Open Government License v3.0", None],
    "Viral pharyngitis": [None, "Open Government License v3.0", None],
    "Whooping cough": [None, None, None],
    "Ebola": ["Public Domain", "Public Domain", None],
    "Pulmonary embolism": ["Public Domain", None, None],
    "Epiglottitis": [None, None, None],
    "Boerhaave": [None, None, None],
    "Myocarditis": [None, None, None]
}

# ICD Map, icd-10 -> disease
ICD_DICT = {
    'J02.9': 'Viral pharyngitis',
    'G61.0': 'Guillain-Barré syndrome',
    'J47': 'Bronchiectasis',
    'B57': 'Chagas',
    'A37': 'Whooping cough',
    'a98.4': 'Ebola',
    'i26': 'Pulmonary embolism',
    'J05.1': 'Epiglottitis',
    'K22.3': 'Boerhaave',
    'I51.4': 'Myocarditis',
}


def update_missing_instruction(instruction: str,
                               disease: str,
                               index: int,
                               benchmark: str = "./data/benchmark",
                               missing_instructions_benchmark: str = "./data/ddx_data_no_instructions/benchmark"):
    """Updates a dataset that has a missing instruction with the new instruction.

    :param: instruction: The new instruction string.
    :param: disease: The name of the disease.
    :param: index: the index of the instruction.
    :param: benchmark: The path to the benchmark.
    :param: missing_instructions_benchmark: The path to the missing instructions.
    """
    # Validate
    if not os.path.exists(
            os.path.join(missing_instructions_benchmark,
                         "A37")):
        raise ValueError("Data without instructions does not exist.")
    # Make sure benchmark path exists
    os.makedirs(
        os.path.join(benchmark, f"{disease}/prototypes-naturallanguage-performance-{index}"),
        exist_ok=True
    )
    # Copy csv's
    for split in ['train', 'test']:
        shutil.copy(
            os.path.join(missing_instructions_benchmark,
                         f"{disease}/prototypes-naturallanguage-performance-{index}/{split}.csv"),
            os.path.join(benchmark,
                         f"{disease}/prototypes-naturallanguage-performance-{index}/{split}.csv"),
        )
    # Add instructions
    train_json = f"{disease}/prototypes-naturallanguage-performance-{index}/train.json"
    test_json = f"{disease}/prototypes-naturallanguage-performance-{index}/test.json"
    write_task_with_instruction(benchmark, instruction, missing_instructions_benchmark, train_json, disease)
    write_task_with_instruction(benchmark, instruction, missing_instructions_benchmark, test_json, disease)


def write_task_with_instruction(benchmark: str,
                                instruction: str,
                                missing_instructions_benchmark: str,
                                json_file: str,
                                disease: str):
    """Writes the task to the benchmark with the new instruction."""
    with open(os.path.join(missing_instructions_benchmark, json_file), "r") as f:
        json_data = json.load(f)
        for item in json_data:
            item["instructions"] = INTRO.format(condition=ICD_DICT[disease]) + \
                                   instruction + \
                                   OUTRO.format(condition=ICD_DICT[disease])
        json_data = json.dumps(json_data, indent=4)
    new_path = os.path.join(benchmark, json_file)
    with open(new_path, "w") as f:
        f.write(json_data)


def write_completed_instructions(benchmark: str = "./data/benchmark/performance",
                                 missing_instructions_benchmark: str = "./data/ddx_data_no_instructions/benchmark"):
    """Call this function once the worksheet is finished to write the completed instruction set.

    :param: benchmark: The path to the benchmark.
    :param: missing_instructions_benchmark: The path to the benchmark with datasets that have incomplete instructions.
    """
    inv_icd = {v: k for k, v in zip(ICD_DICT.keys(), ICD_DICT.values())}
    for disease in tqdm(ADD_MISSING_INSTRUCTIONS_HERE.keys()):
        for i, instruction in enumerate(ADD_MISSING_INSTRUCTIONS_HERE[disease]):
            if instruction == "complete":
                continue
            update_missing_instruction(
                instruction,
                inv_icd[disease],
                index=i,
                benchmark=benchmark,
                missing_instructions_benchmark=missing_instructions_benchmark
            )


if __name__ == "__main__":
    FORCE = False
    if not FORCE:
        for v in ADD_MISSING_INSTRUCTIONS_HERE.values():
            if "" in v:
                raise ValueError("Some instructions are still incomplete!")
    write_completed_instructions()
