"""Creates a new TABLET task."""
import copy
import json
import logging
import os
from functools import partial
from typing import Tuple, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from transformers import trainer_utils

from Tablet import prototypes, ruleset
from Tablet import util
from Tablet.synthetic_language import get_gpt3_revisions, create_revision_instructions

DEFAULT_BENCHMARK_NAME = "performance"


def create_task(train_df: pd.DataFrame,
                test_df: pd.DataFrame,
                train_y: np.ndarray,
                test_y: np.ndarray,
                name: str,
                header: str,
                categorical_columns: list[str],
                save_loc: str = "./benchmark",
                nl_instruction: str = None,
                alternate_header: str = "",
                num_gpt3_revisions: int = 20,
                num_templates_per_class: int = 1,
                experiment_name: str = util.PERF,
                num: int = 0,
                temp_y_col_name: str = "y_temp",
                gpt3_model: str = "text-davinci-003",
                one_v_all_style: str = None,
                benchmark_folder_name: str = DEFAULT_BENCHMARK_NAME,
                openai_key_path: str = "oai-key.txt",
                only_natural_language: bool = False,
                serialization_starting_text: str = "",
                seed: int = 0,
                max_features: int = 20) -> None:
    """Top level task creation.

    :param serialization_starting_text: Text which well be prepended to instance serializations.
    :param max_features: The max number of features to include in the prototypes generated instructions.
    :param one_v_all_style: If not None, will encode the task using a one v all style, with the value of str
                            as the "one."
    :param only_natural_language: Only generate the natural langauge instructions.
    :param openai_key_path: The path to the openai key.
    :param gpt3_model: The name of the gpt3 model used to create synthetic revisions.
    :param experiment_name: The name of the experiment.
    :param num: Task number to be added for performance evals.
    :param benchmark_folder_name: The name of the top level folder created for the benchmark. This will be created if
                                  not already there.
    :param temp_y_col_name: Temporary name of y column.
    :param num_gpt3_revisions: The number of gpt3 revisions of each template instruction to be created.
    :param seed: The seed.
    :param train_df: The training data frame.
    :param test_df: The testing data frame.
    :param train_y: The training y values.
    :param test_y: The testing y values.
    :param name: The name of the dataset.
    :param header: The header for the task.
    :param categorical_columns: The names of the categorical features in the dataframe.
    :param save_loc: The folder to store the task.
    :param nl_instruction: The natural language instructions.
    :param alternate_header: If specified, will create task without instruction
                             (ie., lift style) and this will be the header.
    :param num_templates_per_class: The number of template instructions per task to create.
    :return: None
    """
    trainer_utils.set_seed(seed)
    # Set up directory that will store results
    dataset_dir = setup_storage(save_loc, name, benchmark_folder_name)
    # Label encode y
    encoded_train_y, encoded_test_y, semantic_outcome_map = label_encode_y(train_y,
                                                                           test_y)
    class_text = get_class_text(train_y)
    write_tab_func = partial(write_csv_data,
                             temp_y_col_name=temp_y_col_name,
                             test_df=test_df,
                             train_df=train_df,
                             train_y=train_y,
                             test_y=test_y)
    # Get training & testing serializations
    test_serializations, train_serializations = get_data_serializations("{datapoint}", semantic_outcome_map, test_df,
                                                                        encoded_test_y, train_df, encoded_train_y,
                                                                        one_v_all_style, serialization_starting_text)
    if not only_natural_language:
        # 1) Learn the prototype template instructions
        prototype_instruction = write_prototypes_task(alternate_header, categorical_columns, class_text, dataset_dir,
                                                      encoded_train_y, header,
                                                      num_templates_per_class, semantic_outcome_map, test_serializations,
                                                      test_y, train_df,
                                                      train_serializations, train_y, write_tab_func, num,
                                                      experiment_name, max_features)
        logging.info(f"Done writing prototypes task for {name}.")

        # 2) Learn the rule-set template instruction
        ruleset_instruction = write_ruleset_task(alternate_header, categorical_columns, class_text, dataset_dir,
                                                 encoded_train_y, header, 1, semantic_outcome_map,
                                                 test_serializations, test_y, train_df, train_serializations, train_y,
                                                 write_tab_func, num, experiment_name)
        logging.info(f"Done writing ruleset task for {name}.")

        # 3) Revised prototype templates, computes for both rule set and natural
        synthetic_instructions(alternate_header, class_text, dataset_dir, experiment_name, header, name,
                               num_gpt3_revisions, ruleset_instruction, prototype_instruction, test_serializations,
                               test_y, train_serializations, train_y, write_tab_func, gpt3_model, openai_key_path)

    # 4) Natural language templates (if it exists)
    if nl_instruction is not None:
        create_natural_language(alternate_header, class_text, dataset_dir, experiment_name, header,
                                nl_instruction, test_serializations, test_y, train_serializations, train_y,
                                write_tab_func, num)

    # Done!
    logging.info(f"Done writing all tasks for {name}.")


def create_natural_language(alternate_header, class_text, dataset_dir, experiment_name, header, nl_instruction,
                            test_serializations, test_y, train_serializations, train_y, write_tab_func, num):
    # By default, natural language is stored until prototypes instruction name, though it does not have an instruction
    # type
    cur_name = util.naming_convention(util.PROTOTYPES,
                                      util.NATLANGUAGE,
                                      experiment_name,
                                      num)
    save_task(cur_name,
              dataset_dir,
              header,
              alternate_header,
              class_text,
              nl_instruction,
              train_serializations,
              train_y,
              test_serializations,
              test_y,
              write_tab_func)


def synthetic_instructions(alternate_header, class_text, dataset_dir, experiment_name, header, name, num_gpt3_revisions,
                           ruleset_instruction, prototype_instruction, test_serializations, test_y,
                           train_serializations, train_y,
                           write_tab_func, model, openai_key_path):
    logging.info("Creating synthetic instructions...")
    for cur_instruction, instruction_type in tqdm(zip([ruleset_instruction, prototype_instruction],
                                                      [util.RULESET, util.PROTOTYPES])):
        for cur_revision in tqdm(range(num_gpt3_revisions)):
            name = util.naming_convention(instruction_type,
                                          util.SYNTHETIC,
                                          experiment_name,
                                          cur_revision)
            revision_instructions = create_revision_instructions(cur_instruction, instruction_type, header)
            current_instructions = get_gpt3_revisions(revision_instructions,
                                                      model=model,
                                                      n=1,
                                                      key_path=openai_key_path)
            save_task(name,
                      dataset_dir,
                      header,
                      alternate_header,
                      class_text,
                      current_instructions[0],
                      train_serializations,
                      train_y,
                      test_serializations,
                      test_y,
                      write_tab_func)
    logging.info("Done writing synthetic instructions...")


def write_csv_data(dataset_dir, temp_y_col_name, test_df, train_df, train_y, test_y):
    if temp_y_col_name in list(train_df.columns):
        raise ValueError(f"temp_y_col_name {temp_y_col_name} already in columns. choose another.")
    train_df_y = copy.deepcopy(train_df)
    test_df_y = copy.deepcopy(test_df)
    train_df_y[temp_y_col_name] = train_y
    test_df_y[temp_y_col_name] = test_y
    train_df_y.to_csv(os.path.join(dataset_dir, "train.csv"))
    test_df_y.to_csv(os.path.join(dataset_dir, "test.csv"))


def write_ruleset_task(alternate_header, categorical_columns, class_text, dataset_dir, encoded_train_y, header,
                       num_templates_per_class, semantic_outcome_map, test_serializations, test_y, train_df,
                       train_serializations, train_y, write_tab_func, num, experiment_name):
    """Writes the rule set template task"""
    ruleset_instructions = create_rule_set_instructions(train_df,
                                                        encoded_train_y,
                                                        categorical_columns,
                                                        semantic_outcome_map,
                                                        num_templates_per_class)
    name = util.naming_convention(util.RULESET, util.TEMPLATES, experiment_name, num)
    save_task(name,
              dataset_dir,
              header,
              alternate_header,
              class_text,
              ruleset_instructions,
              train_serializations,
              train_y,
              test_serializations,
              test_y,
              write_tab_func)
    return ruleset_instructions


def write_prototypes_task(alternate_header, categorical_columns, class_text, dataset_dir, encoded_train_y, header,
                          num_templates_per_class, semantic_outcome_map, test_serializations, test_y, train_df,
                          train_serializations, train_y, write_tab_func, num, experiment_name, max_features: int = 20):
    """Writes the prototypes template tasks"""
    prototypes_instructions = create_prototypes_instructions(train_df,
                                                             encoded_train_y,
                                                             categorical_columns,
                                                             semantic_outcome_map,
                                                             num_templates_per_class,
                                                             max_features)
    name = util.naming_convention(util.PROTOTYPES, util.TEMPLATES, experiment_name, num)
    save_task(name,
              dataset_dir,
              header,
              alternate_header,
              class_text,
              prototypes_instructions,
              train_serializations,
              train_y,
              test_serializations,
              test_y,
              write_tab_func)
    return prototypes_instructions


def setup_storage(benchmark_folder: str, name: str, benchmark_name: str = "tab-instruct-benchmark"):
    """Setup folders containing tasks."""
    benchmark_dir = os.path.join(benchmark_folder, benchmark_name)
    dataset_dir = os.path.join(benchmark_dir, name)
    os.makedirs(dataset_dir, exist_ok=True)
    return dataset_dir


def get_data_serializations(class_text,
                            semantic_outcome_map,
                            test_df,
                            test_y,
                            train_df,
                            train_y,
                            one_v_all,
                            serialization_starting_text):
    """Gets data serializations

    :param serialization_starting_text:
    :param one_v_all:
    :param class_text:
    :param semantic_outcome_map:
    :param test_df:
    :param test_y:
    :param train_df:
    :param train_y:
    :return:
    """
    if len(test_df) != len(test_y):
        raise ValueError(f"len train_df df != len train_y")
    if len(test_df) != len(test_y):
        raise ValueError(f"len Testing df != len testing y")
    train_serializations = prototypes.get_descriptions_for_rows(train_df,
                                                                train_y,
                                                                class_text,
                                                                list(train_df.columns),
                                                                add_label=False,
                                                                one_v_all=one_v_all,
                                                                serialization_starting_text=serialization_starting_text,
                                                                semantic_outcome_map=semantic_outcome_map)
    test_serializations = prototypes.get_descriptions_for_rows(test_df,
                                                               test_y,
                                                               class_text,
                                                               list(train_df.columns),
                                                               add_label=False,
                                                               one_v_all=one_v_all,
                                                               serialization_starting_text=serialization_starting_text,
                                                               semantic_outcome_map=semantic_outcome_map)
    return test_serializations, train_serializations


def get_class_text(classes: np.ndarray):
    unique_classes = [str(v) for v in np.unique(classes)]
    class_values = " | ".join(unique_classes)
    return f"Answer with one of the following: {class_values}."


def convert_save(save_dict: dict):
    final_dicts = []
    for i in range(len(save_dict["header"])):
        final_dicts.append({v: save_dict[v][i] for v in save_dict.keys()})
    return final_dicts


def write_file(base: str, fname: str, data: str):
    with open(os.path.join(base, fname), "w") as file:
        file.write(data)


def save_task(task_name: str,
              base_dir: str,
              header: str,
              lift_header: str,
              class_text: str,
              instructions: str,
              train_serializations: list[str],
              train_labels: np.ndarray,
              test_serializations: list[str],
              test_labels: np.ndarray,
              write_tab_func: Any):
    """Saves task."""
    save_dir = os.path.join(base_dir, task_name)
    os.makedirs(save_dir, exist_ok=True)
    # this writes the tabular data to the task folder
    write_tab_func(save_dir)
    nt = len(train_serializations)
    nte = len(test_serializations)
    train_save = {
        "header": [header] * nt,
        "lift_header": [lift_header] * nt,
        "class_text": [class_text] * nt,
        "instructions": [instructions] * nt,
        "label": [str(v) for v in train_labels],
        "serialization": train_serializations
    }
    test_save = {
        "header": [header] * nte,
        "lift_header": [lift_header] * nte,
        "class_text": [class_text] * nte,
        "instructions": [instructions] * nte,
        "label": [str(v) for v in test_labels],
        "serialization": test_serializations
    }
    train_save = json.dumps(convert_save(train_save), indent=4)
    test_save = json.dumps(convert_save(test_save), indent=4)
    write_file(save_dir, "train.json", train_save)
    write_file(save_dir, "test.json", test_save)


def label_encode_y(train_y: np.ndarray,
                   test_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Label encode train and test ys.

    :param train_y: Training df.
    :param test_y: Testing df.
    :return: label_encoder: The sklearn label encoder used to transform the data.
    """
    label_encoder = LabelEncoder()
    train_y = label_encoder.fit_transform(train_y)
    test_y = label_encoder.transform(test_y)
    semantic_outcome_map = {v: label_encoder.inverse_transform(v.reshape(-1))[0] for v in np.unique(train_y)}
    return train_y, test_y, semantic_outcome_map


def create_prototypes_instructions(df: pd.DataFrame,
                                   y: np.ndarray,
                                   categorical_cols: list[str],
                                   semantic_outcome_map: dict,
                                   n_clusters: int,
                                   max_features: int = 20) -> str:
    results = prototypes.get_kmeans_rules(df,
                                          y,
                                          categorical_cols,
                                          semantic_outcome_map,
                                          max_features=max_features,
                                          n_clusters=n_clusters)
    instructions = results[0]
    return instructions


def create_rule_set_instructions(df: pd.DataFrame,
                                 y: np.ndarray,
                                 categorical_cols: list[str],
                                 semantic_outcome_map: dict,
                                 n_rules: int) -> str:
    results = ruleset.get_decision_tree_rules(df, y, categorical_cols, semantic_outcome_map, n_rules=n_rules)
    instructions = results[0]
    return instructions
