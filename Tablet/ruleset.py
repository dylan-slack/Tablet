"""Rule set template instructions."""
from copy import deepcopy
import pandas as pd
import numpy as np
import logging
import re

import imodels
import scipy.sparse
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn import preprocessing

from Tablet.prototypes import myround


def get_return_string(rules: dict, class_val: str):
    """Gets string from rules."""
    rs = []
    risk = "unknown"
    for r in rules:
        if "col" in r:
            cn = r["col"].split("__")
            cn_type, cn_name = cn[0], cn[1]
            if not cn_type.startswith("cat_cols"):
                if r["flip"]:
                    ineq = "less than"
                else:
                    ineq = "greater than"
                # round and strip trailing zeros
                cutoff_num = re.sub(r'\.0$', '', f"{r['cutoff']:.1f}")
                rs.append(f" {cn_name.replace('_', ' ')} is {ineq} {cutoff_num}")
            else:
                name_split = cn_name.split("_")
                category = name_split[0]
                category_val = "_".join(name_split[1:])
                if r["flip"]:
                    ineq = "not equal to"
                else:
                    ineq = "equal to"
                rs.append(f" {category.replace('_', ' ')} is {ineq} {category_val}")
        if "val" in r:
            risk = myround(100 - r["val"] * 100)
    final_s = "If" + " and ".join(rs)
    final_s += f" then the answer is {class_val} with {risk}% probability. "
    return final_s


def perturb_rules(data, rules, sigma):
    new_rules = []
    for r in rules:
        if "cutoff" not in r:
            continue
        col = r["col"]
        val = r["cutoff"]
        if col.startswith("num_cols"):
            real_name = col.split("__")[1]
            original_col_std = np.std(data[real_name])
            p_val = val + np.random.normal(0, sigma * original_col_std, 1)
            new_rule = deepcopy(r)
            new_rule["cutoff"] = round(p_val[0], 2)
        else:
            real_name = col.split("__")[1].split("_")[0]
            all_vals = data[real_name]
            new_rule = deepcopy(r)
            if np.random.choice([True, False], p=[sigma, 1 - sigma]):
                new_val = np.random.choice(all_vals.unique())
                new_rule["col"] = f"cat_cols__{real_name}_{new_val}"
        new_rules.append(new_rule)
    return new_rules


def get_rule_set(data: pd.DataFrame, classes: pd.DataFrame, class_val: int, categorical_columns: list[int],
                 semantic_outcome_map: dict, depth: int = 1, sigma_perturbation: float = 0.0):
    """Gets rule set for particular class value."""
    one_hot_classes = np.array([1 if cv == class_val else 0 for cv in classes])
    clf = imodels.GreedyRuleListClassifier(max_depth=depth)
    ohe = preprocessing.OneHotEncoder(handle_unknown="ignore")
    num_cols = [n for n in data.columns if n not in categorical_columns]
    col_transf = ColumnTransformer([("cat_cols", ohe, categorical_columns),
                                    ("num_cols", preprocessing.FunctionTransformer(feature_names_out="one-to-one"),
                                     num_cols)])
    oh_data = col_transf.fit_transform(data)
    rules = clf.fit(oh_data, one_hot_classes, feature_names=col_transf.get_feature_names_out())
    if sigma_perturbation > 0:
        logging.info(f"Using sigma {sigma_perturbation}")
        # experiments for robustness
        rules = perturb_rules(data, rules, sigma_perturbation)
    string_rules = get_return_string(rules, semantic_outcome_map[class_val])
    return string_rules, clf, col_transf


def encode_cols_as_float(data: pd.DataFrame, cat_cols: list[str]):
    """Encode cols as floats."""
    label_encoders = []
    for c in cat_cols:
        cur_col = data[c]
        le = preprocessing.LabelEncoder()
        data[c] = le.fit_transform(cur_col)
        label_encoders.append(le)
    return data, label_encoders


def get_decision_tree_rules(dataset: pd.DataFrame,
                            classes: np.ndarray,
                            categorical_columns: list,
                            semantic_outcome_map: dict = None,
                            decision_tree: DecisionTreeClassifier = None,
                            sigma_perturbation: float = 0.0,
                            n_rules: int = 1):
    """Gets the decision tree rules & returns rules + used features."""
    if semantic_outcome_map is None:
        semantic_outcome_map = {c: c for c in np.unique(classes)}

    if not decision_tree:
        decision_tree = []
        for c in np.unique(classes):
            cur_rule_set, clf, col_transf = get_rule_set(dataset,
                                                         classes,
                                                         c,
                                                         categorical_columns,
                                                         semantic_outcome_map,
                                                         depth=n_rules,
                                                         sigma_perturbation=sigma_perturbation)  # could add depth here
            decision_tree.append([cur_rule_set, clf, c, col_transf])
    all_rules = [cv[0] for cv in decision_tree]
    formatted_human_rules = "\n".join(all_rules)
    # get predictions
    decision_tree_predictions = rs_preds(dataset, decision_tree)
    f1 = f1_score(classes, decision_tree_predictions, average="macro")
    return formatted_human_rules, decision_tree_predictions, decision_tree, list(dataset.columns), f1


def rs_preds(dataset, decision_tree):
    pred_probs = []
    for i in range(len(decision_tree)):
        cur_dset = decision_tree[i][3].transform(dataset)
        if scipy.sparse.issparse(cur_dset):
            cur_dset = cur_dset.toarray()
        preds = decision_tree[i][1].predict_proba(cur_dset)[:, 1]
        pred_probs.append(preds)
    decision_tree_predictions = np.argmax(pred_probs, axis=0)
    return decision_tree_predictions


class RSWrapper:

    def __init__(self, max_depth: int = 1):
        self.max_depth = max_depth
        self.dt = None

    def fit(self, train_x, train_y):
        output = get_decision_tree_rules(train_x, train_y, [])
        self.dt = output[2]

    def predict(self, eval_x):
        return rs_preds(eval_x, self.dt)
