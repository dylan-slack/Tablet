"""Prototypes template instructions."""
from copy import deepcopy
import pandas as pd
import numpy as np
import logging

from kmodes import kmodes
from kmodes.kprototypes import KPrototypes
from kmodes import kprototypes
from kmodes.kmodes import KModes
from sklearn.cluster import KMeans
from kmodes.util import encode_features
from xgboost import XGBClassifier


class ProtoWrapper:

    def __init__(self, n_clusters=1, max_features=1):
        self.largest_keys = None
        self.max_features = max_features
        self.n_clusters = n_clusters
        self.model = None
        self.cat_cols = []
        self.cat_cols_inds = None

    def fit(self, train_x, train_y):
        self.cat_cols = list(train_x.columns)
        kmeans = get_kmeans_rules(train_x, train_y, max_features=self.max_features,
                                  categorical_columns=self.cat_cols, n_clusters=self.n_clusters)
        self.model = kmeans[2]['model']
        self.largest_keys = kmeans[2]['largest_keys']
        self.cat_cols_inds = [list(train_x.columns).index(c) for c in self.cat_cols if c in train_x.columns]

    def predict(self, test_x):
        test_x = test_x[self.largest_keys]
        kp = kmeans_preds(self.model,
                          test_x.to_numpy(),
                          test_x.columns,
                          [0, 1])
        return kp


def perturb_k_modes_centers(dataset_no_classes, centers, sigma, cat_cols_inds):
    new_centers = []
    cols = list(dataset_no_classes.columns)
    for c in centers:
        nc = []
        for i in range(len(c)):
            if i in cat_cols_inds:
                if np.random.choice([True, False], p=[sigma, 1 - sigma]):
                    nv = np.random.choice(dataset_no_classes[cols[i]])
                else:
                    nv = c[i]
            else:
                nv = float(c[i]) + np.random.normal(0, sigma * np.std(dataset_no_classes[cols[i]]))
            nc.append(nv)
        new_centers.append(nc)
    return np.array(new_centers)


def get_kmeans_rules(dataset: pd.DataFrame,
                     classes: np.ndarray,
                     categorical_columns: list,
                     semantic_outcome_map: dict = None,
                     k_modes: KMeans = None,
                     sigma_perturbation: float = 0.0,
                     max_features: int = 20,
                     **kwargs):
    if semantic_outcome_map is None:
        semantic_outcome_map = {c: c for c in np.unique(classes)}
    largest_keys = list(dataset.columns)
    if len(dataset.columns) > max_features:
        dataset[dataset.select_dtypes(['object']).columns] = \
            dataset.select_dtypes(['object']).apply(lambda x: x.astype('category'))
        dataset[dataset.select_dtypes(['bool']).columns] = \
            dataset.select_dtypes(['bool']).apply(lambda x: x.astype('category'))
        logging.info("Filtering down features for prototypes rules...")
        clf = XGBClassifier(tree_method="hist", enable_categorical=True)
        clf.fit(dataset, classes)
        importances = clf.get_booster().get_score(importance_type='gain')
        largest_keys = sorted(importances, key=importances.get, reverse=True)[:max_features]
        dataset = dataset[largest_keys]
    cat_cols_inds = [list(dataset.columns).index(c) for c in categorical_columns if c in list(dataset.columns)]
    text = ""
    if k_modes is None:
        classifier_store = {}
    else:
        classifier_store = k_modes
    to_use_classes = classes
    if k_modes is None and len(np.unique(classes)) > 10:
        logging.info("down sampling classes...")
        to_use_classes = np.random.choice(np.unique(classes), size=10, replace=False)
        logging.info("using: " + str(to_use_classes))
    for cur_class in np.unique(to_use_classes):
        cur_inds = classes == cur_class
        cur_dataset_no_classes = dataset.to_numpy()[cur_inds]
        if len(cur_dataset_no_classes) <= 2:
            print(f"Shortening clusters...")
            use_n_clusters = len(cur_dataset_no_classes)
        else:
            use_n_clusters = None
        if not cat_cols_inds:
            logging.info("Using k-means...")
            if k_modes is None:
                if use_n_clusters is None:
                    classifier = KMeans(**kwargs).fit(cur_dataset_no_classes)
                else:
                    classifier = KMeans(n_clusters=use_n_clusters).fit(cur_dataset_no_classes)
                classifier_store[cur_class] = classifier
            else:
                if cur_class not in classifier_store:
                    continue
                classifier = classifier_store[cur_class]
            centers = classifier.cluster_centers_
            cat_cols_inds = []
        elif len(cat_cols_inds) == len(list(dataset.columns)):
            logging.info("Using k-modes...")
            if k_modes is None:
                if use_n_clusters is None:
                    classifier = KModes(verbose=True, n_jobs=-1, **kwargs).fit(cur_dataset_no_classes)
                else:
                    classifier = KModes(verbose=True, n_jobs=-1, n_clusters=use_n_clusters).fit(cur_dataset_no_classes)
                classifier_store[cur_class] = classifier
            else:
                if cur_class not in classifier_store:
                    continue
                classifier = classifier_store[cur_class]
            centers = classifier.cluster_centroids_
        else:
            logging.info("Using kprototypes...")
            if k_modes is None:
                if use_n_clusters is None:
                    classifier = KPrototypes(verbose=True, n_jobs=-1, **kwargs).fit(cur_dataset_no_classes,
                                                                                    categorical=cat_cols_inds)
                else:
                    classifier = KPrototypes(verbose=True,
                                             n_jobs=-1,
                                             n_clusters=use_n_clusters).fit(cur_dataset_no_classes,
                                                                            categorical=cat_cols_inds)
                classifier_store[cur_class] = classifier
            else:
                if cur_class not in classifier_store:
                    continue
                classifier = classifier_store[cur_class]
            centers = classifier.cluster_centroids_
        num_cols_inds = [i for i in range(len(dataset.columns)) if i not in cat_cols_inds]
        if len(cat_cols_inds) == 0:
            centers_num = centers
            centers_cat = []
        elif len(cat_cols_inds) == len(dataset.columns):
            centers_num = []
            centers_cat = centers
        else:
            centers_num = deepcopy(centers[:, :-len(cat_cols_inds)])
            centers_cat = deepcopy(centers[:, -len(cat_cols_inds):])
        centers[:, cat_cols_inds] = centers_cat
        centers[:, num_cols_inds] = centers_num
        if sigma_perturbation > 0:
            centers = perturb_k_modes_centers(dataset,
                                              centers,
                                              sigma_perturbation,
                                              cat_cols_inds)
        for i, c in enumerate(centers):
            # ensure float type
            new_c = []
            for ci in c:
                try:
                    new_ci = ci
                    if isinstance(ci, str):
                        new_ci = float(ci)
                except:
                    new_ci = ci
                new_c.append(new_ci)
            c = new_c
            center_desc = row_to_description(c,
                                             list(dataset.columns),
                                             cur_class,
                                             "{datapoint}Then, the answer is",
                                             semantic_outcome_map=semantic_outcome_map)
            center_desc = "If the data point is most similar to: " + center_desc + "."
            text += center_desc + "\n"
    kp = kmeans_preds(classifier_store,
                      dataset.to_numpy(),
                      dataset.columns,
                      [dataset.columns[q] for q in cat_cols_inds])
    out = {"largest_keys": largest_keys, "model": classifier_store}
    return text, kp, out, None, 0


def myround(x, base=1):
    return base * round(x / base)


def join_list(items):
    # by chatgpt
    starter_text = ""
    if len(items) == 0:
        return ''
    elif len(items) == 1:
        return starter_text + str(items[0])
    elif len(items) == 2:
        return f"{starter_text}{items[0]} and {items[1]}"
    else:
        joined_items = ', '.join(items[:-1])
        return f"{starter_text}{joined_items}, and {items[-1]}"


def create_one_v_all_style(cur_row: np.ndarray, cur_col_names: list[str]):
    all_cols = {}
    for i, name in enumerate(cur_col_names):
        value = cur_row[i]
        if "Response" in name:
            name, value = name.split(" Response=")
            if value == "N":
                value = "no"
            if value == "Y":
                value = "yes"
        if value.isdigit():
            value = f"{value}/10"
        if "(L)" in value:
            s_value = value.split("(L)")[0]
            value = f"left {s_value}"
        if "(R)" in value:
            s_value = value.split("(R)")[0]
            value = f"right {s_value}"
        if name not in all_cols:
            all_cols[name] = [value]
        else:
            all_cols[name].append(value)
    cur_col_names = list(all_cols.keys())
    cur_row = np.array([join_list(all_cols[v]) for v in cur_col_names])
    return cur_row, cur_col_names


def get_descriptions_for_rows(dataset: pd.DataFrame,
                              classes: np.ndarray,
                              class_text: str,
                              column_names: list,
                              add_label: bool = True,
                              semantic_outcome_map: dict = None,
                              serialization_type: str = "list",
                              one_v_all: str = None,
                              serialization_starting_text: str = "",
                              rounding: int = 2):
    """Gets descriptions from each row."""
    descriptions = []
    if not isinstance(dataset, pd.DataFrame):
        raise ValueError(f"dataset should be a pd data frame but is {type(dataset)}")
    for c, t_row in enumerate(dataset.iterrows()):
        class_value = classes[c]
        row = t_row[1]
        if one_v_all is not None:
            # create one v all task
            cur_row = row.to_numpy()
            one_vs = cur_row == one_v_all
            cur_row = cur_row[one_vs]
            cur_col_names = np.array(column_names)[one_vs].tolist()
            cur_row, cur_col_names = create_one_v_all_style(cur_row, cur_col_names)
        else:
            cur_row = row.to_numpy()
            cur_col_names = column_names
        description = row_to_description(cur_row,
                                         cur_col_names,
                                         class_value,
                                         class_text,
                                         add_label=add_label,
                                         semantic_outcome_map=semantic_outcome_map,
                                         serialization_type=serialization_type,
                                         rounding=rounding)
        if serialization_starting_text != "":
            description = serialization_starting_text + "\n" + description
        descriptions.append(description)
    return descriptions


def row_to_description(row: np.ndarray,
                       column_names: list,
                       class_value: str,
                       class_text: str,
                       add_label: bool = True,
                       semantic_outcome_map: dict = None,
                       serialization_type: str = "list",
                       rounding: int = 2,
                       replace_under_score: bool = True):
    """Serializes row in a dataset to a dataset."""
    if len(row) != len(column_names):
        raise ValueError(f"Row length {len(row)} != column_names length {len(column_names)}")
    return_string = ""
    for i in range(len(column_names)):
        if serialization_type == "list":
            return_string += add_list_serialization(row[i], column_names[i], rounding,
                                                    replace_under_score=replace_under_score)
        elif serialization_type == "paragraph":
            return_string += add_paragraph_serialization(row[i], column_names[i], rounding)
        else:
            raise ValueError(f"Unknown serialization_type {serialization_type}.")
    if semantic_outcome_map is not None:
        semantic_class_text = semantic_outcome_map[class_value]
    else:
        semantic_class_text = class_value
    return_string = class_text.replace("{datapoint}", return_string)
    if add_label:
        if isinstance(semantic_class_text, str):
            semantic_class_text = semantic_class_text.replace("_", " ")
        return_string += f" {semantic_class_text}"
    return return_string


def do_rounding(value, rounding: int = 2):
    if isinstance(value, float):
        value = str(round(value, rounding))
        # remove extra .0 for ints
        if value.endswith(".0"):
            value = value[:-len(".0")]
    return value


def add_list_serialization(value: float, column_name: str, rounding: int = 2, replace_under_score: bool = True):
    """Adds a column name and value of the type 'list seralization'"""
    value = do_rounding(value, rounding)
    if replace_under_score:
        column_name = column_name.replace("_", " ")
        column_name = column_name.replace("-", " ")
        if isinstance(value, str):
            value = value.replace("_", " ")
    return f"{column_name}: {value}\n"


def add_paragraph_serialization(value: float, column_name: str, rounding: int = 2):
    """Adds a paragraph serialization."""
    value = do_rounding(value, rounding)
    return f" The {column_name} is {value}."


def kmeans_preds(classifier_store, eval_x_no_classes, columns, categorical_columns):
    preds = []
    for c in sorted(list(classifier_store.keys())):
        classifier = classifier_store[c]
        if isinstance(classifier, KMeans):
            cur_preds = []
            for p in eval_x_no_classes:
                score = classifier.score(p[None, :])
                cur_preds.append(score)
            preds.append(cur_preds)
        elif isinstance(classifier, KPrototypes):
            cur_preds = []
            cat_cols_inds = [list(columns).index(c) for c in categorical_columns]
            for p in eval_x_no_classes:
                cur_p = p[None, :]
                xnum, xcat = _split_num_cat(cur_p, cat_cols_inds)
                score = -1 * kprototypes.labels_cost(xnum, xcat, classifier._enc_cluster_centroids,
                                                     classifier.num_dissim, classifier.cat_dissim, classifier.gamma)[1]
                cur_preds.append(score)
            preds.append(cur_preds)
        elif isinstance(classifier, KModes):
            cur_preds = []
            for p in eval_x_no_classes:
                encoded_f = encode_features(p[None, :], enc_map=classifier._enc_map)[0]
                score = -1 * kmodes.labels_cost(encoded_f,
                                                classifier._enc_cluster_centroids,
                                                classifier.cat_dissim)[1]
                cur_preds.append(score)
            preds.append(cur_preds)
        else:
            raise ValueError("blah")
    predictions = np.argmax(np.array(preds), axis=0)
    return predictions


def _split_num_cat(X, categorical):
    """Extract numerical and categorical columns.
    Convert to numpy arrays, if needed.
    :param X: Feature matrix
    :param categorical: Indices of categorical columns
    """
    Xnum = np.asanyarray(X[:, [ii for ii in range(X.shape[1])
                               if ii not in categorical]]).astype(np.float64)
    Xcat = np.asanyarray(X[:, categorical])
    return Xnum, Xcat
