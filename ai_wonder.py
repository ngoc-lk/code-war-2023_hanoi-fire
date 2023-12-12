# -*- coding: utf-8 -*-

#
# AI Wonder APIs for XAI-enabled apps
#

import pandas as pd
import numpy as np
import pickle
import re
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import euclidean_distances
from lime import lime_tabular
from input_handler import input_piped_model             # Just to export outside

# Loaders
def load_model(path):
    with open(path, 'rb') as f:   
        model = pickle.load(f)
    return model

def load_state(path):
    class DotDict(dict):
        def __getattr__(self, attr):
            return self.get(attr)

        def __setattr__(self, key, value):
            self[key] = value

    with open(path, 'rb') as f:   
        state = DotDict(pickle.load(f))
    return state

# Data handlers
def read_dataset(source, drop_unnamed_cols=True):
    def read_in_csv(file, compressed=False):
        read_args = {'skipinitialspace': True, 'low_memory': True}
        if compressed:
            read_args['compression'] = 'zip'
        return pd.read_csv(file, **read_args)

    def rename_columns(data,
            allowed_characters=r"A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ0-9_%#\(\)\+\-\.\?\!\<\>\="):
        old_cols = data.columns.tolist()
        new_cols = [re.sub(f'[^{allowed_characters}]', '_', col.strip()) for col in data.columns]
        renamed_ = False
        if old_cols != new_cols:
            data.columns = new_cols
            renamed_ = True
        return renamed_
    ###
    name = source if type(source) == str else source.name     # URL or UploadedFile
    if ".xlsx" in name:
        data = pd.read_excel(source)
    elif ".zip" in name:
        data = read_in_csv(source, compressed=True)
    else:
        data = read_in_csv(source)
    if drop_unnamed_cols:
        data = data.loc[:, ~data.columns.str.startswith('Unnamed: ')]
    return data, rename_columns(data)

# Transformers
def forward_transform(state, dataset, look_nums=False, look_cats=False):
    # Apply transformers (encoders, scalers) partially as needed

    if type(dataset) is not pd.DataFrame:
        dataset = pd.DataFrame(dataset, columns=state.cat_cols+state.num_cols)
    cat_cols = None
    num_cols = None

    if len(state.cat_cols) > 0:
        if look_cats:
            cat_cols = pd.DataFrame(state.cat_encoder.transform(dataset[state.gen_cols]),
                                    columns=state.cat_cols)
        else:
            cat_cols = dataset[state.cat_cols]
        cat_cols = cat_cols.reset_index(drop=True)

    if len(state.num_cols) > 0:
        if look_nums:
            num_cols = pd.DataFrame(state.num_scaler.transform(dataset[state.num_cols]),
                                    columns=state.num_cols)
        else:
            num_cols = dataset[state.num_cols]
        num_cols = num_cols.reset_index(drop=True)

    if cat_cols is None:
        forward = num_cols
    elif num_cols is None:
        forward = cat_cols
    else:
        forward = pd.concat([cat_cols, num_cols], axis=1)
    return forward

def inverse_transform(state, dataset, nums_only=False):
    if type(dataset) is not pd.DataFrame:
        dataset = pd.DataFrame(dataset, columns=state.cat_cols+state.num_cols)

    cat_cols = None
    num_cols = None

    if len(state.cat_cols) > 0:
        if not nums_only:
            cat_cols = pd.DataFrame(
                state.cat_encoder.inverse_transform(dataset[state.cat_cols]),
                columns=state.gen_cols
            )
        else:
            cat_cols = pd.DataFrame(dataset[state.cat_cols], columns=state.cat_cols)
        cat_cols = cat_cols.reset_index(drop=True)

    if len(state.num_cols) > 0:
        num_cols = pd.DataFrame(
            np.round(state.num_scaler.inverse_transform(dataset[state.num_cols]), 2),
            columns=state.num_cols
        )
        int_cols = dict(zip(state.num_cols, [num_type=='int64' for num_type in state.num_type]))
        for col in int_cols:
            if int_cols[col]:
                num_cols[col] = num_cols[col].astype(int)
        num_cols = num_cols.reset_index(drop=True)

    if cat_cols is None:
        inverse = num_cols
    elif num_cols is None:
        inverse = cat_cols
    else:
        inverse = pd.concat([cat_cols, num_cols], axis=1)
    return inverse

# Explainers
def local_explanations(state, orig, form="as_list"):
    # Format a LIME explanation
    def listify_explanation(exp):
        exp_map = exp.as_map()
        if state.task_type == "classification":
            explanation = exp_map[list(exp_map.keys())[0]]  # 1st and only
        elif state.task_type == "regression":
            explanation = exp_map[1]
        point_dict = orig.to_dict(orient='records')[0]
        importances = []
        feature_names = state.X_train.columns.tolist()
        for feature_id, importance in explanation:
            feature = feature_names[feature_id]
            importances.append((feature, point_dict[feature], round(importance, 2)))
        return importances

    # Here we go!
    X = state.X_train
    try:
        # Gather explanations
        categorical_names = dict(zip(range(len(state.cat_cols)), state.cat_vals))
        explainer = lime_tabular.LimeTabularExplainer(
            forward_transform(state, inverse_transform(state, X), look_cats=True).values,
            mode=state.task_type,
            feature_names=X.columns,
            categorical_features=list(range(len(state.cat_cols))),
            categorical_names=categorical_names
        )

        state.is_binary = False
        if state.task_type == "classification":
            explainer.class_names = state.model.classes_
            state.is_binary = len(explainer.class_names) == 2
            def predict_fn(x):
                transformed = forward_transform(state, x, look_nums=True)
                return state.model.model.estimator.predict_proba(transformed.values)
        elif state.task_type == "regression":
            def predict_fn(x):
                transformed = forward_transform(state, x, look_nums=True)
                return state.model.model.estimator.predict(transformed.values)

        # Format result as HTML
        datapoint = forward_transform(state, orig, look_cats=True).iloc[0]
        explanation = explainer.explain_instance(
            datapoint,
            predict_fn,
            num_features=20,
            top_labels=None if state.is_binary else 1) 
        if form == "as_list":
            return listify_explanation(explanation)
        return explanation
    except Exception as e:
        raise e

# Counterfactuals
def whatif_instances(state, point):
    def fix_target_type(data, target_type):
        # data = data.flatten()
        if target_type == 'float64':
            return data.astype(target_type)
        if target_type == 'int64':
            return np.round(data).astype(target_type)
        return data

    # Prep data
    point = state.transformers.transform(point)
    test_data = state.X_train.values

    preds = state.model.predict(test_data)
    ppred = state.model.predict(point.reshape(1,-1))[0]
    if state.task_type == "classification":
        cfs = test_data[preds != ppred]
    else:
        y_mean = state.y[state.target].mean()
        cfs = test_data[preds < y_mean] if ppred > y_mean else test_data[preds > y_mean]
    if len(cfs) == 0:
        return None
    
    # Sort them in the increasing order of distances
    ### Appropriately scale the data to have the distance not distorted by a few features
    scaler = StandardScaler()
    scaler.fit(test_data)
    distances = euclidean_distances(scaler.transform(point).reshape(1,-1), scaler.transform(cfs))
    nearest_indexes = np.argsort(distances)
    
    # Compose DataFrame of counterfactuals
    total_cfs = pd.DataFrame(cfs[nearest_indexes][0], columns=state.X_test.columns)
    preds_cfs = fix_target_type(state.model.predict(total_cfs.values), state.y[state.target].dtype)
    final_cfs = inverse_transform(state, total_cfs)
    final_cfs[f"{state.target}_"] = preds_cfs
    return final_cfs

