#
# AI Wonder Input Handler
#

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

# Constants
_NA = "<NA>"

# Transformers
class InputSelector(TransformerMixin):
    # Select or drop columns
    def __init__(self, target, select):
        self.target = target
        self.select = select

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        cols = list(X.columns)
        for col in cols:                        # Drop uninterested columns
            if col not in self.select:
                X = X.drop(col, axis=1)
        if self.target in X:                    # Drop target
            X = X.drop(self.target, axis=1)
        return X

class ColumnSorter(TransformerMixin):
    # Split columns into numerical and categorical columns
    def __init__(self, dtype):
        self.dtype = dtype
    
    def fit(self, X, y=None):
        if self.dtype == 'numerical':
            self.cols = list(X.select_dtypes(exclude='object').columns)
        elif self.dtype == 'categorical':
            self.cols = list(X.select_dtypes(include='object').columns)
            for col in self.cols:
                X[col] = X[col].astype(str)
        return self

    def transform(self, X, y=None):
        return X[self.cols]

class DoNotScale(TransformerMixin, BaseEstimator):
    def __init__(self, **kwargs):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X

    def inverse_transform(self, X, y=None):
        return X

class GenericScaler(TransformerMixin):
    # Run StandardScaler with empty dataframe in mind
    def __init__(self, scaler_=StandardScaler):
        self.scaler_ = scaler_
        self.scaler = None

    def fit(self, X, y=None):
        if self.scaler is None:
            self.scaler = self.scaler_()
        if len(X.columns) > 0:
            self.scaler.fit(X)
        return self

    def transform(self, X, y=None):
        if len(X.columns) > 0:
            return pd.DataFrame(self.scaler.transform(X), columns=X.columns)
        return X

    def inverse_transform(self, X, y=None):
        if len(X.columns) > 0:
            return pd.DataFrame(self.scaler.inverse_transform(X), columns=X.columns)
        return X

def transformers(state, encoder_=OrdinalEncoder, scaler_=StandardScaler):
    # Shorthands for pipeline makers
    def make_encoder():
        if encoder_==OneHotEncoder:
            return encoder_(sparse=False, handle_unknown='infrequent_if_exist')
        else:
            return encoder_(handle_unknown='use_encoded_value', unknown_value=-1)

    return Pipeline(steps=[
               ("input_selector", InputSelector(target=state.target, select=state.select)),
               ("preprocessing", FeatureUnion([
                   ("cat_pipe", Pipeline(steps=[
                       ("cat_selector", ColumnSorter(dtype='categorical')),
                       ("cat_encoder", make_encoder())
                   ])),
                   ("num_pipe", Pipeline(steps=[
                       ("num_selector", ColumnSorter(dtype='numerical')),
                       ("num_scaler", GenericScaler(scaler_)),
                       ]))
                   ])
               )
           ])

def rational_imputer(data):
    """
    Median for numeric columns,
    "<NA>" for categorical columns
    """

    data = data.copy()
    # numerical columns
    num_cols = data.select_dtypes(include=np.number).columns
    medians = data[num_cols].median()
    data[num_cols] = data[num_cols].fillna(medians)
    # categorical columns
    cat_cols = data.select_dtypes(exclude=np.number).columns
    data[cat_cols] = data[cat_cols].fillna("<NA>")
    # All set
    return data

def input_piped_model(state):
    # Build a pipeline composed up of input transformers and a trained model
    return Pipeline(steps=[
               ("input_transformers", state.transformers),
               ("trained_model", state.model)
           ])

def union_component(pipe, lev1, lev2):
    # Hack to access named components of a FeatureUnion
    return dict(pipe.named_steps['preprocessing'].transformer_list).get(lev1).named_steps[lev2]

def encoder_name(state):
    return type(state.cat_encoder).__name__

def scaler_name(state):
    return type(state.num_scaler.scaler).__name__

