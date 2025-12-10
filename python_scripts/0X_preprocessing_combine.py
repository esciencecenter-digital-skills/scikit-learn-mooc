# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Using numerical and categorical features together
#
# > Note: this is a shortened version of `03_categorical_pipeline_column_transformer.py`
#
# In the previous notebooks, we showed the required preprocessing to apply when
# dealing with numerical and categorical variables. However, we decoupled the
# process to treat each type individually. In this notebook, we show how to
# combine these preprocessing steps.
#
# We first load the entire adult census dataset.

# %%
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")
# drop the duplicated column `"education-num"` as stated in the first notebook
adult_census = adult_census.drop(columns="education-num")

target_name = "class"
target = adult_census[target_name]

data = adult_census.drop(columns=[target_name])

# %% [markdown]
# ## Selection based on data types
#
# We separate categorical and numerical variables using their data types to
# identify them, as we saw previously that `object` corresponds to categorical
# columns (strings). We make use of `make_column_selector` helper to select the
# corresponding columns.

# %%
from sklearn.compose import make_column_selector as selector

numerical_columns_selector = selector(dtype_exclude=object)
categorical_columns_selector = selector(dtype_include=object)

numerical_columns = numerical_columns_selector(data)
categorical_columns = categorical_columns_selector(data)

# %% [markdown]
# ```{caution}
# Here, we know that `object` data type is used to represent strings and thus
# categorical features. Be aware that this is not always the case. Sometimes
# `object` data type could contain other types of information, such as dates that
# were not properly formatted (strings) and yet relate to a quantity of
# elapsed  time.
#
# In a more general scenario you should manually introspect the content of your
# dataframe not to wrongly use `make_column_selector`.
# ```

# %% [markdown]
# ## Dispatch columns to a specific processor
#
# In the previous sections, we saw that we need to treat data differently
# depending on their nature (i.e. numerical or categorical).
#
# Scikit-learn provides a `ColumnTransformer` class which sends specific
# columns to a specific transformer, making it easy to fit a single predictive
# model on a dataset that combines both kinds of variables together
# (heterogeneously typed tabular data).
#
# We first define the columns depending on their data type:
#
# * **one-hot encoding** is applied to categorical columns. Besides, we use
#   `handle_unknown="ignore"` to solve the potential issues due to rare
#   categories.
# * **numerical scaling** numerical features which will be standardized.
#
# Now, we create our `ColumnTransfomer` using the helper function
# `make_column_transformer`. We specify two values: the transformer, and the
# columns. First, let's create the preprocessors for the numerical and
# categorical parts.

# %%
from sklearn.preprocessing import OneHotEncoder, StandardScaler

categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
numerical_preprocessor = StandardScaler()

# %% [markdown]
# Now, we create the transformer and associate each of these preprocessors with
# their respective columns.

# %%
from sklearn.compose import make_column_transformer

preprocessor = make_column_transformer(
    (categorical_preprocessor, categorical_columns),
    (numerical_preprocessor, numerical_columns),
)

# %% [markdown]
# We can take a minute to represent graphically the structure of a
# `ColumnTransformer`:
#
# ![columntransformer diagram](../figures/api_diagram-columntransformer.svg)
#
# A `ColumnTransformer` does the following:
#
# * It **splits the columns** of the original dataset based on the column names
#   or indices provided. We obtain as many subsets as the number of transformers
#   passed into the `ColumnTransformer`.
# * It **transforms each subsets**. A specific transformer is applied to each
#   subset: it internally calls `fit_transform` or `transform`. The output of
#   this step is a set of transformed datasets.
# * It then **concatenates the transformed datasets** into a single dataset.

# The important thing is that `ColumnTransformer` is like any other scikit-learn
# transformer. In particular it can be combined with a classifier in a
# `Pipeline`:

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(preprocessor, LogisticRegression(max_iter=500))
model

# %% [markdown]
# The final model is more complex than the previous models but still follows the
# same API (the same set of methods that can be called by the user):
#
# - the `fit` method is called to preprocess the data and then train the
#   classifier of the preprocessed data;
# - the `predict` method makes predictions on new data;
# - the `score` method is used to predict on the test data and compare the
#   predictions to the expected test labels to compute the accuracy.
#
# Let's start by splitting our data into train and test sets.

# %%
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=42
)

# %% [markdown]
#
# ```{caution}
# Be aware that we use `train_test_split` here for didactic purposes, to show
# the scikit-learn API. In a real setting one might prefer to use
# cross-validation to also be able to evaluate the uncertainty of our estimation
# of the generalization performance of a model, as previously demonstrated.
# ```
#
# Now, we can train the model on the train set.

# %%
_ = model.fit(data_train, target_train)

# %% [markdown]
# Then, we can send the raw dataset straight to the pipeline. Indeed, we do not
# need to make any manual preprocessing (calling the `transform` or
# `fit_transform` methods) as it is already handled when calling the `predict`
# method.

# %% [markdown]
# We can call the `score` method to compute the accuracy score on the test set.

# %%
model.score(data_test, target_test)
