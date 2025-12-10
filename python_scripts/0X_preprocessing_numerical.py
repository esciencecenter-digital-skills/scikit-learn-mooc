# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Working with numerical variables
#
# note: taken from `numerical_pipeline_hands_on.py`
#
# In the previous notebook, we trained a k-nearest neighbors model on some data.
#
# However, we oversimplified the procedure by loading a dataset that contained
# exclusively numerical data. Besides, we used datasets which were already split
# into train-test sets.
#
# In this notebook, we aim at:
#
# * identifying numerical data in a heterogeneous dataset;
# * selecting the subset of columns corresponding to numerical data;
# * using a scikit-learn helper to separate data into train-test sets;
# * training and evaluating a more complex scikit-learn model.
#
# We start by loading the adult census dataset used during the data exploration.
#
# ## Loading the entire dataset
#
# As in the previous notebook, we rely on pandas to open the CSV file into a
# pandas dataframe.

# %%
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")
# drop the duplicated column `"education-num"` as stated in the first notebook
adult_census = adult_census.drop(columns="education-num")
adult_census

# %% [markdown]
# The next step separates the target from the data. We performed the same
# procedure in the previous notebook.

# %%
data, target = adult_census.drop(columns="class"), adult_census["class"]

# %%
data

# %%
target

# %% [markdown]
# ```{note}
# Here and later, we use the name `data` and `target` to be explicit. In
# scikit-learn documentation, `data` is commonly named `X` and `target` is
# commonly called `y`.
# ```

# %% [markdown]
# At this point, we can focus on the data we want to use to train our predictive
# model.
#
# ## Identify numerical data
#
# Numerical data are represented with numbers. They are linked to measurable
# (quantitative) data, such as age or the number of hours a person works a week.
#
# Predictive models are natively designed to work with numerical data. Moreover,
# numerical data usually requires very little work before getting started with
# training.
#
# The first task here is to identify numerical data in our dataset.
#
# ```{caution}
# Numerical data are represented with numbers, but numbers do not always
# represent numerical data. Categories could already be encoded with
# numbers and you may need to identify these features.
# ```
#
# Thus, we can check the data type for each of the column in the dataset.

# %%
data.dtypes

# %% [markdown]
# We seem to have only two data types: `int64` and `object`. We can make sure by
# checking for unique data types.

# %%
data.dtypes.unique()

# %% [markdown]
# Indeed, the only two types in the dataset are integer `int64` and `object`. We
# can look at the first few lines of the dataframe to understand the meaning of
# the `object` data type.

# %%
data

# %% [markdown]
# We see that the `object` data type corresponds to columns containing strings.
# As we saw in the exploration section, these columns contain categories and we
# will see later how to handle those. We can select the columns containing
# integers and check their content.

# %%
numerical_columns = ["age", "capital-gain", "capital-loss", "hours-per-week"]
data[numerical_columns]

# %% [markdown]
# Now that we limited the dataset to numerical columns only, we can analyse
# these numbers to figure out what they represent. We can identify two types of
# usage.
#
# The first column, `"age"`, is self-explanatory. We can note that the values
# are continuous, meaning they can take up any number in a given range. Let's
# find out what this range is:

# %%
data["age"].describe()

# %% [markdown]
# We can see the age varies between 17 and 90 years.
#
# We could extend our analysis and we would find that `"capital-gain"`,
# `"capital-loss"`, and `"hours-per-week"` are also representing quantitative
# data.
#
# Now, we store the subset of numerical columns in a new dataframe.

# %%
data_numeric = data[numerical_columns]

# %% [markdown]
# ## Train-test split the dataset
#
# In the previous notebook, we loaded two separate datasets: a training one and
# a testing one. However, having separate datasets in two distincts files is
# unusual: most of the time, we have a single file containing all the data that
# we need to split once loaded in the memory.
#
# Scikit-learn provides the helper function
# `sklearn.model_selection.train_test_split` which is used to automatically
# split the dataset into two subsets.

# %%
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    data_numeric, target, random_state=42, test_size=0.25
)

# %% [markdown]
# ```{tip}
# In scikit-learn setting the `random_state` parameter allows to get
# deterministic results when we use a random number generator. In the
# `train_test_split` case the randomness comes from shuffling the data, which
# decides how the dataset is split into a train and a test set).
# ```

# %% [markdown]
# When calling the function `train_test_split`, we specified that we would like
# to have 25% of samples in the testing set while the remaining samples (75%)
# are assigned to the training set. We can check quickly if we got what we
# expected.

# %%
print(
    f"Number of samples in testing: {data_test.shape[0]} => "
    f"{data_test.shape[0] / data_numeric.shape[0] * 100:.1f}% of the"
    " original set"
)

# %%
print(
    f"Number of samples in training: {data_train.shape[0]} => "
    f"{data_train.shape[0] / data_numeric.shape[0] * 100:.1f}% of the"
    " original set"
)

# %% [markdown]
# In the previous notebook, we used a k-nearest neighbors model. While this
# model is intuitive to understand, it is not widely used in practice. Now, we
# use a more useful model, called a logistic regression, which belongs to the
# linear models family.
#
# ```{note}
# To recap, linear models find a set of weights to combine features linearly
# and predict the target. For instance, the model can come up with a rule such
# as:
# * if `0.1 * age + 3.3 * hours-per-week - 15.1 > 0`, predict `high-income`
# * otherwise predict `low-income`
#
# ```
#
# To create a logistic regression model in scikit-learn you can do:

# %%
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

# %% [markdown]
# Now that the model has been created, you can use it exactly the same way as we
# used the k-nearest neighbors model in the previous notebook. In particular, we
# can use the `fit` method to train the model using the training data and
# labels:

# %%
model.fit(data_train, target_train)

# %% [markdown]
# We can also use the `score` method to check the model generalization
# performance on the test set.

# %%
accuracy = model.score(data_test, target_test)
print(f"Accuracy of logistic regression: {accuracy:.3f}")

# %% [markdown]
# ## Notebook recap
#
# In scikit-learn, the `score` method of a classification model returns the
# accuracy, i.e. the fraction of correctly classified samples. In this case,
# around 8 / 10 of the times the logistic regression predicts the right income
# of a person. Now the real question is: is this generalization performance
# relevant of a good predictive model? Find out by solving the next exercise!
#
# In this notebook, we learned to:
#
# * identify numerical data in a heterogeneous dataset;
# * select the subset of columns corresponding to numerical data;
# * use the scikit-learn `train_test_split` function to separate data into a
#   train and a test set;
# * train and evaluate a logistic regression model.


# %% [markdown]
# # Preprocessing for numerical features
#
# Note: taken from `numerical_pipeline_scaling.py`
#
# In this notebook, we still use numerical features only.
#
# Here we introduce these new aspects:
#
# * an example of preprocessing, namely **scaling numerical variables**;
# * using a scikit-learn **pipeline** to chain preprocessing and model training.
#
# ## Data preparation
#
# First, let's load the full adult census dataset.

# %%
import pandas as pd

adult_census = pd.read_csv("../datasets/adult-census.csv")

# %% [markdown]
# We now drop the target from the data we use to train our predictive model.

# %%
target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=target_name)

# %% [markdown]
# Then, we select only the numerical columns, as seen in the previous notebook.

# %%
numerical_columns = ["age", "capital-gain", "capital-loss", "hours-per-week"]

data_numeric = data[numerical_columns]

# %% [markdown]
# Finally, we can divide our dataset into a train and test sets.

# %%
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    data_numeric, target, random_state=42
)

# %% [markdown]
# ## Model fitting with preprocessing
#
# A range of preprocessing algorithms in scikit-learn allow us to transform the
# input data before training a model. In our case, we will standardize the data
# and then train a new logistic regression model on that new version of the
# dataset.
#
# Let's start by printing some statistics about the training data.

# %%
data_train.describe()

# %% [markdown]
# We see that the dataset's features span across different ranges. Some
# algorithms make some assumptions regarding the feature distributions and
# normalizing features is usually helpful to address such assumptions.
#
# ```{tip}
# Here are some reasons for scaling features:
#
# * Models that rely on the distance between a pair of samples, for instance
#   k-nearest neighbors, should be trained on normalized features to make each
#   feature contribute approximately equally to the distance computations.
#
# * Many models such as logistic regression use a numerical solver (based on
#   gradient descent) to find their optimal parameters. This solver converges
#   faster when the features are scaled, as it requires less steps (called
#   **iterations**) to reach the optimal solution.
# ```
#
# Whether or not a machine learning model requires scaling the features depends
# on the model family. Linear models such as logistic regression generally
# benefit from scaling the features while other models such as decision trees do
# not need such preprocessing (but would not suffer from it).
#
# We show how to apply such normalization using a scikit-learn transformer
# called `StandardScaler`. This transformer shifts and scales each feature
# individually so that they all have a 0-mean and a unit standard deviation.
# We recall that transformers are estimators that have a `transform` method.
#
# We now investigate different steps used in scikit-learn to achieve such a
# transformation of the data.
#
# First, one needs to call the method `fit` in order to learn the scaling from
# the data.

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(data_train)

# %% [markdown]
# The `fit` method for transformers is similar to the `fit` method for
# predictors. The main difference is that the former has a single argument (the
# data matrix), whereas the latter has two arguments (the data matrix and the
# target).
#
# ![Transformer fit diagram](../figures/api_diagram-transformer.fit.svg)
#
# In this case, the algorithm needs to compute the mean and standard deviation
# for each feature and store them into some NumPy arrays. Here, these statistics
# are the model states.
#
# ```{note}
# The fact that the model states of this scaler are arrays of means and standard
# deviations is specific to the `StandardScaler`. Other scikit-learn
# transformers may compute different statistics and store them as model states,
# in a similar fashion.
# ```
#
# We can inspect the computed means and standard deviations.

# %%
scaler.mean_

# %%
scaler.scale_

# %% [markdown]
# ```{note}
# scikit-learn convention: if an attribute is learned from the data, its name
# ends with an underscore (i.e. `_`), as in `mean_` and `scale_` for the
# `StandardScaler`.
# ```

# %% [markdown]
# Scaling the data is applied to each feature individually (i.e. each column in
# the data matrix). For each feature, we subtract its mean and divide by its
# standard deviation.
#
# Once we have called the `fit` method, we can perform data transformation by
# calling the method `transform`.

# %%
data_train_scaled = scaler.transform(data_train)
data_train_scaled

# %% [markdown]
# Let's illustrate the internal mechanism of the `transform` method and put it
# to perspective with what we already saw with predictors.
#
# ![Transformer transform
# diagram](../figures/api_diagram-transformer.transform.svg)
#
# The `transform` method for transformers is similar to the `predict` method for
# predictors. It uses a predefined function, called a **transformation
# function**, and uses the model states and the input data. However, instead of
# outputting predictions, the job of the `transform` method is to output a
# transformed version of the input data.

# %% [markdown]
# Finally, the method `fit_transform` is a shorthand method to call successively
# `fit` and then `transform`.
#
# ![Transformer fit_transform diagram](../figures/api_diagram-transformer.fit_transform.svg)
#
# In scikit-learn jargon, a **transformer** is defined as an estimator (an
# object with a `fit` method) supporting `transform` or `fit_transform`.

# %%
data_train_scaled = scaler.fit_transform(data_train)
data_train_scaled

# %% [markdown]
# By default, all scikit-learn transformers output NumPy arrays. Since
# scikit-learn 1.2, it is possible to set the output to be a pandas dataframe,
# which makes data exploration easier as it preserves the column names. The
# method `set_output` controls this behaviour. Please refer to this [example
# from the scikit-learn
# documentation](https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_set_output.html)
# for more options to configure the output of transformers.
# %%
scaler = StandardScaler().set_output(transform="pandas")
data_train_scaled = scaler.fit_transform(data_train)
data_train_scaled.describe()

# %% [markdown]
# Notice that the mean of all the columns is close to 0 and the standard
# deviation in all cases is close to 1. We can also visualize the effect of
# `StandardScaler` using a jointplot to show both the histograms of the
# distributions and a scatterplot of any pair of numerical features at the same
# time. We can observe that `StandardScaler` does not change the structure of
# the data itself but the axes get shifted and scaled.
#
# *Note to instructor*: We strongly advise to copy-paste below code and discuss the visualization with your audience.
# We discourage using live-coding here, because you will spend a lot of time fiddling with the jointplot, time which
# we can better use to introduce machine learning.
#

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# number of points to visualize to have a clearer plot
num_points_to_plot = 300

sns.jointplot(
    data=data_train[:num_points_to_plot],
    x="age",
    y="hours-per-week",
    marginal_kws=dict(bins=15),
)
plt.suptitle(
    "Jointplot of 'age' vs 'hours-per-week' \nbefore StandardScaler", y=1.1
)

sns.jointplot(
    data=data_train_scaled[:num_points_to_plot],
    x="age",
    y="hours-per-week",
    marginal_kws=dict(bins=15),
)
_ = plt.suptitle(
    "Jointplot of 'age' vs 'hours-per-week' \nafter StandardScaler", y=1.1
)

# %% [markdown]
# We can easily combine sequential operations with a scikit-learn `Pipeline`,
# which chains together operations and is used as any other classifier or
# regressor. The helper function `make_pipeline` creates a `Pipeline`: it
# takes as arguments the successive transformations to perform, followed by the
# classifier or regressor model.

# %%
import time
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

model = make_pipeline(StandardScaler(), LogisticRegression())
model

# %% [markdown]
# The `make_pipeline` function did not require us to give a name to each step.
# Indeed, it was automatically assigned based on the name of the classes
# provided; a `StandardScaler` step is named `"standardscaler"` in the resulting
# pipeline. We can check the name of each steps of our model:

# %%
model.named_steps

# %% [markdown]
# This predictive pipeline exposes the same methods as the final predictor:
# `fit` and `predict` (and additionally `predict_proba`, `decision_function`, or
# `score`).

# %%
model.fit(data_train, target_train)

# %% [markdown]
# We can represent the internal mechanism of a pipeline when calling `fit` by
# the following diagram:
#
# ![pipeline fit diagram](../figures/api_diagram-pipeline.fit.svg)
#
# When calling `model.fit`, the method `fit_transform` from each underlying
# transformer (here a single transformer) in the pipeline is called to:
#
# - learn their internal model states
# - transform the training data. Finally, the preprocessed data are provided to
#   train the predictor.
#
# To predict the targets given a test set, one uses the `predict` method.

# %%
predicted_target = model.predict(data_test)
predicted_target[:5]

# %% [markdown]
# Let's show the underlying mechanism:
#
# ![pipeline predict diagram](../figures/api_diagram-pipeline.predict.svg)
#
# The method `transform` of each transformer (here a single transformer) is
# called to preprocess the data. Note that there is no need to call the `fit`
# method for these transformers because we are using the internal model states
# computed when calling `model.fit`. The preprocessed data is then provided to
# the predictor that outputs the predicted target by calling its method
# `predict`.
#
# As a shorthand, we can check the score of the full predictive pipeline calling
# the method `model.score`. Thus, let's check the
# generalization performance of such a predictive pipeline.

# %%
model_name = model.__class__.__name__
score = model.score(data_test, target_test)
print(f"The accuracy using a {model_name} is {score:.3f} ")

# %% [markdown]
# We can compare the pipeline using scaling and logistic regression with
# using a logistic regression model without scaling like we did before.
# We will not go into how to do this comparison with Python in sake of time,
# instead we directly give you the results:
#
# The accuracy using a Pipeline is 0.807 with a fitting time of
# 0.043 seconds in 9 iterations
# The accuracy using a LogisticRegression is 0.807 with a fitting time of
# 0.110 seconds in 60 iterations
#
# We see that scaling the data before training the logistic regression was
# beneficial in terms of computational performance. Indeed, the number of
# iterations decreased as well as the training time. The generalization
# performance did not change since both models converged.

# %% [markdown]
# ```{warning}
# Working with non-scaled data will potentially force the algorithm to iterate
# more as we showed in the example above. There is also the catastrophic
# scenario where the number of required iterations is larger than the maximum
# number of iterations allowed by the predictor (controlled by the `max_iter`)
# parameter. Therefore, before increasing `max_iter`, make sure that the data
# are well scaled.
# ```

# %% [markdown]
# In this notebook we:
#
# * saw the importance of **scaling numerical variables**;
# * used a **pipeline** to chain scaling and logistic regression training.
