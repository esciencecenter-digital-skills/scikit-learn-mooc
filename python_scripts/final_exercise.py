# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # 📝 Exercise: Try out learned skills on Ames Housing dataset
# The goal of this exercise is to apply all learned skills to a new dataset: The Ames Housing dataset.

# We use this dataset in a regression setting to predict the sale prices of houses based on house features. That is, the goal is to predict the target `SalePrice` from numeric and/or categorical features in the dataset.

# Remember to explore the dataset before building models. Then, start simple and step-by-step expand your approach to create better and better models.

# You can load the data as follows:
# %%
import pandas as pd

house_prices = pd.read_csv("../datasets/ames_housing_no_missing.csv")

# %% [markdown]
# **In case this exercise is difficult:**
# To help you get started, you can take a look at the [description](datasets_ames_housing) of the dataset.
