# Introduction

## Course presentation

```{admonition} Welcome!
:class: remove-from-content-only

The goal of this course is to teach machine learning with scikit-learn to
beginners, even without a strong technical background.
```

This hands-on workshop will provide you with the basics of machine learning using Python.

Machine learning is the field devoted to methods and algorithms that 'learn' from data. It can be applied to a vast range of different domains, from linguistics to physics and from medical imaging to history.

This workshop covers the basics of machine learning in a practical and hands-on manner, so that upon completion, you will be able to train your first machine learning models and understand what next steps to take to improve them.

We start with data exploration and prepare the data so that it is suitable for machine learning. Then we learn how to train a model on the data using scikit-learn. We learn how to select the best model from the trained models and how to use different machine learning models (like linear regression, logistic regression, and decision tree models). Finally, we discuss some of the best practices when starting your own machine learning project.


## Setup instructions

### Installing Python

[Python][python] is a popular language for scientific computing, and a frequent choice
for machine learning as well.
To install Python, follow the [Beginner's Guide](https://wiki.python.org/moin/BeginnersGuide/Download) or head straight to the [download page](https://www.python.org/downloads/).

Please set up your python environment at least a day in advance of the workshop.
If you encounter problems with the installation procedure, ask your workshop organizers via e-mail for assistance so
you are ready to go as soon as the workshop begins.

## Installing the required packages{#packages}

[Pip](https://pip.pypa.io/en/stable/) is the package management system built into Python.
Pip should be available in your system once you installed Python successfully.

Open a terminal (Mac/Linux) or Command Prompt (Windows) in a location that you will use for the workshop 
and run the following commands.

### 1. Create a virtual environment
Create a [virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#create-and-use-virtual-environments) called `ml_workshop`:
#### On Linux/macOs

```shell
python3 -m venv ml_workshop
```

#### On Windows

```shell
py -m venv ml_workshop
```

### 2. Activate the newly created virtual environment


#### On Linux/macOs

```shell
source ml_workshop/bin/activate
```

#### On Windows

```shell
ml_workshop\Scripts\activate
```

Remember that you need to activate your environment every time you restart your terminal!

### 3. Install the required packages:

#### On Linux/macOs

```shell
python3 -m pip install matplotlib jupyter seaborn scikit-learn pandas
```

#### On Windows

```shell
py -m pip install matplotlib jupyter seaborn scikit-learn pandas
```


## Starting Jupyter Lab

> Jupyter Lab is compatible with Firefox, Chrome, Safari and Chromium-based browse
> Note that Internet Explorer and Edge are *not* supported.
> See the [Jupyter Lab documentation](https://jupyterlab.readthedocs.io/en/latest/getting_started/accessibility.html#compatibility-with-browsers-and-assistive-technology) for an up-to-date list of supported browsers.

To start Jupyter Lab, open a terminal (Mac/Linux) or Command Prompt (Windows) and type the command:

```shell
jupyter lab
```

### Download datasets
Download and extract [this datasets.zip file](https://zenodo.org/records/14851649/files/datasets.zip)
into the location that you will use for the workshop 
(make sure it is the same location as where you created the virtual environment).

## Prerequisites

The course aims to be accessible without a strong technical background. The
requirements for this course are:
- basic knowledge of Python programming : defining variables, writing
  functions, importing modules
- some prior experience with the NumPy, pandas and Matplotlib libraries is
  recommended but not required.

For a quick introduction on these requirements, you can go through these
[course materials](http://swcarpentry.github.io/python-novice-gapminder/)
or use the following resources:
- [Introduction to Python](https://scipy-lectures.org/intro/language/python_language.html)
- [Introduction to NumPy](https://sebastianraschka.com/blog/2020/numpy-intro.html)
- [Introduction to Pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
- [Introduction to Matplotlib](https://sebastianraschka.com/blog/2020/numpy-intro.html#410-matplotlib)


## MOOC material

The MOOC material is developed publicly under the [CC-BY license](
https://github.com/INRIA/scikit-learn-mooc/blob/main/LICENSE).

You can cite the original material through the project's Zenodo archive using the following DOI:
[10.5281/zenodo.7220306](https://doi.org/10.5281/zenodo.7220306).

It is possible to use the rocket icon at the top of each notebook
page to interactively execute the code cells via the Binder
service.

The videos are available as YouTube playlist at the Inria Learning Lab channel:

  https://www.youtube.com/playlist?list=PL2okA_2qDJ-m44KooOI7x8tu85wr4ez4f

[python]: https://python.org
[jupyter]: http://jupyter.org/
