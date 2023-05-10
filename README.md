# Machine Learning & Statistical Modelling Enabled Portfolio Optimisation 

## Introduction

This project accompanies my Machine Learning and Statistical Modelling Portfolio Optimisation dissertation. The project was used to get data from a financial modelling API, store the data, create ML models, evaluate models and produce outputs for my dissertation.

## Getting set up

Clone the repository onto your local computer.

Set up your virtual environment: `python -m venv venv`

Activate the virtual environment:
- Windows: `source venv/Scripts/activate`
- Mac/Linux: `source venv/bin/activate`

Install the packages from the requirements.txt: `pip install -r requirements.txt`

You will also need the environment variables in a .env file at the top of the project. You will need to include two variables:
- DB_PATH = 'sqlite:///data/priceData.db'
- FMP_API_KEY = # your own financial modelling prep api key here #

To get an API key see the FMP website: https://site.financialmodelingprep.com/developer/docs/

Once set up run the analysis.ipynb notebook or some of the data scripts (see details below)!

## Overview of the project structure

The project is split into two parts each within their own sub directories: data - for getting and storing data, and analysis - for performing the evaluation of models and producing output for the dissertation.

### Data

The data directory contains the database structure in the data_model folder.

To create the database run the createDatabase script: `python createDatabase.py`

To load data into the database run the dataLoader script: `python dataLoader.py`

### Analysis

To run the analysis run the cells in the jupyter notebook.

The notebook imports classes that are used for the modelling and output from the analysis directory and uses methods to produce charts/tables.

