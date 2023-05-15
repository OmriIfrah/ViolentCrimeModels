# ViolentCrimeModels
A collection of models for predicting violent crime in US neighborhoods based on various parameters such as demographics, socioeconomic factors, and historical crime data.

## Data Source
The data used in this project is sourced from the Communities and Crime dataset from the UCI Machine Learning Repository.

The Communities and Crime dataset contain various socio-economic, law enforcement, and demographic variables for different communities in the United States. It provides valuable information for analyzing and predicting crime rates in different neighborhoods.

To access the dataset, you can visit the provided link and follow the instructions provided on the UCI Machine Learning Repository website.
https://archive.ics.uci.edu/ml/datasets/Communities+and+Crime

By including this information, you not only provide clarity about the data source but also acknowledge the original creators and maintain compliance with any applicable licensing terms.

## Dataset Overview
The dataset contains 1994 observations and 128 variables. The objective is to predict the "ViolentCrimesPerPop" column, which represents the number of violent crime cases (murder, violent robbery, rape, assault) in different places in the USA.

Preprocessing Steps:
- Remove the first five variables (fold, communityname, community, country, state) as they are not used for prediction.
- Normalize all variables to a range of [0, 1].

Data File:
The dataset is stored in the "communities.data" file.

Column Names:
The data file does not contain column names. The column names are obtained from the "communities.names" file.


## Project Overview

This project focuses on predicting violent crime in US neighborhoods using a variety of parameters. The goal is to develop models that can accurately forecast the likelihood of violent crime occurrences based on demographic, socioeconomic, and historical crime data. By leveraging machine learning techniques and analyzing the relationships between these parameters, the project aims to provide valuable insights for researchers, policymakers, and law enforcement agencies.

Key Features:
- Predicting violent crime occurrences in US neighborhoods.
- Analyzing the impact of various parameters on crime rates.
- Identifying important factors contributing to crime patterns.
- Providing insights for crime prevention and resource allocation.

## Libraries Used

The following libraries are used in this project:

- **pandas**: Used for data manipulation and analysis. You can install it using `pip install pandas`.
- **scipy.stats**: Provides statistical functions and distributions. You can install it using `pip install scipy`.
- **matplotlib.pyplot**: Used for data visualization, creating plots, and charts. You can install it using `pip install matplotlib`.
- **numpy**: Offers support for large, multi-dimensional arrays and mathematical functions. You can install it using `pip install numpy`.
- **statsmodels.api**: Provides a range of statistical models and tools for analysis. You can install it using `pip install statsmodels`.

