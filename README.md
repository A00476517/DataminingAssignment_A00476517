# DataminingAssignment_A00476517 - Titanic Data Analysis

This repository contains Python code for analyzing the Titanic dataset as part of an assignment. The goal of the assignment is to perform exploratory data analysis (EDA) on the Titanic dataset and find interesting insights or patterns that may have influenced the survival of passengers.

## Dataset

The dataset used for this analysis is the "Python Walk-Through for Titanic Data Analysis" dataset. It contains information about passengers aboard the Titanic, including features such as:

- Passenger ID
- Survived (Survival status)
- Pclass (Passenger class)
- Name
- Sex
- Age
- SibSp (Number of siblings/spouses aboard)
- Parch (Number of parents/children aboard)
- Ticket (Ticket number)
- Fare
- Cabin
- Embarked (Port of embarkation)

The dataset is available in two CSV files:
- `train.csv`: Training dataset containing information about passengers, including survival status.
- `test.csv`: Test dataset containing similar information but without survival status.

## Analysis

The Python code in this repository performs the following tasks:

1. Loading the dataset using pandas.
2. Exploratory data analysis (EDA) to understand the structure of the data, including:
   - Displaying the first few rows of the dataset.
   - Summary statistics for numerical features.
   - Checking for missing values.
3. Visualizing relationships between different features and survival status using seaborn and matplotlib.
4. Printing various statistics and insights obtained from the analysis, such as median age of survivors and non-survivors.

## Usage
