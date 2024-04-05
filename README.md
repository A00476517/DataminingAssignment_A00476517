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

## Graph

## Analysis summary from the graph

In this summary dashboard, we present visualizations to explore the distributions of individual features in the Titanic dataset. Using Matplotlib's subplot tool, we line up the individual plots in a grid, combining overlapping histograms for ordinal features and bar plots for categorical features. Let's delve into the insights gained from studying these features:

- Age:

   - The median ages for survivors and non-survivors are identical. However, there's a noticeable difference in survival rates among age groups.
   - Fewer young adults (ages 18 - 30-ish) survived, whereas children younger than 10-ish had a higher survival rate.
   - There are no obvious outliers indicating problematic input data. The distribution of ages is consistent with expectations, with a notable shortage of teenagers compared to younger children.

- Pclass (Passenger Class):

   - A clear trend emerges, indicating that being a 1st class passenger increases the chances of survival.
   - Unfortunately, this trend highlights the inequity present in life, where socio-economic status influences survival outcomes.

- SibSp & Parch (Number of Siblings/Spouses and Parents/Children):

   - Having 1-3 siblings/spouses/parents/children on board suggests proportionally better survival numbers compared to being alone or traveling with a large family.
   - This finding underscores the importance of social support networks during crises like the Titanic disaster.

- Embarked (Port of Embarkation):

   - Unexpectedly, embarking at "C" (Cherbourg) resulted in a higher survival rate than embarking at "S" (Southampton).
   - While this correlation is intriguing, it may be influenced by other variables not explored here.

- Fare:

   - Linear scaling isn't effective due to a smaller number of extreme fare values. Logarithmic transformation is a suitable alternative.
   - The plot reveals lower survival chances for passengers with cheaper cabins, possibly due to their location deeper inside the ship, away from lifeboats.

# More analysis
- Survived: 342 (38.4 percent)
- Not Survived: 549 (61.6 percent)
- Total: 891

These numbers reveal important insights into the survival outcomes of passengers aboard the Titanic:

      - Survival Rate:

         1) Out of the total 891 passengers in the dataset, 342 individuals (approximately 38.4%) survived the disaster.
         2) Conversely, 549 individuals (approximately 61.6%) did not survive.

      - Imbalance in Survival:

         1) The dataset demonstrates a significant imbalance in survival outcomes, with a higher number of passengers perishing compared to those who survived.
         2) This disparity underscores the tragic nature of the Titanic disaster, where a majority of passengers did not survive.

      - Potential Challenges for Analysis:

         1) The unequal distribution of survival outcomes may pose challenges for predictive modeling and analysis tasks, particularly in ensuring that models accurately capture patterns and factors influencing survival.
         2) Addressing class imbalance may be necessary to prevent biased model performance and improve the overall reliability of analyses.
