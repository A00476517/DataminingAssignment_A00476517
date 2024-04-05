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

# Exploring Feature Relationships: Correlation Analysis

Here we present a chart that shows how each number in our dataset relates to all the other numbers. We left out PassengerID because it's just a way to identify each passenger. In the chart, stronger relationships are shown with brighter colors. If the color is more red, it means the numbers tend to go up together. If it's more blue, they tend to go in opposite directions. If the color is closer to white, it means the relationship between the numbers is not very strong.

IMG

Pclass and Fare are related because first-class tickets are usually more expensive than third-class ones. This makes sense because higher-class passengers pay more for better accommodations.

SibSp and Parch are somewhat connected because larger families tend to have more siblings, spouses, parents, or children traveling with them. Meanwhile, solo travelers typically don't have any family members with them.

Also, Pclass is noticeably linked to survival. This means that your ticket class has a big influence on whether you survived or not. Passengers in higher classes had better chances of surviving compared to those in lower classes. This shows how your social status on the Titanic affected your chances of making it out alive.

# Feature Engineering for two variable

- Child

      Pclass    1    2    3
      Child                
      False   213  167  447
      True      3   17   44
      Sex    female  male
      Child              
      False     283   544
      True       31    33

   The plot for passengers in Pclass 1 seems intriguing initially, but upon closer inspection, it's revealed that there are only 3 children in this group. This small sample size makes any apparent pattern likely just random noise. However, the plots for the other two passenger classes are more compelling, especially concerning male children. It's important to note that since we're selecting by Age, which has many missing values, some children will be categorized as "Child == False." Nevertheless, this analysis appears to be useful in understanding the survival patterns among different passenger classes and age groups.

- Family

      Survived    0    1
      Family            
      0         374  163
      1          72   89
      2          43   59
      3           8   21
      4          12    3
      5          19    3
      6           8    4
      7           6    0
      10          7    0
 
      Once again, we observe that having 1 to 3 family members onboard tends to lead to better chances of survival. This feature combines both the number of siblings/spouses (SibSp) and parents/children (Parch) aboard, giving us a larger dataset to analyze.

# Models accuracy and summary
   IM

 The accuracies obtained from the three models are as follows:

   - Logistic Regression: 0.7989
   - Decision Tree Classifier: 0.8045
   - Random Forest Classifier: 0.8045

      Comparing these results, we can observe that both the Decision Tree Classifier and the Random Forest Classifier outperform the Logistic Regression model slightly, with both achieving an accuracy of approximately 80.45%. This indicates that the ensemble methods, particularly the Random Forest, were able to capture more complex patterns in the data compared to the linear Logistic Regression model.

      However, it's worth noting that the difference in accuracy between the Decision Tree Classifier and the Random Forest Classifier is negligible in this case. This could suggest that the complexity added by the ensemble methods might not be significantly beneficial for this particular dataset.

      Overall, while the Decision Tree and Random Forest models perform slightly better than Logistic Regression in terms of accuracy, the difference is minimal, and further analysis, such as examining other evaluation metrics or fine-tuning hyperparameters, may be needed to determine the best model for this specific task.
