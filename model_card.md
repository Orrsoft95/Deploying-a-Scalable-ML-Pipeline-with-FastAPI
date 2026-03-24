# Model Card
  

## Model Details
- **Model Type**: This model is a **Random Forest Classifier**.
-  **Training Details**: **3-fold cross validation** was utliized to optimize the n_estimators, max_depth, min_samples_split, and min_samples_leaf for the random forest classifier.
    - ***Best Parameters***:
        - **max_depth**: 30
        - **min_samples_leaf**: 2
        - **min_samples_split**: 5
        - **n_estimators**: 200

  

## Intended Use
This model is intended to provide broad statistical estimates of salary ranges based on demographic & socioeconomic features derived from census data. It's designed for **research & exploratory analysis**, such as researching income distribution trends or identifying wage gaps across population groups.

This model is **not** intended for:
- Making hiring or compensation decisions for specific individuals
- Use as a legally-binding salary benchmark
- Applications in any automated decision-making pipelines that directly affect individuals' employment or financial outcomes.
  

## Training Data
- **Source**: Training data was sourced from Ron Kohavi's 1994 census data extraction, located [here](https://archive.ics.uci.edu/dataset/20/census+income).
- **Size**:  **80%** of the census data (~26,049 records) were utilized for model training.
- **Features**
    - Age
    - Workclass (private employment, self-employed, local govt. employee, etc.)
    - education: highest achieved level of education
    - marital-status
    - occupation
    - relationship
    - race
    - sex
    -  capital-gain
    - capital-loss
    - hours-per-week: # of hours worked per week
    - native-country
 - **Known Biases**: *American Indians* and *Other* individuals are grossly underrepresented in the census data, and as such, the model is unlikely to accurately predict the incomes of individuals of those races. This same concern is held for those with a marital status of *Married-AF-spouse* or *Married-spouse-absent*.



  

## Evaluation Data
The model was evaluated on a test set of 6,512 records (20% of census data). Evaluation metrics include precision, recall, and F1-score, reported both overall (see below) and across demographic subgroups & stored in **slice_output.txt**.


## Metrics
- **Precision**: 0.7817 on test set.
- **Recall**: 0.62 on test set.
- **F1 Score**: 0.6915 on test set.
- **Key Slices**: This version of the model performs notably conservatively within *several* values for **marital-status**. It is exceedingly conservative when predicting the salary of never-married or separated individuals for example, as indicated by these slices having very high precision scores, but extremely low recall scores.

As mentioned above, **slice_output.txt** can be viewed to assess the model's performance across different subgroups.

  

## Ethical Considerations
This model & its predictions may reflect and reinforce the historical income disparities present in the underlying census data. Users should be aware that applying this model to individual-level compensation decisions risks further perpetuating any such systemic wage inequities.
  

## Caveats and Recommendations
- Census data ages quickly, and this data was sourced from the **1994 census**. As such, salary estimates may not reflect current income distributions.
- Performance may be poor for demographic groups that are underrepresented in the training data.