# Backorder Prediction Modeling
Personal Data Analysis project of Backorder Prediction using Kaggle dataset

## Objective
Predicting Backorders in Inventory Mangement Context. Original dataset from Kaggle's "Can You Predict Product Backorders?",

Product backorder may be the result of strong sales performance (e.g. the product is in such high demand that production cannot keep up with sales). However, backorders can upset consumers, lead to canceled orders and decreased customer loyalty. Companies want to avoid backorders, but also avoid overstocking every product (leading to higher inventory costs).


## Data
I used [data](https://www.kaggle.com/tiredgeek/predict-bo-trial) from Kaggle, which had 1.9 million observations of parts in an 8 week period. The source of the data is unreferenced.

-   **Outcome**: whether the part went on backorder
-   **Predictors**: Current inventory, sales history, forecasted sales, recommended stocking amount, part risk flags etc. (22 predictors in total)

**Update**: The dataset seem no longer available on kaggle website. Do some digging. Its easy to find online.

## Key considerations of the data:

-   **Imbalanced outcome**: Only 0.7% of parts actually go on backorder.
-   **Outliers and skewed predictors**: Part quantities (stock, sales etc.) can be on very different scales.
-   **Missing data**: A few variables have data that are missing (not at random).
-   **n&gt;&gt;p**: There are many observations (1.9 million) relative to the number of predictors (22).

## Usage & Installation

### Dependencies
- pandas: read / write csv file
- numpy: data format
- sklearn: classifiers
- Imbalanced-learn 0.2.1

### Install
``` pip install requirements.txt```


## Tasks
### Data Wrangling
- [x] Handling inconsistent column names and datatype
- [x] Missing Data handling
- [x] Removal of duplicate rows
- [x] Handling columns with repetitive values
- [x] Handling the outliers
- [x] Write the clean data into a new file for further steps

### Data trend analysis

- [] Relationship among features
- [] Data questioning

### Model training and validation

- [] train-test split of the data
- [] Model training
- [] Model Tuning and cross validation

## Tests
- Training sample order_classifier.build_classifier_sample()
- Evaluating sample order_classifier.eval_result_rf()

## API Reference
- Training from csv data-file and writing model to model-file order_classifier.training_from_csv_data(data_file, model_file)
- Loading model from model-file order_classifier.readModel(file_path)
- Read from file and convert data to numpy format order_preprocess.read_data(file_path)
- Predict order_classifier.predict(data, classifier)



## Reference
R. B. Santis, E. P. Aguiar and L. Goliatt, "Predicting Material Backorders in Inventory Management using Machine Learning," 4th IEEE Latin American Conference on Computational Intelligence, Arequipa, Peru, 2017.

Available from: https://www.researchgate.net/publication/319553365_Predicting_Material_Backorders_in_Inventory_Management_using_Machine_Learning
