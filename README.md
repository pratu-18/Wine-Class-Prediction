# WINE CLASSIFICATION USING K-NEAREST NEIGHBORS

1. PROJECT DESCRIPTION

---

This project implements a Machine Learning classification pipeline to classify
wine samples into their respective classes using the K-Nearest Neighbors (KNN)
algorithm.

The project follows a step-by-step Machine Learning workflow including:

* Dataset loading
* Data cleaning
* Feature and target separation
* Train-test splitting
* Feature scaling
* Model creation
* Model training
* Model testing
* Model evaluation

The KNN model is implemented using the KNeighborsClassifier provided by
Scikit-learn.

2. PROJECT OBJECTIVE

---

The main objective of this project is to build a KNN classification model
that can predict the class of a wine sample based on its chemical properties.

The project also demonstrates the importance of feature scaling when using
distance-based Machine Learning algorithms such as KNN.

3. DATASET

---

Dataset file:

WinePredictor.csv

The dataset contains wine-related chemical measurements and a target column
named:

Class

The Class column represents the output class of each wine sample.

The remaining columns are used as independent features for classification.

4. MACHINE LEARNING ALGORITHM

---

Algorithm:

K-Nearest Neighbors (KNN)

Python implementation:

KNeighborsClassifier

Number of neighbors:

K = 9

The value K = 9 was selected after hyperparameter tuning based on model
accuracy.

KNN classifies a new sample by identifying its nearest training samples and
using their class labels to determine the final prediction.

5. TECHNOLOGIES USED

---

Programming Language:

Python 3.10 or higher

Development Environment:

Visual Studio Code (VS Code)

Libraries:

* Pandas
* Matplotlib
* Scikit-learn

6. PROJECT WORKFLOW

---

The project is implemented as a Machine Learning pipeline.

Step 1: Load the Dataset

The dataset is loaded from a CSV file using Pandas.

Function used:

pd.read_csv()

Step 2: Clean the Dataset

Missing values are removed from the dataset using:

df.dropna(inplace=True)

The program then displays:

* Number of records
* Number of columns
* Shape of the dataset

Step 3: Separate Independent and Dependent Variables

Independent variables:

All columns except Class

Dependent variable:

Class

The input features are stored in X and the target variable is stored in Y.

Step 4: Split the Dataset

The dataset is divided into training and testing datasets using
train_test_split().

Configuration:

test_size = 0.5
random_state = 42
stratify = Y

The stratify parameter maintains approximately the same class distribution
in both the training and testing datasets.

Step 5: Feature Scaling

KNN is a distance-based algorithm, so feature scaling is performed using
StandardScaler.

Training data is used to calculate the scaling parameters.

The correct implementation is:

X_train_scaled = scalar.fit_transform(X_train)
X_test_scaled = scalar.transform(X_test)

The scaler must not be fitted again on the testing data.

This prevents data leakage from the test dataset into the training process.

Step 6: Build the Model

A KNN classification model is created using:

KNeighborsClassifier(n_neighbors=9)

Step 7: Train the Model

The KNN model is trained using the scaled training data:

model.fit(X_train_scaled, Y_train)

Step 8: Test the Model

The trained model predicts the class of the test dataset:

Y_pred = model.predict(X_test_scaled)

Step 9: Model Evaluation

The model performance is evaluated using:

* Accuracy Score
* Confusion Matrix

7. DATA PREPROCESSING

---

The project performs the following preprocessing operations:

1. Missing value removal
2. Feature and target separation
3. Train-test splitting
4. Feature scaling

These preprocessing steps prepare the data before model training.

8. FEATURE SCALING

---

StandardScaler is used to standardize the input features.

Standardization transforms the features so that they have approximately:

Mean = 0
Standard Deviation = 1

This is important for KNN because the algorithm calculates distances between
data points.

Without scaling, features with larger numerical ranges may have a greater
influence on the distance calculation.

9. TRAIN-TEST SPLIT

---

The dataset is divided using:

test_size = 0.5

This means:

50 percent of the data is used for training.

50 percent of the data is used for testing.

The random_state value is set to 42 to make the split reproducible.

The stratify parameter is set to Y so that the class proportions are
approximately preserved in both datasets.

10. MODEL CONFIGURATION

---

Model:

KNeighborsClassifier

Number of neighbors:

9

The K value was selected after hyperparameter tuning based on the accuracy
obtained during experimentation.

11. MODEL EVALUATION

---

Accuracy Score:

Accuracy measures the proportion of correctly classified test samples.

The following function is used:

accuracy_score(Y_test, Y_pred)

Confusion Matrix:

The confusion matrix provides a detailed view of correct and incorrect
classification results for each wine class.

The following function can be used:

confusion_matrix(Y_test, Y_pred)

12. PROJECT STRUCTURE

---

Wine-Classification-KNN/
|
|-- WinePredictor.csv
|-- wine_classifier.py
|-- README.txt
|-- requirements.txt

13. INSTALLATION

---

Requirement:

Python 3.10 or higher

Check Python version:

python --version

Create a virtual environment:

python -m venv venv

Activate the virtual environment on Windows:

venv\Scripts\activate

Install project dependencies:

pip install -r requirements.txt

14. EXECUTION

---

Open the project folder in Visual Studio Code.

Run the Python program using:

python wine_classifier.py

15. EXPECTED OUTPUT

---

The program displays:

* Initial dataset records
* Dataset shape
* Number of records
* Number of columns
* Input feature names
* Output column name
* Training dataset shape
* Testing dataset shape
* Feature scaling completion message
* Model creation message
* Model training completion message
* Model accuracy

16. IMPORTANT IMPLEMENTATION NOTE

---

For StandardScaler, the scaler must be fitted only on the training dataset.

Correct:

X_train_scaled = scalar.fit_transform(X_train)
X_test_scaled = scalar.transform(X_test)

Incorrect:

X_test_scaled = scalar.fit_transform(X_test)

Fitting the scaler separately on the test data can introduce data leakage and
can result in an unreliable evaluation of the model.

17. LEARNING OUTCOMES

---

This project demonstrates practical knowledge of:

* Python
* Pandas
* Data preprocessing
* Missing value handling
* Feature engineering fundamentals
* Train-test splitting
* Stratified sampling
* Feature scaling
* StandardScaler
* K-Nearest Neighbors
* Hyperparameter selection
* Model training
* Model prediction
* Model evaluation
* Accuracy calculation
* Confusion matrix
* Machine Learning pipeline

18. FUTURE ENHANCEMENTS

---

Possible improvements include:

* Perform systematic hyperparameter tuning using GridSearchCV.
* Compare different K values.
* Add cross-validation.
* Add precision, recall and F1-score.
* Visualize the confusion matrix.
* Compare KNN with other classification algorithms.
* Add a prediction function for new wine samples.
* Save the trained model using joblib or pickle.
* Create a Streamlit interface for prediction.

19. AUTHOR

---

Author:

Pratiksha Mahale

Project Type:

Machine Learning Classification Case Study

Algorithm:

K-Nearest Neighbors

Domain:

Machine Learning and Data Science
