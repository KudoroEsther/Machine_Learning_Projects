# Importing libraries

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import permutation_importance

import joblib

#Loading data
url = r"C:\Users\owner\OneDrive\Desktop\ML_projects\Fault_Analysis\merged_dataset.csv"

# url = r"C:\Users\ncc333\Desktop\ML_projects\Fault_Analysis\merged_dataset.csv"
def load_data(url):
    df = pd.read_csv(url, sep=",")
    return df

dff = load_data(url)
df = dff.copy()

#droppping time
df= df.drop(['t'], axis=1)


# Preprocessing

# Feature engineering
def engineered_features(df, window=100, min_periods =1):
    # df=df.copy()
    volt_cols = ['Va', 'Vb', 'Vc']  # Phase voltages
    curr_cols = ['Ia', 'Ib', 'Ic']  # Phase currents
    all_cols = volt_cols + curr_cols 

    #Time domain features
    #Rolling Root mean square
    for col in all_cols:
        df[f"{col}_rms"] = df[col].rolling(window, min_periods).apply(lambda x: np.sqrt(np.mean(x**2)), raw= True)

    #Rolling peak-to-peak
    for col in all_cols:
        df[f"{col}_ptp"] = df[col].rolling(window, min_periods).apply(lambda x: np.ptp(x), raw=True)

    # Frequency domain features
    # Phase-to-phase voltage differences (line-line voltages)
    df['Vab'] = df['Va'] - df['Vb']
    df['Vbc'] = df['Vb'] - df['Vc']
    df['Vca'] = df['Vc'] - df['Va']

    # Phase-to-phase current differences
    df['Iab'] = df['Ia'] - df['Ib']
    df['Ibc'] = df['Ib'] - df['Ic']
    df['Ica'] = df['Ic'] - df['Ia']

    #Power features
    df['P_total'] = (
        df['Va']*df['Ia'] +
        df['Vb']*df['Ib'] +
        df['Vc']*df['Ic']
    )
    print(f"New engineered features: {[col for col in df.columns if col not in dff.columns]}")

engineered_features(df)

#Encoding
df = df.replace({"Fault":{0: "No fault", 1: "LLLG fault", 2 :"LG fault", 3: "LLG fault"}})

# Data splitring and scaling
#First level splitting: Splitting into features and target
X = df.drop(columns="Fault")
y = df['Fault']

#Scaling 
scaler = MinMaxScaler(feature_range=(0,1))
X_scaled = scaler.fit_transform(X)

#Seond level splitting: splitting into test and train
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=234, stratify=y)


#Modelling
models = {
    "LogisticRegression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbours": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Ada Boost": AdaBoostClassifier(),
    "Extra Trees": ExtraTreesClassifier()
}

# Initializing a dictionary to store the models
results = {}

for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[model_name] = accuracy
    print(f"{model_name} Accuracy: {accuracy:.3f}")

    #Adding confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                display_labels=['No fault (0)',"LLLG fault (1)", "LG fault (2)", "LLG fault (3)" ])
    print(classification_report(y_test, y_pred, zero_division=1))

    #Plotting with model name as title
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xticks(rotation = 25)
    plt.tight_layout()
    plt.show()

result_df = pd.DataFrame(results.items(), columns=["Model", "Accuracy"])
result_df = result_df.sort_values(by="Accuracy", ascending=False)
result_df.reset_index(drop=True)
result_df


#Hyperparameter tuning
#creating an instance of the best model
rf = RandomForestClassifier()
# exacting the pareameters of the model
rf.get_params()

param_grid = {
    "n_estimators": [10, 50, 100],
    "max_depth": [10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=10, cv=5, verbose=2, random_state=234, n_jobs=-1)
random_search.fit(X_train, y_train)
print(f"Best parameters for Random Forest {random_search.best_params_}")

#Cross validation
best_model = random_search.best_estimator_
#might be x_train againt y_train
cv_score = cross_val_score(best_model, X_scaled, y, cv=5)
# cv_score = cross_val_score(best_model, X_train, y_train, cv=5)
print(f"Cross validation scores for Random Forest : {cv_score}")
print(f"Mean cross validation score: {cv_score.mean()}")

y_pred = best_model.predict(X_test)
print(f"\nClassification Report: ")
print(classification_report(y_test, y_pred, zero_division=1))


#Saving preprocessed data
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(best_model, 'best_model.pkl')
# joblib.dump(selected_features, 'selected_features.pkl')
print("Model and scaler saved successfully!")