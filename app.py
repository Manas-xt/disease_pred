from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model and symptom columns
model = joblib.load('disease_model.pkl')
symptom_cols = joblib.load('symptom_cols.pkl')

ML_TRAINING_CODE = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import joblib
import warnings
warnings.filterwarnings('ignore')

print('All libraries imported successfully!')

from google.colab import files
print('Upload Training.csv and Testing.csv')
uploaded = files.upload()

# Load datasets
train_df = pd.read_csv('Training.csv').drop(columns=['Unnamed: 133'], errors='ignore')
test_df  = pd.read_csv('Testing.csv')

# Strip whitespace from disease names
train_df['prognosis'] = train_df['prognosis'].str.strip()
test_df['prognosis']  = test_df['prognosis'].str.strip()

print('Train shape:', train_df.shape)
print('Test shape :', test_df.shape)
print()
train_df.head()

print('Dataset Info:')
print(f'Total training samples : {len(train_df)}')
print(f'Total testing samples  : {len(test_df)}')
print(f'Number of symptoms     : {train_df.shape[1] - 1}')
print(f'Number of diseases     : {train_df["prognosis"].nunique()}')
print()
print('Missing values:', train_df.isnull().sum().sum())
print()
print('All diseases:')
for i, d in enumerate(sorted(train_df['prognosis'].unique()), 1):
    print(f'{i}. {d}')

# Disease distribution
plt.figure(figsize=(14, 6))
train_df['prognosis'].value_counts().plot(kind='bar', color='steelblue')
plt.title('Disease Distribution in Training Data')
plt.xlabel('Disease')
plt.ylabel('Number of Samples')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Top 15 most common symptoms across all records
symptom_cols = train_df.columns[:-1].tolist()
top15_symptoms = train_df[symptom_cols].sum().sort_values(ascending=False).head(15)

plt.figure(figsize=(12, 4))
top15_symptoms.plot(kind='bar', color='seagreen')
plt.title('Top 15 Most Frequently Occurring Symptoms')
plt.xlabel('Symptom')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

symptom_cols = train_df.columns[:-1].tolist()

X_train = train_df[symptom_cols]
y_train = train_df['prognosis']

X_test  = test_df[symptom_cols]
y_test  = test_df['prognosis']

print('X_train shape:', X_train.shape)
print('X_test shape :', X_test.shape)
print('Unique diseases in train:', y_train.nunique())

# criterion='entropy' makes it ID3
dt_model = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_model.fit(X_train, y_train)

dt_pred     = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)

print(f'Decision Tree Accuracy: {dt_accuracy * 100:.2f}%')
print()
print(classification_report(y_test, dt_pred))

# Visualize top 3 levels of the tree
plt.figure(figsize=(22, 8))
plot_tree(
    dt_model,
    feature_names=symptom_cols,
    class_names=dt_model.classes_,
    filled=True,
    max_depth=3,
    fontsize=8
)
plt.title('Decision Tree - Top 3 Levels (ID3 / Entropy)', fontsize=14)
plt.tight_layout()
plt.show()

# Top 15 most important symptoms
importances = pd.Series(dt_model.feature_importances_, index=symptom_cols)
top15 = importances.sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 5))
top15.plot(kind='bar', color='steelblue')
plt.title('Top 15 Most Important Symptoms (Decision Tree)')
plt.ylabel('Importance Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

nb_pred     = nb_model.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_pred)

print(f'Naive Bayes Accuracy: {nb_accuracy * 100:.2f}%')
print()
print(classification_report(y_test, nb_pred))

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

knn_pred     = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)

print(f'KNN Accuracy: {knn_accuracy * 100:.2f}%')
print()
print(classification_report(y_test, knn_pred))

# Print correct and wrong predictions (Lab 9 requirement)
print('KNN - All Test Predictions:\n')
print(f'{"#":<5} {"Actual Disease":<42} {"Predicted Disease":<42} {"Result"}')
print('-' * 105)
correct = 0
for i, (actual, predicted) in enumerate(zip(y_test, knn_pred), 1):
    result = 'Correct' if actual == predicted else 'Wrong'
    if actual == predicted:
        correct += 1
    print(f'{i:<5} {actual:<42} {predicted:<42} {result}')
print()
print(f'Total Correct: {correct}/{len(y_test)}')

results = pd.DataFrame({
    'Model': ['Decision Tree (ID3)', 'Naive Bayes', 'KNN (k=5)'],
    'Accuracy (%)': [
        round(dt_accuracy * 100, 2),
        round(nb_accuracy * 100, 2),
        round(knn_accuracy * 100, 2)
    ]
})

print('=' * 40)
print('Model Accuracy Comparison')
print('=' * 40)
print(results.to_string(index=False))
print('=' * 40)

# Accuracy bar chart
plt.figure(figsize=(8, 5))
colors = ['steelblue', 'seagreen', 'tomato']
bars = plt.bar(results['Model'], results['Accuracy (%)'], color=colors, width=0.5)
plt.ylim(0, 110)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy (%)')
for bar, acc in zip(bars, results['Accuracy (%)']):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 1,
             f'{acc}%', ha='center', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.show()

# Confusion matrix for Decision Tree
cm = confusion_matrix(y_test, dt_pred, labels=dt_model.classes_)
plt.figure(figsize=(18, 14))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=dt_model.classes_,
            yticklabels=dt_model.classes_)
plt.title('Confusion Matrix - Decision Tree')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

def predict_disease(symptoms_input, model, symptom_cols):
    input_vector = [1 if col in symptoms_input else 0 for col in symptom_cols]
    input_df = pd.DataFrame([input_vector], columns=symptom_cols)
    return model.predict(input_df)[0]

# Sample test
sample = ['itching', 'skin_rash', 'nodal_skin_eruptions']
print(f'Input Symptoms : {sample}')
print(f'Decision Tree  : {predict_disease(sample, dt_model, symptom_cols)}')
print(f'Naive Bayes    : {predict_disease(sample, nb_model, symptom_cols)}')
print(f'KNN            : {predict_disease(sample, knn_model, symptom_cols)}')

print('All available symptoms:')
print(symptom_cols)

# Change this list and run!
my_symptoms = ['fever', 'chills', 'joint_pain', 'vomiting']

print(f'\nYour symptoms     : {my_symptoms}')
print(f'Predicted disease : {predict_disease(my_symptoms, dt_model, symptom_cols)}')

joblib.dump(nb_model, 'disease_model.pkl')
joblib.dump(symptom_cols, 'symptom_cols.pkl')

print('disease_model.pkl saved')
print('symptom_cols.pkl saved')
print()
print('Downloading files...')
files.download('disease_model.pkl')
files.download('symptom_cols.pkl')
"""

ML_LIBRARIES = [
    {"name": "pandas", "purpose": "Data loading and dataframe operations", "type": "Library"},
    {"name": "numpy", "purpose": "Numerical arrays and vector handling", "type": "Library"},
    {"name": "matplotlib", "purpose": "Charts and model visualizations", "type": "Visualization"},
    {"name": "seaborn", "purpose": "Confusion matrix heatmap", "type": "Visualization"},
    {"name": "scikit-learn", "purpose": "ML models, metrics, and utilities", "type": "ML Framework"},
    {"name": "joblib", "purpose": "Model and feature-list serialization", "type": "Tool"},
    {"name": "Flask", "purpose": "Web serving and prediction API", "type": "Web Framework"}
]

ML_ALGORITHMS = [
    {"name": "Decision Tree (ID3)", "details": "DecisionTreeClassifier(criterion='entropy', random_state=42)", "use": "Interpretable baseline model"},
    {"name": "Gaussian Naive Bayes", "details": "GaussianNB()", "use": "Primary deployment model"},
    {"name": "K-Nearest Neighbors", "details": "KNeighborsClassifier(n_neighbors=5)", "use": "Comparison model"}
]

ML_TOOLS = [
    "Google Colab (training notebook execution)",
    "CSV datasets: Training.csv and Testing.csv",
    "Python warnings module for clean output",
    "Flask templates for web report and predictor UI"
]

@app.route('/')
def index():
    return render_template('index.html', symptoms=symptom_cols)

@app.route('/report')
def report():
    return render_template(
        'report.html',
        training_code=ML_TRAINING_CODE,
        libraries=ML_LIBRARIES,
        algorithms=ML_ALGORITHMS,
        tools=ML_TOOLS
    )

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    selected_symptoms = data.get('symptoms', [])

    # Build input vector
    input_vector = [1 if col in selected_symptoms else 0 for col in symptom_cols]
    input_array = np.array(input_vector).reshape(1, -1)

    # Predict
    prediction = model.predict(input_array)[0]

    # Get top 3 probabilities if available
    try:
        proba = model.predict_proba(input_array)[0]
        top3_idx = np.argsort(proba)[::-1][:3]
        top3 = [
            {"disease": model.classes_[i], "confidence": round(float(proba[i]) * 100, 1)}
            for i in top3_idx if proba[i] > 0
        ]
    except:
        top3 = [{"disease": prediction, "confidence": 100.0}]

    return jsonify({
        "prediction": prediction,
        "top3": top3
    })

if __name__ == '__main__':
    app.run(debug=True)
