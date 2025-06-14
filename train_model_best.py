from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
from preprocess import preprocess_data
import warnings

warnings.filterwarnings("ignore")

# Load and preprocess data
file_path = 'C:/Users/alpya/Documents/telco_churn_project/data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
X, y = preprocess_data(file_path)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Define hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5]
}

# GridSearchCV for RandomForest
print('==== Grid Search for Random Forest Classifier =====')
rf_model = RandomForestClassifier(random_state=42)
grid = GridSearchCV(rf_model, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid.fit(X_train_resampled, y_train_resampled)

best_rf = grid.best_estimator_
print(f'Best parameters: {grid.best_params_}')
print(f'Best F1 score: {grid.best_score_:.4f}')

# Voting Classifier
voting_clf = VotingClassifier(
    estimators=[('rf', best_rf)],
    voting='soft'
)

voting_clf.fit(X_train_resampled, y_train_resampled)
y_pred = voting_clf.predict(X_test)
y_proba = voting_clf.predict_proba(X_test)[:, 1]

print("\n==== Voting Classifier ====")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label="ROC Curve", color='darkorange')
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Voting Classifier")
plt.legend()
plt.tight_layout()
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(6, 4))
plt.plot(recall, precision, label="Precision-Recall Curve", color='darkgreen')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Voting Classifier")
plt.legend()
plt.tight_layout()
plt.show()
