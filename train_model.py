from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
from preprocess import preprocess_data
import warnings
import seaborn as sns
import joblib

warnings.filterwarnings("ignore")
# Load and preprocess data
file_path = 'C:/Users/alpya/Documents/telco_churn_project/data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
X, y = preprocess_data(file_path)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Apply SMOTE for handling class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Define parameter grids for each model
param_grids = {
    "RandomForest": {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5]
    },
    "LightGBM": {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'learning_rate': [0.01, 0.1, 0.2],
    },
    "XGBoost": {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'learning_rate': [0.01, 0.1, 0.2],
    },
    "GradientBoosting": {
        'n_estimators': [100, 200],
        'max_depth': [5, 10],
        'learning_rate': [0.01, 0.1, 0.2]
    }
}

# Define and tune the model
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'LightGBM': lgb.LGBMClassifier(random_state=42),
    'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42)
}
# Perform grid search for each model
best_estimators = {}
model_scores = {}
for name, model in models.items():
    print(f'==== Grid Search for {name} =====')
    grid = GridSearchCV(model, param_grids[name], cv=5, scoring='f1', n_jobs=-1)
    grid.fit(X_train_resampled, y_train_resampled)
    best_estimators[name] = grid.best_estimator_
    model_scores[name] = grid.best_score_
    print(f'Best parameters for {name}: {grid.best_params_}')
    print(f'Best F1 score for {name}: {grid.best_score_}')

# Plot model scores
plt.figure(figsize=(10, 5))
sns.barplot(x=list(model_scores.keys()), y=list(model_scores.values()))
plt.ylabel('Best F1 Score')
plt.ylim(0.5, 0.9)
plt.title('Best F1 Scores by Model')
for i, score in enumerate(model_scores.values()):
    plt.text(i, score + 0.01, f'{score:.4f}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.show()

# Save best model
for name, model in best_estimators.items():
    joblib.dump(model, f'{name}_best_model.pkl')

# Voting Classifier with the best estimators
voting_clf = VotingClassifier(
    estimators=[
        ('rf', best_estimators['RandomForest']),
        ('lgbm', best_estimators['LightGBM']),
        ('xgb', best_estimators['XGBoost']),
        ('gb', best_estimators['GradientBoosting'])
    ],
    voting='soft'
)

voting_clf.fit(X_train_resampled, y_train_resampled)
y_pred = voting_clf.predict(X_test)
y_proba = voting_clf.predict_proba(X_test)[:, 1]

print("==== Voting Classifier ====")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, label="Voting ROC")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Voting Classifier")
plt.legend()
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_proba)
plt.figure()
plt.plot(recall, precision, label="Voting PR")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Voting Classifier")
plt.legend()
plt.show()
