import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "credit_risk_dataset.csv"
OUTPUT_PATH = Path(__file__).parent.parent.parent / "output"
OUTPUT_PATH.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
cats = ['person_home_ownership','loan_intent','loan_grade','cb_person_default_on_file']
df = pd.get_dummies(df, columns=cats)
X = df.drop("loan_status", axis=1)
y = df["loan_status"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)[:,1]
preds = clf.predict(X_test)

results = pd.DataFrame({'y_true': y_test, 'y_pred_prob': proba, 'y_pred': preds})
results.to_csv(OUTPUT_PATH / "random_forest_preds.csv", index=False)

importances = pd.DataFrame({"Feature": X.columns, "Importance": clf.feature_importances_})
importances.to_csv(OUTPUT_PATH / "random_forest_importances.csv", index=False)

print("Random Forest test probabilities:", proba[:5])