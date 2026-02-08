#Churn Prediction With Graphs

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

#Loading CSV
df = pd.read_csv("Telco-Customer-Churn.csv")  # local file

#Target variable
df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

#Cleaning TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

#Features and labels
X = df.drop(["Churn", "customerID"], axis=1)
y = df["Churn"]

#Split numeric/categorical
num_features = ["tenure", "MonthlyCharges", "TotalCharges"]
cat_features = [c for c in X.columns if c not in num_features]

#Preprocessing pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

#Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Logistic Regression pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])
model.fit(X_train, y_train)

#Predict & evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

#Visualizations
plt.figure(figsize=(12,4))

#Churn distribution
plt.subplot(1,2,1)
y.value_counts().plot(kind='bar', color=['skyblue','salmon'])
plt.title("Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Count")

#Tenure vs Churn
plt.subplot(1,2,2)
df.boxplot(column="tenure", by="Churn", grid=False, patch_artist=True,
           boxprops=dict(facecolor="lightgreen"))
plt.title("Tenure vs Churn")
plt.suptitle("")

plt.tight_layout()
plt.show()

#Conclusion

print("Poject Completed Successfully")
