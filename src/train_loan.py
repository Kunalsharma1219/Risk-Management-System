import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE 
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "loan_cleaned.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "saved_models", "loan_model.pkl")

def train_loan_model():
    print("🚀 Training High-Accuracy Credit Model (XGBoost)...")
    
    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Could not find data at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    target = 'target'
    features = [col for col in df.columns if col != target]
    
    X = df[features]
    y = df[target]
    
    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. SMOTE (Balance the data)
    print("   -> Applying SMOTE to generate synthetic data...")
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    
    # 4. Initialize XGBoost (OPTIMIZED FOR ACCURACY)
    # We REMOVED 'scale_pos_weight' so the model stops panicking.
    # We added 'max_depth=8' to make it smarter.
    clf = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,  # Slightly deeper trees for better accuracy
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False
    )
    
    # 5. Train
    print("   -> Fitting Model...")
    clf.fit(X_train_bal, y_train_bal)
    
    # 6. Evaluate
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n✅ XGBoost Model Trained! Accuracy: {acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=['Paid', 'Default']))
    
    # 7. Save
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_SAVE_PATH)
    print(f"   -> Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_loan_model()