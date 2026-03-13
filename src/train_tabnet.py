import pandas as pd
import numpy as np
import torch
import os
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Paths
DATA_PATH = "data/processed/loan_cleaned.csv"
# FIX: Do not add .zip here. TabNet adds it automatically.
MODEL_SAVE_PATH = "models/saved_models/tabnet_loan_model" 

def train_loan_model():
    print("🚀 Starting TabNet Training (Final Polish)...")
    
    # --- STEP 0: CLEANUP OLD FILES ---
    # We delete the old model so the saver doesn't crash
    if os.path.exists(MODEL_SAVE_PATH + ".zip"):
        os.remove(MODEL_SAVE_PATH + ".zip")
        print("   -> Deleted old model file to prevent conflicts.")

    # 1. Load Data
    df = pd.read_csv(DATA_PATH)
    
    # Use 'target' (your actual column name)
    target = 'target' 
    features = [col for col in df.columns if col != target]
    
    X = df[features].values
    y = df[target].values
    
    print(f"   -> Loaded {X.shape[0]:,} records.")
    
    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Define Model
    clf = TabNetClassifier(
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size":10, "gamma":0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='entmax',
        verbose=1
    )
    
    # 4. Train
    print("   -> Training started...")
    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_name=['valid'],
        eval_metric=['auc'],
        max_epochs=15,          
        patience=5,
        batch_size=1024, 
        virtual_batch_size=128,
        num_workers=0,
        weights=1,
        drop_last=False
    )
    
    # 5. Save
    # The library will automatically create 'models/saved_models/tabnet_loan_model.zip'
    clf.save_model(MODEL_SAVE_PATH)
    print(f"   ✅ TabNet Model Saved successfully to {MODEL_SAVE_PATH}.zip")

if __name__ == "__main__":
    train_loan_model()