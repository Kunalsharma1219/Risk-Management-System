import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
import joblib

# Paths
RAW_PATH = "data/raw"
PROCESSED_PATH = "data/processed"
MAPPINGS_PATH = "models/saved_models"

class RiskDataProcessor:
    def __init__(self):
        print("🔧 Initializing Data Processor...")
        os.makedirs(PROCESSED_PATH, exist_ok=True)
        os.makedirs(MAPPINGS_PATH, exist_ok=True)

    def process_paysim(self):
        """ Cleans PaySim Data for Graph Neural Network (Gate 1) """
        print("\n[1/2] Processing PaySim (Fraud Data)...")
        file_path = f"{RAW_PATH}/paysim.csv"
        
        if not os.path.exists(file_path):
            print(f"❌ Error: {file_path} not found.")
            return

        df = pd.read_csv(file_path)
        
        # Filter to suspicious types
        df = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()

        # Map Users to IDs
        all_users = pd.concat([df['nameOrig'], df['nameDest']]).unique()
        user_map = {name: i for i, name in enumerate(all_users)}
        joblib.dump(user_map, f"{MAPPINGS_PATH}/node_map.pkl")
        
        df['source_id'] = df['nameOrig'].map(user_map)
        df['dest_id'] = df['nameDest'].map(user_map)
        
        # Save
        cols = ['step', 'type', 'amount', 'source_id', 'dest_id', 'isFraud']
        df[cols].to_csv(f"{PROCESSED_PATH}/paysim_cleaned.csv", index=False)
        print("   ✅ PaySim Data Processed & Saved!")

    def process_loan_data(self):
        """ Cleans LendingClub Data for TabNet (Gate 2) """
        print("\n[2/2] Processing LendingClub (Loan Data)...")
        file_path = f"{RAW_PATH}/loan_data.csv"
        
        if not os.path.exists(file_path):
            print(f"❌ Error: {file_path} not found.")
            return

        print("   -> Loading large loan file...")
        df = pd.read_csv(file_path, low_memory=False)
        
        # 1. Select Key Features
        keep_cols = [
            'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 
            'emp_length', 'home_ownership', 'annual_inc', 'verification_status',
            'loan_status', 'dti', 'open_acc', 'pub_rec', 'revol_bal', 'total_acc'
        ]
        existing_cols = [c for c in keep_cols if c in df.columns]
        df = df[existing_cols]

        # 2. Target Encoding
        df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]
        df['target'] = df['loan_status'].apply(lambda x: 1 if x == 'Charged Off' else 0)
        df = df.drop(columns=['loan_status'])
        df = df.dropna()

        # 3. FIX: Label Encode All Text Columns
        # This converts " 36 months" -> 0, "RENT" -> 1, etc.
        cat_cols = ['term', 'grade', 'emp_length', 'home_ownership', 'verification_status']
        
        print("   -> Encoding text columns to numbers...")
        for col in cat_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

        # 4. Save
        df.to_csv(f"{PROCESSED_PATH}/loan_cleaned.csv", index=False)
        print(f"   ✅ Loan Data Processed! ({len(df):,} records saved)")

if __name__ == "__main__":
    processor = RiskDataProcessor()
    processor.process_paysim()
    processor.process_loan_data()
    print("\n🎉 DATA PIPELINE COMPLETE.")