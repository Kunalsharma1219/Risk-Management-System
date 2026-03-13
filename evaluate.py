import pandas as pd
import joblib
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# XGBoost Paths
XGB_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "loan_cleaned.csv")
XGB_MODEL_PATH = os.path.join(BASE_DIR, "models", "saved_models", "loan_model.pkl")

# GNN Paths
GNN_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "graph_data.pt")
# Updated to match the exact save path from your train_fraud.py
GNN_MODEL_PATH = os.path.join(BASE_DIR, "models", "saved_models", "gnn_fraud_model.pt") 

# --- GNN MODEL DEFINITION ---
# We must define the "blueprint" so PyTorch can load the weights into it
class FraudGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv3(x, edge_index)
        return x 

# --- EVALUATION LOGIC ---

def evaluate_xgboost():
    print("\n==================================================")
    print("📊 EVALUATING CREDIT MODEL (XGBoost)")
    print("==================================================")
    
    if not os.path.exists(XGB_DATA_PATH) or not os.path.exists(XGB_MODEL_PATH):
        print("❌ Error: XGBoost data or model not found. Run 'train_loan.py' first.")
        return
        
    df = pd.read_csv(XGB_DATA_PATH)
    target = 'target'
    features = [col for col in df.columns if col != target]
    
    X = df[features]
    y = df[target]
    
    # Split using the same random state as training
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"   -> Testing on {len(X_test)} unseen tabular records.")
    
    model = joblib.load(XGB_MODEL_PATH)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    print("\n--- 🏆 XGBoost Performance Results ---")
    print(f"Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, preds, target_names=['Paid', 'Default']))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

def evaluate_gnn():
    print("\n==================================================")
    print("🕸️ EVALUATING FRAUD MODEL (GNN)")
    print("==================================================")
    
    device = torch.device('cpu')
    
    if not os.path.exists(GNN_DATA_PATH):
        print(f"❌ Error: Graph data not found at {GNN_DATA_PATH}")
        return
        
    if not os.path.exists(GNN_MODEL_PATH):
        print(f"❌ Error: GNN model weights not found at '{GNN_MODEL_PATH}'.")
        return

    # 1. Load the Graph Data
    print("   -> Loading Graph Data...")
    # Using weights_only=False to suppress the PyTorch future warning for the data file
    graph_data = torch.load(GNN_DATA_PATH, map_location=device, weights_only=False)
    
    # 2. Initialize the Model Blueprint
    print("   -> Initializing FraudGNN...")
    model = FraudGNN(in_channels=1, hidden_channels=64, out_channels=1).to(device)
    
    # 3. Load the Weights (State Dict) into the Blueprint
    print("   -> Loading Trained Weights...")
    # Using weights_only=True is safer for loading state_dicts
    model.load_state_dict(torch.load(GNN_MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    # 4. Run Predictions
    print(f"   -> Testing on graph with {graph_data.num_nodes:,} nodes.")
    with torch.no_grad():
        out = model(graph_data.x.to(device), graph_data.edge_index.to(device))
        # Since we used BCEWithLogitsLoss during training, we apply sigmoid now
        probabilities = torch.sigmoid(out.squeeze())
        preds = (probabilities > 0.5).int()
        
    y_true = graph_data.y.cpu().numpy()
    preds = preds.cpu().numpy()

    acc = accuracy_score(y_true, preds)
    
    print("\n--- 🏆 GNN Performance Results ---")
    print(f"Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_true, preds, target_names=['Safe', 'Fraud']))
    print("Confusion Matrix:\n", confusion_matrix(y_true, preds))

if __name__ == "__main__":
    evaluate_xgboost()
    evaluate_gnn()
    print("\n==================================================")
    print("✅ FULL SYSTEM EVALUATION COMPLETE")
    print("==================================================")