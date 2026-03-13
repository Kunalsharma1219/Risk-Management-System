import joblib
import torch
import random

# Paths
NODE_MAP_PATH = "models/saved_models/node_map.pkl"
GRAPH_DATA_PATH = "data/processed/graph_data.pt"

def get_demo_ids():
    print("🔍 Scanning VERITAS Brain for Valid IDs...")
    
    # 1. Load the Map and the Graph
    try:
        node_map = joblib.load(NODE_MAP_PATH)
        graph_data = torch.load(GRAPH_DATA_PATH, map_location="cpu")
    except FileNotFoundError:
        print("❌ Error: Could not find model files. Are you in the right directory?")
        return

    # 2. Invert map to get ID from Index
    # node_map is { 'ID': 0, 'ID2': 1 ... }
    # we need { 0: 'ID', 1: 'ID2' ... }
    idx_to_id = {v: k for k, v in node_map.items()}
    
    # 3. Find Criminal Indices (where y == 1)
    # Convert tensor to list of indices
    fraud_indices = (graph_data.y == 1).nonzero(as_tuple=True)[0].tolist()
    safe_indices = (graph_data.y == 0).nonzero(as_tuple=True)[0].tolist()
    
    print(f"\n✅ Database Loaded: {len(node_map):,} Total Users.")
    print(f"   - Known Fraudsters: {len(fraud_indices):,}")
    print(f"   - Safe Users: {len(safe_indices):,}")
    
    # 4. Pick Random Samples
    print("\n📋 USE THESE IDS FOR YOUR DEMO:")
    print("-" * 40)
    
    print("\n🔴 TRY THESE (Should be REJECTED/FRAUD):")
    for _ in range(5):
        idx = random.choice(fraud_indices)
        real_id = idx_to_id.get(idx, "Unknown")
        print(f"   👉 {real_id}")
        
    print("\n🟢 TRY THESE (Should be APPROVED/SAFE):")
    for _ in range(5):
        idx = random.choice(safe_indices)
        real_id = idx_to_id.get(idx, "Unknown")
        print(f"   👉 {real_id}")

    print("-" * 40)

if __name__ == "__main__":
    get_demo_ids()