import joblib
import torch
import random

# Load the map and the graph data
print("Loading Database...")
try:
    user_map = joblib.load("models/saved_models/node_map.pkl")
    graph_data = torch.load("data/processed/graph_data.pt")
    
    # Get list of all IDs
    all_users = list(user_map.keys())
    
    # Find indices of fraudsters (where label y == 1)
    fraud_indices = (graph_data.y == 1).nonzero(as_tuple=True)[0].tolist()
    
    # Find indices of safe users (where label y == 0)
    safe_indices = (graph_data.y == 0).nonzero(as_tuple=True)[0].tolist()

    print(f"✅ Database loaded: {len(all_users):,} users.")
    print(f"   - Known Fraudsters: {len(fraud_indices):,}")
    print(f"   - Safe Users: {len(safe_indices):,}")

    print("\n--- 🕵️‍♀️ TRY THESE IDs TO SEE THE GRAPH ---")
    
    print("\n🔴 KNOWN FRAUDSTERS (Should trigger Red Graph):")
    for _ in range(3):
        idx = random.choice(fraud_indices)
        # Find the user ID that maps to this index (reverse lookup)
        # Note: This is slow for huge maps, but fine for a demo script
        for user_id, node_idx in user_map.items():
            if node_idx == idx:
                print(f"   👉 {user_id}")
                break

    print("\n🟢 SAFE USERS (Should trigger Green Graph):")
    for _ in range(3):
        idx = random.choice(safe_indices)
        for user_id, node_idx in user_map.items():
            if node_idx == idx:
                print(f"   👉 {user_id}")
                break
                
    print("------------------------------------------------")

except FileNotFoundError:
    print("❌ Error: Could not find model files. Make sure you are in the project folder.")