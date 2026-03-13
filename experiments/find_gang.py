import torch
import joblib
from torch_geometric.utils import degree

print("🕵️‍♀️ Hunting for a 'Connected' Criminal...")

# 1. Load Data
graph_data = torch.load("data/processed/graph_data.pt")
node_map = joblib.load("models/saved_models/node_map.pkl")

# 2. Calculate connections for everyone
# degree(index) tells us how many edges each node has
node_degrees = degree(graph_data.edge_index[0], num_nodes=graph_data.num_nodes)

# 3. Find a Fraudster (y=1) with Friends (degree > 0)
found_count = 0
print("\n🔥 TRY THESE USER IDs (They have friends):")

# Create a reverse map to look up names
idx_to_id = {v: k for k, v in node_map.items()}

for i in range(len(graph_data.y)):
    is_fraud = graph_data.y[i].item() == 1
    num_friends = int(node_degrees[i].item())
    
    if is_fraud and num_friends >= 3: # Must have at least 3 friends
        user_id = idx_to_id.get(i, "Unknown")
        print(f"   👉 {user_id} (Is Fraud? YES | Friends: {num_friends})")
        found_count += 1
        
        if found_count >= 5: # Stop after finding 5 good examples
            break

if found_count == 0:
    print("   ❌ No connected fraudsters found. Try running preprocessing.py again to reshuffle.")