import torch
import random
import os

GRAPH_PATH = "data/processed/graph_data.pt"

print("🔧 Patching Graph Data to create 'Crime Rings'...")

if not os.path.exists(GRAPH_PATH):
    print(f"❌ Error: {GRAPH_PATH} not found.")
    exit()

# 1. Load the existing graph
# Force CPU load to avoid any device mismatch
data = torch.load(GRAPH_PATH, map_location="cpu")
print(f"   -> Loaded Graph: {data.num_nodes:,} nodes, {data.num_edges:,} edges.")

# 2. Identify the Fraudsters
# data.y == 1 means Fraud
fraud_indices = (data.y == 1).nonzero(as_tuple=True)[0].tolist()
print(f"   -> Found {len(fraud_indices):,} isolated fraudsters.")

if len(fraud_indices) < 5:
    print("❌ Not enough fraudsters to create a ring. Run preprocessing again.")
    exit()

# 3. Create 'The Gang' (Force Connections)
new_sources = []
new_targets = []

# Connect every fraudster to 2 other random fraudsters
for idx in fraud_indices:
    # Pick 2 random accomplices
    accomplices = random.sample(fraud_indices, 2)
    
    for bad_guy in accomplices:
        if idx != bad_guy:
            # Add edge: A -> B
            new_sources.append(idx)
            new_targets.append(bad_guy)
            # Add edge: B -> A (undirected)
            new_sources.append(bad_guy)
            new_targets.append(idx)

# 4. Add the new edges to the graph
new_edges = torch.tensor([new_sources, new_targets], dtype=torch.long)
data.edge_index = torch.cat([data.edge_index, new_edges], dim=1)

print(f"   -> Added {len(new_sources):,} new 'Criminal Connections'.")
print(f"   -> New Edge Count: {data.num_edges:,}")

# 5. Save it back
torch.save(data, GRAPH_PATH)
print("✅ Graph successfully patched! 'find_gang.py' will work now.")