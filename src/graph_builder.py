import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
import os
from tqdm import tqdm

# Paths
PROCESSED_PATH = "data/processed"
GRAPH_PATH = "data/processed/graph_data.pt"

class GraphBuilder:
    def __init__(self):
        print("🕸️  Initializing Graph Builder...")

    def build_graph(self):
        print("   -> Loading Processed PaySim Data...")
        df = pd.read_csv(f"{PROCESSED_PATH}/paysim_cleaned.csv")
        
        # 1. Define Edges (Source -> Destination)
        # PyG expects a LongTensor of shape [2, num_edges]
        print("   -> Constructing Edge Index (Money Trails)...")
        sources = df['source_id'].values
        destinations = df['dest_id'].values
        edge_index = torch.tensor([sources, destinations], dtype=torch.long)

        # 2. Define Node Features (The 'x' matrix)
        # For this version, we will use the transaction amount attached to the sender
        # In a Pro version, we would aggregate features (Total sent, avg amount, etc.)
        # Here, we keep it simple: Feature = [1.0] (Placeholder to exist) 
        # We need as many nodes as the max ID in our map
        num_nodes = max(df['source_id'].max(), df['dest_id'].max()) + 1
        print(f"   -> Creating Features for {num_nodes:,} Users...")
        
        # Simple Identity Feature for now (Can be upgraded to DeepWalk embeddings later)
        # We use a small feature vector of size 1 for memory efficiency on Mac
        x = torch.ones((num_nodes, 1), dtype=torch.float)

        # 3. Define Labels (Who is the Fraudster?)
        # We map fraud labels from transactions back to the Source User
        print("   -> Assigning Fraud Labels...")
        y = torch.zeros(num_nodes, dtype=torch.long)
        
        # Get indices of fraud transactions
        fraud_tx = df[df['isFraud'] == 1]
        fraudsters = fraud_tx['source_id'].values
        
        # Mark them as Class 1 (Fraud)
        y[fraudsters] = 1

        # 4. Create the PyG Data Object
        data = Data(x=x, edge_index=edge_index, y=y)
        
        # Save it
        print(f"   -> Saving Graph Object to {GRAPH_PATH}...")
        torch.save(data, GRAPH_PATH)
        print("   ✅ Graph Built Successfully!")
        
        # Stats
        print(f"\n📊 GRAPH STATS:")
        print(f"   - Nodes (Users): {data.num_nodes:,}")
        print(f"   - Edges (Txns):  {data.num_edges:,}")
        print(f"   - Fraudsters:    {data.y.sum().item():,}")
        print(f"   - Feature Shape: {data.x.shape}")

if __name__ == "__main__":
    builder = GraphBuilder()
    builder.build_graph()