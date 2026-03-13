import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
import os
from tqdm import tqdm

# Paths
GRAPH_PATH = "data/processed/graph_data.pt"
MODEL_SAVE_PATH = "models/saved_models/gnn_fraud_model.pt"

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

def train_fraud_model():
    print("🚀 Starting GNN Training (AGGRESSIVE RECALL MODE)...")
    
    device = torch.device('cpu') 
    print(f"   -> Training on: {device}")

    if not os.path.exists(GRAPH_PATH):
        print("❌ Graph data not found.")
        return
        
    data = torch.load(GRAPH_PATH, map_location=device)
    print(f"   -> Loaded Graph: {data.num_nodes:,} Nodes.")

    # --- 1. CALCULATE AGGRESSIVE WEIGHTS ---
    num_safe = (data.y == 0).sum().item()
    num_fraud = (data.y == 1).sum().item()
    
    # Standard Weight
    base_weight = num_safe / num_fraud
    
    # AGGRESSIVE MULTIPLIER: Force the model to care 5x more about fraud
    # This pushes Recall UP but might lower Precision slightly (which is fine)
    final_weight = base_weight * 5.0 
    
    print(f"   -> Class Imbalance: {num_safe} Safe vs {num_fraud} Fraud.")
    print(f"   -> Base Weight: {base_weight:.2f}")
    print(f"   -> ⚔️ AGGRESSIVE Weight applied: {final_weight:.2f}x")

    # --- 2. DEEPER SAMPLING ---
    # Increased to [10, 10] to catch more distant connections
    train_loader = NeighborLoader(
        data,
        num_neighbors=[10, 10], 
        batch_size=2048,            
        input_nodes=None,           
        shuffle=True,
        num_workers=0               
    )

    model = FraudGNN(in_channels=1, hidden_channels=64, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005) # Lower LR for stability
    
    pos_weight = torch.tensor([final_weight]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print("\n⚡ Training Started (10 Epochs)...")
    model.train()
    
    # --- 3. LONGER TRAINING ---
    for epoch in range(10): 
        total_loss = 0
        total_batches = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/10"):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch.x, batch.edge_index)
            
            batch_size = batch.batch_size
            out = out[:batch_size]
            target = batch.y[:batch_size].float().unsqueeze(1)
            
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_batches += 1
            
        print(f"   Epoch {epoch+1}: Avg Loss = {total_loss / total_batches:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"   ✅ Aggressive Model Saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_fraud_model()