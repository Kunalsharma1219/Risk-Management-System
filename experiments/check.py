import torch
import networkx as nx
from pytorch_tabnet.tab_model import TabNetClassifier

def verify_system():
    print("\n🔍 CHECKING SYSTEM CONFIGURATION FOR MAC M3...\n")
    
    # 1. Check Python & PyTorch
    print(f"✅ PyTorch Version: {torch.__version__}")
    
    # 2. Check GPU Acceleration (MPS)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        x = torch.ones(1, device=device)
        print(f"✅ SUCCESS: Apple Metal (MPS) is AVAILABLE! (Acceleration Enabled)")
        print(f"   [Test Tensor Device: {x.device}]")
    else:
        print("❌ WARNING: Apple MPS not found. Running on CPU (Will be slow).")

    # 3. Check TabNet
    try:
        clf = TabNetClassifier()
        print("✅ TabNet Library is correctly installed.")
    except Exception as e:
        print(f"❌ TabNet Error: {e}")

    # 4. Check Graph Libraries
    try:
        import torch_geometric
        print(f"✅ PyTorch Geometric Version: {torch_geometric.__version__}")
    except ImportError:
        print("❌ PyTorch Geometric not found.")

    print("\n🚀 SYSTEM READY FOR DEVELOPMENT.\n")

if __name__ == "__main__":
    verify_system()__package__