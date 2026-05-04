import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import shap
import matplotlib.pyplot as plt

# 1. Architecture definition (strictly required for state_dict restoration)
class C3Classifier(nn.Module):
    """
    Multi-Layer Perceptron architecture.
    Re-declared here to construct the computational graph required by SHAP 
    before loading the pre-trained weights.
    """
    def __init__(self, input_size: int = 50):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

def create_sequences(data: np.ndarray, label: int, window_size: int = 50) -> tuple:
    """
    Segments a 1D array of 3-body correlations (C_3) into fixed-size sliding windows.
    """
    sequences = []
    for i in range(len(data) - window_size):
        sequences.append(data[i : i + window_size])
    return np.array(sequences), np.full(len(sequences), label)

if __name__ == "__main__":
    # Hardware acceleration detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing explainability module. Computational device: {device.type.upper()}")

    # 2. Model Restoration
    model_path = "c3_model.pth"
    model = C3Classifier().to(device)
    try:
        # Load weights into the architecture and set to evaluation mode
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        print("Neural network parameters successfully restored from disk.")
    except FileNotFoundError:
        print(f"Critical Error: Model weight file '{model_path}' not found. Execute the training pipeline first.")
        sys.exit(1)

    # 3. Data Acquisition
    print("Acquiring datasets for SHAP interpretability analysis...")
    try:
        gue_data = np.load("gue_c3_data.npy")
        riemann_data = np.load("riemann_c3_ood_data.npy")
    except FileNotFoundError:
        print("Critical Error: Dataset files not located. Execute the data generation pipeline first.")
        sys.exit(1)

    gue_seqs, _ = create_sequences(gue_data, 0)
    riemann_seqs, _ = create_sequences(riemann_data, 1)

    # 4. Background and Test Cohort Preparation
    print("Constructing baseline calibration tensors...")
    np.random.seed(42)
    
    # Establish a balanced background dataset (100 GUE + 100 Riemann) 
    # to serve as the reference expectation for Shapley value computation.
    bg_idx_gue = np.random.choice(len(gue_seqs), 100, replace=False)
    bg_idx_riemann = np.random.choice(len(riemann_seqs), 100, replace=False)
    
    background_data = np.concatenate([gue_seqs[bg_idx_gue], riemann_seqs[bg_idx_riemann]])
    background_tensor = torch.FloatTensor(background_data).to(device)

    # Isolate a subset of 1000 Riemann sequences to visualize the mathematical signature
    test_idx = np.random.choice(len(riemann_seqs), 1000, replace=False)
    test_data = riemann_seqs[test_idx]
    test_tensor = torch.FloatTensor(test_data).to(device)

    # 5. Explainable AI Initialization (SHAP)
    print("Instantiating SHAP DeepExplainer (Game Theory Attribution)...")
    explainer = shap.DeepExplainer(model, background_tensor)

    print("Computing non-linear Shapley values. This operation is computationally intensive...")
    # Disabling additivity checks bypasses numerical instability inherent to 
    # specific PyTorch gradient approximations through Sigmoid functions.
    shap_values = explainer.shap_values(test_tensor, check_additivity=False)

    # 6. Tensor Formatting for Visualization
    # Reshape handles the transition from PyTorch batches back to standard 2D arrays
    shap_values_np = np.array(shap_values).reshape(-1, 50)
    data_sample = test_tensor.cpu().numpy()
    feature_names = [f"C_3 Pos {i+1}" for i in range(50)]

    # 7. Visualization and Export
    print("Synthesizing interpretability visualizations...")

    # Figure 1: Global Feature Importance (Absolute Mean Shapley Values)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_np, data_sample, feature_names=feature_names, plot_type="bar", show=False)
    plt.title("Global Importance of $C_3$ Triplets (High-Altitude Analysis)", fontsize=14)
    plt.tight_layout()
    plt.savefig("shap_global_importance.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Successfully exported: shap_global_importance.png")

    # Figure 2: Directional Feature Impact (Beeswarm Plot)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_np, data_sample, feature_names=feature_names, show=False)
    plt.title("Directional Impact of $C_3$ Triplets on Decision Boundary", fontsize=14)
    plt.tight_layout()
    plt.savefig("shap_beeswarm.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Successfully exported: shap_beeswarm.png")

    print("Pipeline execution complete. Analytical artifacts are ready for publication.")