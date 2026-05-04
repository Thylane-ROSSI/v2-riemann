import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import shap
import matplotlib.pyplot as plt

# 1. Define the model architecture (required to load the weights)
class C3Classifier(nn.Module):
    def __init__(self, input_size=50):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

def create_sequences(data, label, window_size=50):
    """
    Creates sequences of a fixed window size from the C3 dataset.
    """
    sequences = []
    for i in range(len(data) - window_size):
        sequences.append(data[i : i + window_size])
    return np.array(sequences), np.full(len(sequences), label)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load the trained model
    model_path = "c3_model.pth"
    model = C3Classifier().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print(f"Error: '{model_path}' not found. Please run train_model.py first.")
        exit(1)

    # 3. Load the data
    print("Loading datasets for SHAP analysis...")
    try:
        gue_data = np.load("gue_c3_data.npy")
        riemann_data = np.load("riemann_c3_ood_data.npy")
    except FileNotFoundError:
        print("Error: Dataset files not found. Please run generate_data.py first.")
        exit(1)

    gue_seqs, _ = create_sequences(gue_data, 0)
    riemann_seqs, _ = create_sequences(riemann_data, 1)

    # 4. Prepare Background and Test datasets
    # We sample a balanced background set to calibrate the explainer
    np.random.seed(42)
    bg_idx_gue = np.random.choice(len(gue_seqs), 100, replace=False)
    bg_idx_riemann = np.random.choice(len(riemann_seqs), 100, replace=False)
    
    background_data = np.concatenate([gue_seqs[bg_idx_gue], riemann_seqs[bg_idx_riemann]])
    background_tensor = torch.FloatTensor(background_data).to(device)

    # We sample 1000 sequences for the actual explanation to ensure visual clarity
    test_idx = np.random.choice(len(riemann_seqs), 1000, replace=False)
    test_data = riemann_seqs[test_idx]
    test_tensor = torch.FloatTensor(test_data).to(device)

    # 5. Initialize SHAP Explainer
    print("Initializing SHAP DeepExplainer...")
    explainer = shap.DeepExplainer(model, background_tensor)

    print("Calculating SHAP values (this may take a minute)...")
    # check_additivity=False prevents the rounding error crash with Sigmoid
    shap_values = explainer.shap_values(test_tensor, check_additivity=False)

    # 6. Format the output
    shap_values_np = np.array(shap_values).reshape(-1, 50)
    data_sample = test_tensor.cpu().numpy()
    feature_names = [f"C3 Pos {i+1}" for i in range(50)]

    # 7. Generate and save the plots
    print("Generating plots...")

    # Plot 1: Global Importance (Bar Chart)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_np, data_sample, feature_names=feature_names, plot_type="bar", show=False)
    plt.title("Global Importance of C3 Triplets (OOD Analysis)", fontsize=14)
    plt.tight_layout()
    plt.savefig("shap_global_importance.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: shap_global_importance.png")

    # Plot 2: Directional Impact (Beeswarm Chart)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_np, data_sample, feature_names=feature_names, show=False)
    plt.title("Directional Impact of C3 Triplets", fontsize=14)
    plt.tight_layout()
    plt.savefig("shap_beeswarm.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: shap_beeswarm.png")

    print("Analysis complete. The graphical assets are ready for publication.")