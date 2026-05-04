import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def create_sequences(data: np.ndarray, label: int, window_size: int = 50) -> tuple:
    """
    Segments a 1D array of 3-body correlations (C_3) into fixed-size sliding windows.

    Parameters:
    - data (np.ndarray): The input 1D array containing C_3 values.
    - label (int): The binary label assigned to these sequences (0 for GUE, 1 for Riemann).
    - window_size (int): The length of each sequential window.

    Returns:
    - tuple: A tuple containing the matrix of sequences and the corresponding label array.
    """
    sequences = []
    # Construct rolling windows to preserve sequential correlations
    for i in range(len(data) - window_size):
        sequences.append(data[i : i + window_size])
    return np.array(sequences), np.full(len(sequences), label)

class C3Classifier(nn.Module):
    """
    Multi-Layer Perceptron (MLP) architecture optimized for binary classification
    of sequential mathematical and physical correlations.

    The architecture utilizes PyTorch functional ReLU activations to ensure
    strict mathematical compatibility with specific Explainable AI frameworks 
    (e.g., SHAP DeepExplainer).
    """
    def __init__(self, input_size: int = 50):
        super().__init__()
        # Hidden Layer 1: Captures initial non-linear representations
        self.fc1 = nn.Linear(input_size, 32)
        # Hidden Layer 2: Dimensionality reduction and feature extraction
        self.fc2 = nn.Linear(32, 16)
        # Output Layer: Binary logit projection
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

if __name__ == "__main__":
    # Hardware acceleration detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing training procedure. Computational device: {device.type.upper()}")

    # 1. Data Acquisition
    print("Loading correlation datasets...")
    try:
        gue_data = np.load("gue_c3_data.npy")
        riemann_data = np.load("riemann_c3_ood_data.npy")
    except FileNotFoundError:
        print("Critical Error: Dataset files not located. Ensure the data generation pipeline has been executed.")
        sys.exit(1)

    # 2. Sequence Generation
    print("Constructing sliding window sequences...")
    gue_seqs, gue_labels = create_sequences(gue_data, 0)
    riemann_seqs, riemann_labels = create_sequences(riemann_data, 1)

    # Class Balancing: Downsampling the majority class to prevent model bias
    min_length = min(len(gue_seqs), len(riemann_seqs))
    gue_seqs = gue_seqs[:min_length]
    gue_labels = gue_labels[:min_length]
    riemann_seqs = riemann_seqs[:min_length]
    riemann_labels = riemann_labels[:min_length]

    print(f"Datasets balanced strictly: {min_length:,} sequences allocated per class.")

    # 3. Data Splitting and Tensor Encapsulation
    X = np.concatenate([gue_seqs, riemann_seqs])
    y = np.concatenate([gue_labels, riemann_labels])

    # Maintaining an 80/20 split for robust generalization assessment
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train).to(device),
        torch.FloatTensor(y_train).to(device)
    )
    # Batch size configured to 64 for stochastic gradient descent
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_tensor = torch.FloatTensor(X_test).to(device)

    # 4. Model Instantiation and Hyperparameters
    model = C3Classifier().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 10

    print(f"Commencing neural network optimization over {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1:02d}/{epochs}] - Binary Cross-Entropy Loss: {avg_loss:.4f}")

    # 5. Model Evaluation
    print("Evaluating generalization capabilities on the test set...")
    model.eval()
    with torch.no_grad():
        y_pred = model(test_tensor).cpu().squeeze().numpy()

    # AUC-ROC provides a threshold-invariant metric for binary classification
    auc_score = roc_auc_score(y_test, y_pred)
    print(f"Empirical Result - Final AUC-ROC Score: {auc_score:.4f}")

    # 6. Model Persistence
    model_path = "c3_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Architecture weights persistently saved to: {model_path}")