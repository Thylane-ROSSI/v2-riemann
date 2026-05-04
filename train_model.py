import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def create_sequences(data, label, window_size=50):
    """
    Creates sequences of a fixed window size from the C3 dataset.
    """
    sequences = []
    for i in range(len(data) - window_size):
        sequences.append(data[i : i + window_size])
    return np.array(sequences), np.full(len(sequences), label)

class C3Classifier(nn.Module):
    """
    Multi-Layer Perceptron for classifying C3 sequences.
    Uses F.relu to maintain compatibility with SHAP Gradient/DeepExplainer.
    """
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

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load data
    print("Loading datasets...")
    try:
        gue_data = np.load("gue_c3_data.npy")
        riemann_data = np.load("riemann_c3_ood_data.npy")
    except FileNotFoundError:
        print("Error: Dataset files not found. Please run generate_data.py first.")
        exit(1)

    # 2. Create sequences
    print("Creating sequences...")
    gue_seqs, gue_labels = create_sequences(gue_data, 0)
    riemann_seqs, riemann_labels = create_sequences(riemann_data, 1)

    # Balance the datasets (downsampling the majority class)
    min_length = min(len(gue_seqs), len(riemann_seqs))
    gue_seqs = gue_seqs[:min_length]
    gue_labels = gue_labels[:min_length]
    riemann_seqs = riemann_seqs[:min_length]
    riemann_labels = riemann_labels[:min_length]

    print(f"Dataset balanced: {min_length} sequences per class.")

    # 3. Prepare Train/Test split
    X = np.concatenate([gue_seqs, riemann_seqs])
    y = np.concatenate([gue_labels, riemann_labels])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train).to(device), 
        torch.FloatTensor(y_train).to(device)
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_tensor = torch.FloatTensor(X_test).to(device)

    # 4. Initialize model and training parameters
    model = C3Classifier().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 10

    print("Starting training process...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

    # 5. Evaluation
    print("Evaluating model on test set...")
    model.eval()
    with torch.no_grad():
        y_pred = model(test_tensor).cpu().squeeze().numpy()

    auc_score = roc_auc_score(y_test, y_pred)
    print(f"Final AUC-ROC Score: {auc_score:.4f}")

    # 6. Save the model weights
    model_path = "c3_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved successfully to {model_path}")