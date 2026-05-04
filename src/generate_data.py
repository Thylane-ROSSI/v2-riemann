import urllib.request
import torch
import numpy as np
import time

def generate_gue_c3(n_matrices=10000, n_size=1000, device='cpu'):
    """
    Generates 3-body correlations (C_3) for Gaussian Unitary Ensemble (GUE) matrices.
    
    Parameters:
    - n_matrices (int): Number of random matrices to generate.
    - n_size (int): Dimension of each square matrix (N x N).
    - device (str): Computational device ('cpu' or 'cuda').
    
    Returns:
    - np.ndarray: Concatenated array of C_3 correlation values.
    """
    print(f"Generating {n_matrices} GUE matrices (N={n_size}) on {device}...")
    start_time = time.time()
    c3_all = []
    
    for i in range(n_matrices):
        # Generate complex Gaussian noise and construct Hermitian matrix
        A = torch.randn(n_size, n_size, dtype=torch.complex64, device=device)
        H = (A + A.mH) / 2.0
        
        # Compute eigenvalues
        eigvals = torch.linalg.eigvalsh(H)
        
        # Apply bulk cut (Wigner semicircle edge avoidance)
        R = torch.sqrt(torch.tensor(2.0 * n_size, device=device))
        eigvals_cut = eigvals[(eigvals > -0.8 * R) & (eigvals < 0.8 * R)]
        
        # Unfold the spectrum using the Wigner Cumulative Distribution Function
        unfolded = R * torch.asin(eigvals_cut / R)
        
        # Compute nearest-neighbor spacings and normalize mean spacing to 1.0
        spacings = torch.diff(unfolded)
        spacings = spacings / torch.mean(spacings)
        
        # Compute 3-body correlations
        std_s = torch.std(spacings)
        s_minus_1 = spacings - 1.0
        c3 = s_minus_1[:-2] * s_minus_1[1:-1] * s_minus_1[2:]
        c3 = c3 / (std_s ** 3)
        
        c3_all.append(c3.cpu().numpy())
        
        if (i + 1) % 2000 == 0:
            print(f"  Progress: {i + 1} / {n_matrices} matrices processed.")

    c3_final = np.concatenate(c3_all)
    elapsed_time = time.time() - start_time
    print(f"GUE generation completed in {elapsed_time:.2f} seconds. ({len(c3_final):,} C_3 elements generated).")
    return c3_final

def download_and_process_riemann_ood(device='cpu'):
    """
    Downloads high-altitude Riemann zeros (T ~ 10^20) from Odlyzko's datasets
    and computes the corresponding 3-body correlations.
    
    Parameters:
    - device (str): Computational device ('cpu' or 'cuda').
    
    Returns:
    - np.ndarray: Array of C_3 correlation values for the Riemann zeros.
    """
    print("Downloading high-altitude Riemann zeros (Odlyzko dataset 'zeros6')...")
    url_ood = "http://www.dtc.umn.edu/~odlyzko/zeta_tables/zeros6"
    
    try:
        response = urllib.request.urlopen(url_ood)
        data = response.read().decode('utf-8').split()
    except Exception as e:
        print(f"Error downloading data: {e}")
        return np.array([])
    
    zeros_ood = np.array([float(val) for val in data])
    print(f"Successfully downloaded {len(zeros_ood):,} zeros.")
    
    print("Unfolding spectrum and computing 3-body correlations...")
    gamma = torch.tensor(zeros_ood, dtype=torch.float64, device=device)
    
    # Analytical unfolding using the leading term of the Riemann-von Mangoldt formula
    x = (gamma / (2.0 * np.pi)) * torch.log(gamma / (2.0 * np.pi * np.e))
    
    # Compute nearest-neighbor spacings and normalize
    spacings = torch.diff(x)
    spacings = spacings / torch.mean(spacings)
    
    # Compute 3-body correlations
    std_s = torch.std(spacings)
    s_minus_1 = spacings - 1.0
    c3 = s_minus_1[:-2] * s_minus_1[1:-1] * s_minus_1[2:]
    c3 = c3 / (std_s ** 3)
    
    print(f"Riemann processing completed. ({len(c3):,} C_3 elements generated).")
    return c3.cpu().numpy()

if __name__ == "__main__":
    # Hardware acceleration detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using computational device: {device.type.upper()}")
    
    # Generate datasets (reduced matrix count for local testing)
    gue_data = generate_gue_c3(n_matrices=5000, n_size=1000, device=device)
    riemann_data = download_and_process_riemann_ood(device=device)
    
    # Persist data to disk for subsequent model training
    np.save("gue_c3_data.npy", gue_data)
    np.save("riemann_c3_ood_data.npy", riemann_data)
    print("Datasets successfully saved as .npy files. Ready for training phase.")