import urllib.request
import torch
import numpy as np
import time

def generate_gue_c3(n_matrices=10000, n_size=1000, device='cpu'):
    """Génère les corrélations à 3 corps pour le chaos quantique (GUE)."""
    print(f"⚛️ Génération de {n_matrices} matrices GUE sur {device}...")
    start_time = time.time()
    c3_all = []
    
    for i in range(n_matrices):
        A = torch.randn(n_size, n_size, dtype=torch.complex64, device=device)
        H = (A + A.mH) / 2
        eigvals = torch.linalg.eigvalsh(H)
        
        R = torch.sqrt(torch.tensor(2.0 * n_size, device=device))
        eigvals_cut = eigvals[(eigvals > -0.8 * R) & (eigvals < 0.8 * R)]
        unfolded = R * torch.asin(eigvals_cut / R)
        
        spacings = torch.diff(unfolded)
        spacings = spacings / torch.mean(spacings)
        
        std_s = torch.std(spacings)
        s_minus_1 = spacings - 1.0
        c3 = s_minus_1[:-2] * s_minus_1[1:-1] * s_minus_1[2:]
        c3 = c3 / (std_s ** 3)
        
        c3_all.append(c3.cpu().numpy())
        
        if (i+1) % 2000 == 0:
            print(f"   Progression : {i+1} / {n_matrices}...")

    c3_final = np.concatenate(c3_all)
    print(f"✅ GUE terminé en {time.time() - start_time:.2f}s. ({len(c3_final):,} C3 générés)")
    return c3_final

def download_and_process_riemann_ood(device='cpu'):
    """Télécharge les zéros à haute altitude (10^20) et calcule les C3."""
    print("🌐 Téléchargement des zéros de Riemann OOD (Odlyzko zeros6)...")
    url_ood = "http://www.dtc.umn.edu/~odlyzko/zeta_tables/zeros6"
    response = urllib.request.urlopen(url_ood)
    data = response.read().decode('utf-8').split()
    
    zeros_ood = np.array([float(val) for val in data[:100000]])
    print(f"✅ {len(zeros_ood):,} zéros téléchargés.")
    
    print("⚙️ Dépliage et calcul des C3...")
    gamma = torch.tensor(zeros_ood, dtype=torch.float64, device=device)
    x = (gamma / (2 * np.pi)) * torch.log(gamma / (2 * np.pi * np.e))
    
    spacings = torch.diff(x)
    spacings = spacings / torch.mean(spacings)
    
    std_s = torch.std(spacings)
    s_minus_1 = spacings - 1.0
    c3 = s_minus_1[:-2] * s_minus_1[1:-1] * s_minus_1[2:]
    c3 = c3 / (std_s ** 3)
    
    print(f"✅ Riemann terminé. ({len(c3):,} C3 générés)")
    return c3.cpu().numpy()

if __name__ == "__main__":
    # Détection automatique du GPU 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Exécution des fonctions
    gue_data = generate_gue_c3(n_matrices=1000, n_size=1000, device=device) # Test réduit à 1000 pour aller vite en local
    riemann_data = download_and_process_riemann_ood(device=device)
    
    # Sauvegarde des données pour le prochain script (le modèle)
    np.save("gue_c3_data.npy", gue_data)
    np.save("riemann_c3_ood_data.npy", riemann_data)
    print("💾 Données sauvegardées avec succès (fichiers .npy). Prêt pour l'entraînement !")