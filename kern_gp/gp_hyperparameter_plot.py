import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from kern_gp_matrices import noiseless_predict

# counts -- minmax kernel
def get_fingerprint(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return AllChem.GetMorganFingerprint(mol, radius=3, useCounts=True)

def independent_tanimoto_gp_predict(
    query_smiles, known_smiles, known_Y, gp_mean, gp_amplitude, gp_noise
):
    known_fp = [get_fingerprint(s) for s in known_smiles]
    query_fp = [get_fingerprint(s) for s in query_smiles]
    K_known_known = np.asarray([DataStructs.BulkTanimotoSimilarity(fp, known_fp) for fp in known_fp])
    K_query_known = np.asarray([DataStructs.BulkTanimotoSimilarity(fp, known_fp) for fp in query_fp])
    K_query_query_diagonal = np.asarray([DataStructs.TanimotoSimilarity(fp, fp) for fp in query_fp])

    residual = known_Y.flatten() - gp_mean
    mu, var = noiseless_predict(
        a=gp_amplitude,
        s=gp_noise,
        k_train_train=K_known_known,
        k_test_train=K_query_known,
        k_test_test=K_query_query_diagonal,
        y_train=residual,
        full_covar=False,
    )

    return mu + gp_mean, var

def generate_synthetic_data(n_points):
    # This function generates synthetic SMILES strings and corresponding properties
    smiles = ["C" * i for i in range(1, n_points + 1)]  # Dummy SMILES strings
    y = np.sin(np.linspace(0, 3 * np.pi, n_points)) + np.random.normal(0, 0.1, n_points)  # Sine wave data with noise
    return smiles, y.reshape(-1, 1)  # Reshape y to be Nx1

def plot_gp_with_confidence_intervals(test_smiles, test_x, train_size, gp_amplitudes, gp_mean, gp_noise):
    plt.figure(figsize=(10, 6))
    
    # Generate training data
    train_smiles, train_y = generate_synthetic_data(train_size)

    # Different colors for different GP amplitudes
    colors = ['magenta', 'pink', 'red']

    for i, gp_amplitude in enumerate(gp_amplitudes):
        pred_means, pred_vars = independent_tanimoto_gp_predict(
            query_smiles=test_smiles,
            known_smiles=train_smiles,
            known_Y=train_y,
            gp_mean=gp_mean,
            gp_amplitude=gp_amplitude,
            gp_noise=gp_noise
        )

        pred_means = pred_means.flatten()
        std_devs = np.sqrt(pred_vars).flatten()

        # Plot predictions
        plt.plot(test_x, pred_means, color=colors[i], label=f'Predicted mean (Amplitude={gp_amplitude})')
        plt.fill_between(test_x, pred_means - 1.96 * std_devs, pred_means + 1.96 * std_devs, 
                         color=colors[i], alpha=0.2, label=f'95% confidence interval (Amplitude={gp_amplitude})')

    # Plot training data
    train_x = np.linspace(0, 10, train_size)
    plt.scatter(train_x, train_y, c='black', label='Training data')

    plt.title(f"Characteristic Amplitudes of GP")
    plt.xlabel('x')
    plt.ylabel('y')
    
    # Move legend outside
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_x = np.linspace(0, 10, 100)
    test_smiles = ["C" * i for i in range(1, 101)]  # Dummy SMILES strings for test data

    train_size = 20  # Fixed number of training points
    gp_mean = 0.0  # Assuming 1D outputs
    gp_noise = 1e-4
    gp_amplitudes = [0.5, 1.0, 2.0]  # Different amplitudes to illustrate the effect

    plot_gp_with_confidence_intervals(test_smiles, test_x, train_size, gp_amplitudes, gp_mean, gp_noise)
