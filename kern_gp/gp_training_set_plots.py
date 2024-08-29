import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from kern_gp_matrices import noiseless_predict

def get_fingerprint(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return AllChem.GetMorganFingerprint(mol, radius=3, useCounts=True)


def independent_tanimoto_gp_predict(
    query_smiles, known_smiles, known_Y, gp_means, gp_amplitudes, gp_noises
):
    known_fp = [get_fingerprint(s) for s in known_smiles]
    query_fp = [get_fingerprint(s) for s in query_smiles]
    K_known_known = np.asarray([DataStructs.BulkTanimotoSimilarity(fp, known_fp) for fp in known_fp])
    K_query_known = np.asarray([DataStructs.BulkTanimotoSimilarity(fp, known_fp) for fp in query_fp])
    K_query_query_diagonal = np.asarray([DataStructs.TanimotoSimilarity(fp, fp) for fp in query_fp])

    means_out = []
    vars_out = []
    for j in range(known_Y.shape[1]):  # iterate over all objectives
        residual_j = known_Y[:, j] - gp_means[j]
        mu_j, var_j = noiseless_predict(
            a=gp_amplitudes[j],
            s=gp_noises[j],
            k_train_train=K_known_known,
            k_test_train=K_query_known,
            k_test_test=K_query_query_diagonal,
            y_train=residual_j,
            full_covar=False,
        )
        means_out.append(mu_j + gp_means[j])
        vars_out.append(var_j)

    return np.asarray(means_out).T, np.asarray(vars_out).T


def generate_synthetic_data(n_points):
    # This function generates synthetic SMILES strings and corresponding properties
    smiles = ["C" * i for i in range(1, n_points + 1)]  # Dummy SMILES strings
    y = np.sin(np.linspace(0, 3 * np.pi, n_points)) + np.random.normal(0, 0.1, n_points)  # Sine wave data with noise
    return smiles, y.reshape(-1, 1)  # Reshape y to be Nx1


def plot_gp_with_confidence_intervals(train_sizes, test_smiles, test_x, gp_means, gp_amplitudes, gp_noises):
    plt.figure(figsize=(15, 5))

    for idx, train_size in enumerate(train_sizes):
        plt.subplot(1, len(train_sizes), idx + 1)
        train_smiles, train_y = generate_synthetic_data(train_size)

        pred_means, pred_vars = independent_tanimoto_gp_predict(
            query_smiles=test_smiles,
            known_smiles=train_smiles,
            known_Y=train_y,
            gp_means=gp_means,
            gp_amplitudes=gp_amplitudes,
            gp_noises=gp_noises
        )

        pred_means = pred_means.flatten()
        std_devs = np.sqrt(pred_vars).flatten()

        # Plot predictions
        plt.plot(test_x, pred_means, 'b-', label='Predicted mean')
        plt.fill_between(test_x, pred_means - 1.96 * std_devs, pred_means + 1.96 * std_devs, color='green', alpha=0.2, label='95% confidence interval')

        # Plot training data
        train_x = np.linspace(0, 10, train_size)
        plt.scatter(train_x, train_y, c='red', label='Training data')

        plt.title(f"GP Regression with {train_size} training examples")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_sizes = [10, 20, 40]  # Number of training points in each plot
    test_x = np.linspace(0, 10, 100)
    test_smiles = ["C" * i for i in range(1, 101)]  # Dummy SMILES strings for test data
    print(test_smiles)

    gp_means = np.array([0.0])  # Assuming 1D outputs
    gp_amplitudes = np.array([1.0])
    gp_noises = np.array([1e-4])

    plot_gp_with_confidence_intervals(train_sizes, test_smiles, test_x, gp_means, gp_amplitudes, gp_noises)
