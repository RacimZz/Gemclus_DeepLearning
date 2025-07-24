import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from gemclus import mlp
import json
import os
import gc
from termcolor import colored

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Fonction pour rendre les objets JSON compatibles
def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

# --- Paramètres globaux ---
kernels = ["linear", "poly"]
n_clusters_list = [30, 50]
lr = 0.00001
hid_list = [10, 20, 50, 100, 150, 200, 250, 300]
n_iter_list = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
results = []

# Charger les données
data_path = "../GEMINI/Data/flux_pretreated_NGC1068_6400_6800_high_velocity_normalise_redshift_filter_subset.csv"
X_raw = pd.read_csv(data_path, delimiter=",").to_numpy()

# standardisation des données
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# Préparer dossier de résultats
os.makedirs("results", exist_ok=True)

output_path = "json/test_wasserstein.jsonl"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    pass  # Vide le fichier dès le début

for n_clusters in n_clusters_list:
    for kernel in kernels:
        for hid in hid_list:
            for n_iter in n_iter_list:
                print(colored(f"\n🔎 Test avec n_clusters={n_clusters}, kernel={kernel}, hid={hid}, n_iter={n_iter}", "cyan"))

                indices = np.arange(X.shape[0])
                sample_size = X.shape[0]
                print(colored("=" * 70, "blue"))
                print(colored(f"▶️ Test: n_clusters={n_clusters}, kernel={kernel}, lr={lr}", "yellow"))
                start = time.time()

                try:
                    model = mlp.MLPWasserstein(
                        n_clusters=n_clusters,
                        metric="euclidean",
                        random_state=0,
                        ovo=False,
                        learning_rate=lr,
                        n_hidden_dim=hid,
                        max_iter=n_iter
                    )
                    y_pred = model.fit_predict(X)
                    duration = time.time() - start

                    n_labels = len(set(y_pred))
                    sil_score = silhouette_score(X, y_pred) if n_labels > 1 else np.nan
                    ch_score = calinski_harabasz_score(X, y_pred) if n_labels > 1 else np.nan
                    db_score = davies_bouldin_score(X, y_pred) if n_labels > 1 else np.nan
                    score = model.score(X)
                    n_iter_final = model.n_iter_

                    result = {
                        "sample_size": sample_size,
                        "kernel": kernel,
                        "n_clusters": n_clusters,
                        "n_labels": n_labels,
                        "time": duration,
                        "silhouette": sil_score,
                        "calinski_harabasz": ch_score,
                        "davies_bouldin": db_score,
                        "learning_rate": lr,
                        "score": score,
                        "n_iter": n_iter_final,
                        "hidden_dim": hid,
                        "ov": False
                    }
                    results.append(result)

                    # Enregistrement JSONL
                    with open(output_path, "a") as f:
                        f.write(json.dumps(result, default=convert_numpy) + "\n")

                    # Enregistrement des labels
                    df_labels = pd.DataFrame({
                        "index": indices,
                        "cluster": y_pred
                    })
                    label_output_path = f"results/labels_wasserstein_sample{sample_size}_clusters{n_clusters}_kernel{kernel}_lr{lr}_hid{hid}"
                    df_labels.to_csv(label_output_path, index=False)
                    print(colored(f"📁 Labels sauvegardés dans : {label_output_path}", "magenta"))

                    # Logs
                    print(colored(f"✅ Terminé en {duration:.2f}s", "green"))
                    print(f"📊 n_labels: {n_labels} | Silhouette: {sil_score:.4f} | CH: {ch_score:.2f} | DB: {db_score:.2f}")
                    print(f"🧠 Score: {score:.2f} | Iterations: {n_iter_final}")

                except Exception as e:
                    print(colored("❌ Erreur pendant l’exécution :", "red"))
                    print(e)

                print(colored("="*70, "blue"))