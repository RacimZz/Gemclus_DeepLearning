# Imports des librairies nécessaires pour le traitement, l'analyse et la visualisation
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

# Désactive les optimisations ONEDNN de TensorFlow pour des raisons de compatibilité
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Fonction utilitaire pour sérialiser les objets numpy dans les fichiers JSON
def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

# ----------------- Paramètres globaux à configurer -----------------
kernel = "linear"                # Type de noyau utilisé dans MLP-MMD
n_clusters_list = [50]           # Liste des nombres de clusters à tester
lr = 0.001                       # Taux d'apprentissage du modèle
hid = 150                        # Nombre de neurones cachés
results = []                     # Liste pour stocker les résultats
n = 50                           # Nombre de clusters (variable pour le nommage)
output_path = f"{kernel}_{lr}_{hid}_{n}.jsonl"
# On vide le fichier de sortie au démarrage
with open(output_path, "w") as f:
    pass

# ----------------- Boucle principale d'expériences -----------------
for sample_size in [68232]:  # Possibilité de tester plusieurs tailles d'échantillons

    print(colored(f"\nÉchantillon de {sample_size} spectres normalisé", "cyan"))

    for n_clusters in n_clusters_list:
        data_path = "../../Data/flux_pretreated_NGC1068_6400_6800_high_velocity_normalise_redshift_filter_subset.csv"
        X_raw = pd.read_csv(data_path, delimiter=",").to_numpy()

        # Standardisation des données (centrage-réduction)
        scaler = StandardScaler()
        X_raw = scaler.fit_transform(X_raw)

        # Échantillonnage aléatoire des spectres
        np.random.seed(42)
        indices = np.random.choice(X_raw.shape[0], size=sample_size, replace=False)
        X = X_raw[indices]
        X_raw = []  # Libération mémoire

        gc.collect()  # Nettoyage mémoire

        print(colored("="*70, "blue"))
        print(colored(f"Test: n_clusters={n_clusters}, sample={sample_size}, kernel={kernel}, lr={lr}", "yellow"))
        start = time.time()

        try:
            # Création et entrainement du modèle Gemini (MLP-MMD)
            model = mlp.MLPMMD(
                n_clusters=n_clusters,
                kernel=kernel,
                random_state=0,
                ovo=True,
                learning_rate=lr,
                n_hidden_dim=hid
            )
            y_pred = model.fit_predict(X)
            duration = time.time() - start

            # Évaluation des partitions obtenues
            n_labels = len(set(y_pred))
            sil_score = silhouette_score(X, y_pred) if n_labels > 1 else np.nan
            ch_score = calinski_harabasz_score(X, y_pred) if n_labels > 1 else np.nan
            db_score = davies_bouldin_score(X, y_pred) if n_labels > 1 else np.nan
            score = model.score(X)
            n_iter = model.n_iter_

            # Agrégation des résultats
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
                "n_iter": n_iter,
                "hidden_dim": hid,
                "ov": True
            }
            results.append(result)

            # Enregistrement en JSONL (ligne par expérience)
            with open(output_path, "a") as f:
                f.write(json.dumps(result, default=convert_numpy) + "\n")

            # Sauvegarde des labels de clusters dans un CSV
            df_labels = pd.DataFrame({
                "index": indices,
                "cluster": y_pred
            })
            label_output_path = f"labels_sample{sample_size}_clusters{n_clusters}_kernel{kernel}_lr{lr}_hid{hid}_standardise_.csv"
            df_labels.to_csv(label_output_path, index=False)
            print(colored(f"Labels sauvegardés dans : {label_output_path}", "magenta"))

            # Logs des scores et métriques
            print(colored(f"Terminé en {duration:.2f}s", "green"))
            print(f"n_labels: {n_labels} | Silhouette: {sil_score:.4f} | CH: {ch_score:.2f} | DB: {db_score:.2f}")
            print(f"Score: {score:.2f} | Iterations: {n_iter}")

        except Exception as e:
            print(colored("Erreur pendant l’exécution :", "red"))
            print(e)

        print(colored("="*70, "blue"))
