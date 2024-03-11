import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Carica i dati da tutti i file CSV
dfs = []
for i in range(1, 19):  # Considera che hai 18 file CSV
    filename = f"tabs/tabs{i}.csv"  # Assumi che i file si chiamino data_1.csv, data_2.csv, ecc.
    df = pd.read_csv(filename)
    dfs.append(df)

# Crea un nuovo plot
plt.figure(figsize=(10, 8))

# Lista per memorizzare i valori di AUC
auc_scores = []

# Per ciascun DataFrame
for i, df in enumerate(dfs):
    # Calcola i tassi di falsi positivi e i tassi di veri positivi
    fpr, tpr, _ = roc_curve(df['anomaly_true'], df['anomaly_pred'])
    # Calcola l'area sotto la curva ROC (AUC)
    roc_auc = auc(fpr, tpr)
    auc_scores.append(roc_auc)  # Memorizza l'AUC
    # Disegna la ROC curve
    plt.plot(fpr, tpr, label=f'ROC curve {i+1} (AUC = {roc_auc:.2f})')

# Aggiungi una linea tratteggiata per la ROC curve random (AUC = 0.5)
plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random ROC curve (AUC = 0.5)')

# Aggiungi una legenda al plot
plt.legend(loc="lower right")

# Aggiungi etichette agli assi e un titolo
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')

# Trova l'indice del modello con il massimo AUC
best_model_index = auc_scores.index(max(auc_scores))
print(f"Il modello migliore è il {best_model_index + 1} con AUC = {max(auc_scores):.2f}")

# Mostra il plot
plt.show()