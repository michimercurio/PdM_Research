import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, recall_score, roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# Carica il CSV con i risultati del modello
df = pd.read_csv("tabs/tabs16.csv")

# Calcola F1 score
f1 = f1_score(df['anomaly_true'], df['anomaly_pred'])

# Calcola recall
recall = recall_score(df['anomaly_true'], df['anomaly_pred'])

# Calcola ROC curve e AUC
fpr, tpr, _ = roc_curve(df['anomaly_true'], df['anomaly_pred'])
roc_auc = roc_auc_score(df['anomaly_true'], df['anomaly_pred'])

# Calcola precision-recall curve e AUC
precision, recall, _ = precision_recall_curve(df['anomaly_true'], df['anomaly_pred'])
pr_auc = average_precision_score(df['anomaly_true'], df['anomaly_pred'])

# Calcola geometric mean
g_mean = np.sqrt(tpr * (1 - fpr))

# Calcola la matrice di confusione
cm = confusion_matrix(df['anomaly_true'], df['anomaly_pred'])

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Plot Precision-Recall curve
plt.figure()
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve: AP={0:0.2f}'.format(pr_auc))
plt.show()

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix")
plt.show()

print(f"F1 Score: {f1}")
print(f"Recall: {recall}")
print(f"ROC-AUC: {roc_auc}")
print(f"PR-AUC: {pr_auc}")
print(f"Geometric Mean: {g_mean}")

ConfusionMatrixDisplay.from_predictions(df['anomaly_true'], df['anomaly_pred'], cmap="Blues", values_format='d')
plt.show()