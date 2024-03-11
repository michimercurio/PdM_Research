import matplotlib.pyplot as plt
import pickle as pkl
import csv

with open("results_prove/final_chunks_offline_losses_AE_lstm_ae_analog_feats_4_5_150_0.001_0.001_64.pkl", "rb") as loss_file:
    losses = pkl.load(loss_file)

# Supponendo che 'losses' sia il dizionario contenente le perdite del modello
losses_encoder_decoder = losses['encoder/decoder'][:15]

# Creazione del grafico delle perdite
plt.plot(losses_encoder_decoder, label='LSTM loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('LSTM Loss Over Epochs')
plt.legend()
plt.grid(True)

# Visualizza il grafico
plt.show()

print('Done')