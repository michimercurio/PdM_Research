import os

epochs = 50
lrs = [0.001, 0.0001]
emb_size = [7,32,64]



n_layers = [2, 5, 10]
hidden_units = [30] #i modelli saranno trainati solo con 30 hidden

model = "lstm_ae"

base_string = lambda n_layers, hidden_units, emb_size, lr: f"python train_chunks.py -model {model} -embedding {emb_size} -epochs {epochs} -lr {lr} -batch_size 64 \
-n_layers {n_layers} -hidden {hidden_units} "

for lr in lrs:
   for emb in emb_size:
       for n_l in n_layers:
           for hid in hidden_units:
                os.system(base_string(n_l, hid, emb, lr))