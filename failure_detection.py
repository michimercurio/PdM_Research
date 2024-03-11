import numpy as np
import pickle as pkl
from operator import itemgetter
from itertools import groupby
import pandas as pd


def extreme_anomaly(dist):
    q25, q75 = np.quantile(dist, [0.25, 0.75])
    return q75 + 3*(q75-q25)

def simple_lowpass_filter(arr, alpha):
    y = arr[0]
    filtered_arr = [y]
    for elem in arr[1:]:
        y = y + alpha * (elem - y)
        filtered_arr.append(y)
    return filtered_arr


def detect_failures(anom_indices):
    failure_list = []
    failure = set()
    for i in range(len(anom_indices) - 1):
        if anom_indices[i] == 1 and anom_indices[i + 1] == 1:
            failure.add(i)
            failure.add(i + 1)
        elif len(failure) > 0:
            failure_list.append(failure)
            failure = set()

    if len(failure) > 0:
        failure_list.append(failure)

    return failure_list


def failure_list_to_interval(cycle_dates, failures):
    failure_intervals = []
    for failure in failures:
        failure = sorted(failure)
        failure_intervals.append(pd.Interval(cycle_dates[failure[0]][0], cycle_dates[failure[-1]][1], closed="both"))
    return failure_intervals


def collate_intervals(interval_list):
    diff_consecutive_intervals = [(interval_list[i+1].left - interval_list[i].right).days for i in range(len(interval_list)-1)]
    lt_1day = np.where(np.array(diff_consecutive_intervals) <= 1)[0]
    collated_intervals = []
    for k, g in groupby(enumerate(lt_1day), lambda ix: ix[0]-ix[1]):
        collated = list(map(itemgetter(1), g))
        collated_intervals.append(pd.Interval(interval_list[collated[0]].left, interval_list[collated[-1]+1].right, closed="both"))

    collated_intervals.extend([interval_list[i] for i in range(len(interval_list)) if i not in lt_1day and i-1 not in lt_1day])
    return sorted(collated_intervals)


def print_failures(cycle_dates, output):
    failures = detect_failures(output)
    failure_intervals = failure_list_to_interval(cycle_dates, failures)
    collated_intervals = collate_intervals(failure_intervals)
    for interval in collated_intervals:
        print(interval)


##### Results from the main paper #####

def generate_intervals(granularity, start_timestamp, end_timestamp):
    current_timestamp = start_timestamp
    interval_length = pd.offsets.DateOffset(**granularity)
    interval_list = []
    while current_timestamp < end_timestamp:
        interval_list.append(pd.Interval(current_timestamp, current_timestamp + interval_length, closed="left"))
        current_timestamp = current_timestamp + interval_length
    return interval_list


with open("pompe_dati/training_chunk_dates_mio.pkl", "rb") as chunk_dates_file:
    training_chunk_dates = pkl.load(chunk_dates_file)

with open("pompe_dati/test_completo/test_chunk_dates_mio_completo.pkl", "rb") as chunk_dates_file:
    test_chunk_dates = pkl.load(chunk_dates_file)

#
train_intervals = generate_intervals({"minutes": 5}, pd.Timestamp(training_chunk_dates[0][0]), pd.Timestamp(training_chunk_dates[-1][0]))
test_intervals = generate_intervals({"minutes": 5}, pd.Timestamp(test_chunk_dates[0][0]), pd.Timestamp(test_chunk_dates[-1][0]))




# def map_cycles_to_intervals(interval_list, chunk_dates):
#     cycles_dates = list(map(lambda x: pd.Interval(pd.Timestamp(x[0]), pd.Timestamp(x[1]), closed="both"), chunk_dates))
#     return list(map(lambda x: np.where([x.overlaps(i) for i in cycles_dates])[0], interval_list))


# train_chunks_to_intervals = map_cycles_to_intervals(train_intervals, training_chunk_dates)
# test_chunks_to_intervals = map_cycles_to_intervals(test_intervals, test_chunk_dates)

# with open("pompe_dati/train_chunks_to_intervals_mio.pkl", "wb") as pklfile:
#     pkl.dump(train_chunks_to_intervals, pklfile)
# with open("pompe_dati/test_completo/test_chunks_to_intervals_mio_completo.pkl", "wb") as pklfile:
#     pkl.dump(test_chunks_to_intervals, pklfile)

with open("pompe_dati/train_chunks_to_intervals_mio.pkl", "rb") as chunk_dates_file:
    train_chunks_to_intervals = pkl.load(chunk_dates_file)
with open("pompe_dati/test_completo/test_chunks_to_intervals_mio_completo.pkl", "rb") as chunk_dates_file:
    test_chunks_to_intervals = pkl.load(chunk_dates_file)
alpha = 0.05


date_output_test = [interval.left for i, interval in enumerate(test_intervals) if len(test_chunks_to_intervals[i]) > 0]
date_output_train = [interval.left for i, interval in enumerate(train_intervals) if
                     len(train_chunks_to_intervals[i]) > 0]

with open("best_model/final_chunks_complete_losses_AE_lstm_ae_analog_feats_64_5_50_0.0001_0.001_64.pkl", "rb") as loss_file:
    tl = pkl.load(loss_file)
    test_losses = tl["test"]
    train_losses = tl["train"]

median_train_losses_lstm = np.array([np.median(np.array(train_losses)[tc]) for tc in train_chunks_to_intervals if len(tc) > 0])
median_test_losses_lstm = np.array([np.median(np.array(test_losses)[tc]) for tc in test_chunks_to_intervals if len(tc) > 0])

date_output_test = [interval.left for i, interval in enumerate(test_intervals) if len(test_chunks_to_intervals[i]) > 0]
date_output_train = [interval.left for i, interval in enumerate(train_intervals) if
                     len(train_chunks_to_intervals[i]) > 0]

anomaly_threshold_lstm = extreme_anomaly(median_train_losses_lstm)

binary_output_lstm = np.array(np.array(median_test_losses_lstm) > anomaly_threshold_lstm, dtype=int)

lstm_output_lstm = np.array(simple_lowpass_filter(binary_output_lstm, 0.05))


# lstm_mask = np.array(simple_lowpass_filter(median_test_losses_lstm, 0.05)) > 0.265
# df = pd.read_csv("tabella_label_per_risultati.csv")
# # df.loc[:, "anomaly_pred"] = lstm_mask.astype(int)
# df.to_csv("tabs/tabs3.csv", index=False)


# #df = pd.DataFrame({"timestamp": date_output_test, "anomaly_pred": lstm_mask.astype(int), "anomaly_true": 0})
# df.loc[(df['timestamp'] >= '2023-09-29') & (df['timestamp'] < '2023-09-30'), "anomaly_true"] = 1
# df.to_csv("tabella_label_per_risultati.csv", index=False)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

daily_count = pd.read_csv('Dati/daily_count.csv')

# Converti la colonna 'timestamp' in formato datetime
daily_count['timestamp'] = pd.to_datetime(daily_count['timestamp'])

# Ordina il DataFrame per timestamp
daily_count = daily_count.sort_values(by='timestamp')

df = pd.read_csv("Dati/dataset.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"],format="%Y-%m-%d %H:%M:%S.%f")
# filtered_df = df[(df['timestamp'] <= "2023-06-30T00:00:00.000000000") & (df['timestamp'] >= "2023-05-04T00:00:00.000000000")] #maggio/giugno
# filtered_df = df[(df['timestamp'] <= "2023-09-30T00:00:00.000000000") & (df['timestamp'] >= "2023-08-01T00:00:00.000000000")] #agosto/settembre
filtered_df = df[(df['timestamp'] <= "2023-12-31T00:00:00.000000000") & (df['timestamp'] >= "2023-05-04T00:00:00.000000000")] #completo
numerical_colums = ['timestamp', 'nominalPower','temperature','txForwardPower','txReflectedPower','amp_1_ForwardPower','amp_2_ForwardPower','amp_1_TotalCurrent', 'amp_2_TotalCurrent','amp_1_Temperature','amp_2_Temperature',  'pressure','freqPumps','freqFans','liquidFlow']
filtered_df = filtered_df[numerical_colums]

# Converti la colonna 'timestamp' in formato datetime, se non è già nel formato corretto
filtered_df['timestamp'] = pd.to_datetime(df['timestamp'])

lstm_output_lstm = np.array(simple_lowpass_filter(median_test_losses_lstm, 0.05))
plt.style.use('ggplot')
dates = [datetime.strptime(str(dt_string), "%Y-%m-%d %H:%M:%S.%f") for dt_string in date_output_test]
df2 = pd.DataFrame(np.array(lstm_output_lstm), index=dates)

# Calcola i count in percentuale rispetto al massimo
max_count = 17280
daily_count['count_percent'] = (daily_count['count'] / max_count) * 100

#TRE SUBPLOT DIFFERENTI
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Traccia delle median_test_losses su ax1
ax1.plot(df2.index, df2[0], color='g', label='LSTM Losses')
ax1.set_ylabel('LSTM Losses', color='g')
ax1.tick_params('y', colors='g')
ax1.legend()

# Definisci gli intervalli da colorare
intervals_to_color = [
    ('2023-07-10 18:55', '2023-07-24 17:15'),
    ('2023-09-22 08:54', '2023-10-10 10:30'),
]

# Colora gli intervalli specificati
for start, end in intervals_to_color:
    ax1.axvspan(start, end, facecolor='0.8', color=[0.5,0.5,0.5])

# Aggiungi la linea orizzontale per l'anomalia sul subplot ax1
ax1.axhline(0.32, color='r', linestyle='--')

# Traccia di txForwardPower su ax2
ax2.plot(filtered_df['timestamp'], filtered_df['txForwardPower'], color='b', label='txForwardPower')
ax2.set_ylabel('txForwardPower', color='b')
ax2.tick_params('y', colors='b')
ax2.legend()

# Tracciamento dei count su ax3
# filtered_daily_count = daily_count[(daily_count['timestamp'] <= "2023-06-30T00:00:00.000000000") & (daily_count['timestamp'] >= "2023-05-04T00:00:00.000000000")] #maggio/giugno
# filtered_daily_count = daily_count[(daily_count['timestamp'] <= "2023-09-30T00:00:00.000000000") & (daily_count['timestamp'] >= "2023-08-01T00:00:00.000000000")] #agosto/settembre
filtered_daily_count = daily_count[(daily_count['timestamp'] <= "2023-12-31T00:00:00.000000000") & (daily_count['timestamp'] >= "2023-05-04T00:00:00.000000000")] #completo
ax3.plot(filtered_daily_count['timestamp'], filtered_daily_count['count_percent'], color='purple', label='Count')
ax3.set_ylabel('Samples Count %', color='purple')
ax3.tick_params('y', colors='purple')
ax3.legend()

# Impostazione delle etichette degli assi x
ax3.set_xlabel('Timestamp')

# Imposta il titolo generale
plt.suptitle('LSTM Losses and txForwardPower Over Time')

plt.show()