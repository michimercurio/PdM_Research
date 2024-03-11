import pickle as pkl
import numpy as np
import torch as th
from sklearn.preprocessing import StandardScaler
import pandas as pd
import sys

def generate_chunks(df, chunk_size, chunk_stride, cols):

    from numpy.lib.stride_tricks import sliding_window_view

    gaps = list((g := df.timestamp.diff().gt(pd.Timedelta(minutes=1)))[g].index)
    c = []
    window_start_date = []
    start = 0
    for gap in gaps:
        tdf = df.iloc[start:gap, :]
        if len(tdf) < chunk_size:
            start = gap
            continue
        vals = tdf[cols].values
        sliding_vals = sliding_window_view(vals, (chunk_size, len(cols))).squeeze(1)[::chunk_stride, :, :]
        window_start_date.append(sliding_window_view(tdf.timestamp.values, chunk_size)[::chunk_stride,[0,-1]])
        c.append(sliding_vals)
        start = gap
    tdf = df.iloc[start:, :]
    if len(tdf) >= chunk_size:
        vals = tdf[cols].values
        sliding_vals = sliding_window_view(vals, (chunk_size, len(cols))).squeeze(1)[::chunk_stride, :, :]
        c.append(sliding_vals)
        window_start_date.append(sliding_window_view(tdf.timestamp.values, chunk_size)[::chunk_stride,[0,-1]])

    c = np.concatenate(c)
    return c, np.concatenate(window_start_date)


pompe_dataset = pd.read_csv("Dati/dataset.csv")
pompe_dataset["timestamp"] = pd.to_datetime(pompe_dataset["timestamp"],format="%Y-%m-%d %H:%M:%S.%f")
pompe_dataset = pompe_dataset.sort_values("timestamp")
pompe_dataset.reset_index(drop=True, inplace=True)

categorical_colums = ['txWorkingState','txStatus','txLowPowerWrn','txLowPowerAlm','txReflPowerWrn','txReflPowerAlm','coolingSystemStatus',
                      'temperatureStatus','pumpsStatus','fanStatus','pressureStatus','flowStatus','amp_1_ConnStatus','amp_2_ConnStatus','amp_1_WorkingState',
                      'amp_2_WorkingState','amp_1_RfStatus','amp_2_RfStatus','amp_1_FanWarning','amp_2_FanWarning','amp_1_PSUAlarm','amp_2_PSUAlarm',
                      'amp_1_PSUWarning','amp_2_PSUWarning','amp_1_ReflectedAlm','amp_2_ReflectedAlm','amp_1_TempAlm','amp_2_TempAlm','amp_1_TempWrn','amp_2_TempWrn']

numerical_colums = ['nominalPower','temperature','txForwardPower','txReflectedPower','amp_1_ForwardPower','amp_2_ForwardPower','amp_1_TotalCurrent',
                    'amp_2_TotalCurrent','amp_1_Temperature','amp_2_Temperature',  'pressure','freqPumps','freqFans','liquidFlow']

print("Read dataset")

# Normalizzazione separata per la feature txForwardPower
# scaler_dataset = StandardScaler()
# pompe_dataset[numerical_colums] = scaler_dataset.fit_transform(pompe_dataset[numerical_colums])

chunks, chunk_dates = generate_chunks(pompe_dataset, 360, 12, numerical_colums)

print("Calculated chunks")

scaler = StandardScaler()
scaled_chunks = np.array(list(map(lambda x: scaler.fit_transform(x), chunks)))

print("Finished scaling")

# training_chunks = th.tensor(chunks[np.where((chunk_dates[:, 1] >= np.datetime64("2023-04-21T00:00:00.000000000")) & (chunk_dates[:, 1] <= np.datetime64("2023-05-03T00:00:00.000000000")))[0]])
test_chunks = th.tensor(chunks[np.where((chunk_dates[:, 1] >= np.datetime64("2023-05-04T00:00:00.000000000")) & (chunk_dates[:, 1] < np.datetime64("2024-01-01T00:00:00.000000000")))[0]])
print("Separated into training and test")

# training_chunk_dates = chunk_dates[np.where((chunk_dates[:, 1] >= np.datetime64("2023-04-21T00:00:00.000000000")) & (chunk_dates[:, 1] <= np.datetime64("2023-05-03T00:00:00.000000000")))[0]]
test_chunk_dates = chunk_dates[np.where((chunk_dates[:, 1] >= np.datetime64("2023-05-04T00:00:00.000000000")) & (chunk_dates[:, 1] < np.datetime64("2024-01-01T00:00:00.000000000")))[0]]

# with open("pompe_dati/training_chunk_dates_mio.pkl", "wb") as pklfile:
#     pkl.dump(training_chunk_dates, pklfile)

with open("pompe_dati/test_chunk_dates_mio_completo.pkl", "wb") as pklfile:
    pkl.dump(test_chunk_dates, pklfile)

# with open("pompe_dati/training_chunks_mio.pkl", "wb") as pklfile:
#     pkl.dump(training_chunks, pklfile)

with open("pompe_dati/test_chunks_mio_completo.pkl", "wb") as pklfile:
    pkl.dump(test_chunks, pklfile)

print("Finished saving")
