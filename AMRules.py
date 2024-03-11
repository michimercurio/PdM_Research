from river import drift
from river import rules
from river import tree
import pickle as pkl
import numpy as np
from river import imblearn


# analog_sensors = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs','Oil_temperature', 'Flowmeter', 'Motor_current']

analog_sensors = ['nominalPower','temperature','txForwardPower','txReflectedPower','amp_1_ForwardPower','amp_2_ForwardPower','amp_1_TotalCurrent',
                    'amp_2_TotalCurrent','amp_1_Temperature','amp_2_Temperature',  'pressure','freqPumps','freqFans','liquidFlow']

with open("best_model/final_chunks_complete_losses_AE_lstm_ae_analog_feats_64_5_50_0.0001_0.001_64.pkl", "rb") as loss_file:
    tl = pkl.load(loss_file)
    test_losses = tl["test"]

with open('pompe_dati/test_chunk_non_scaled.pkl', "rb") as pklfile:
    data = pkl.load(pklfile)

modelCheb = (
    imblearn.ChebyshevOverSampler(
        regressor=rules.AMRules(
             delta=0.05,
             n_min=100,
             pred_type = 'mean',
             drift_detector=drift.ADWIN()
        )
    )
)

# prima anomalia WAE-GAN
# start_anomaly = 4383
# end_anomaly = 4900

# seconda anomalia WAE-GAN
# start_anomaly = 10298
# end_anomaly = 11230

# prima anomalia pompe
# start_anomaly = 5676
# end_anomaly = 8790

# seconda anomalia pompe
# start_anomaly = 25876
# end_anomaly = 31135

start_anomaly = 0
end_anomaly = -1

anoms = data[start_anomaly:end_anomaly, -1, :]
ruless = []
for anom, y in zip(anoms, test_losses[start_anomaly:end_anomaly]):
    x = {n: float(v.numpy()) for v,n in zip(anom,analog_sensors)}
    modelCheb.learn_one(x=x, y=y)
    debug = modelCheb.regressor.debug_one(x)
    #debug = model.debug_one(x).split('\n')
    #start = debug.split('\n').index('Default rule triggered:')
    ruless.append({"debug": debug, "reconstruction_error": y, "Support": modelCheb.regressor.anomaly_score(x)[2]})

set([rls for i, rlss in enumerate(ruless[4383:4900]) for rls in rlss['debug'].split('\n') if rls.startswith('Rule')])

# start_anomaly = 4383
# end_anomaly = 4900
# previous_rule = None
# rules_occ = {}
# n_occ = 0
# rule_occ = 0
# for idx, anom in enumerate(ruless[start_anomaly:end_anomaly]):
#     if anom["Support"]<1:
#         actual_rule = anom['debug'].split('\n')[0]
#         print(actual_rule)
#         if previous_rule == actual_rule:
#             rule_occ +=1
#         elif n_occ in rules_occ.keys():
#             rules_occ.update({n_occ: {'rule': actual_rule, 'start': rules_occ[n_occ]['start'], 'number': rule_occ+1, 'end': idx-1}})
#             previous_rule = actual_rule
#             n_occ+=1
#             rule_occ = 0
#             rules_occ.update({n_occ: {'rule':actual_rule, 'start': idx}})
#         else:
#             rules_occ.update({n_occ: {'rule':actual_rule, 'start': idx}})
#             previous_rule = actual_rule


print('Done')