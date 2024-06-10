import config as cfg
from generator import *
import torch
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from score import *

model_path = sys.argv[1]
# dataset_path = sys.argv[2]

class TestDataset(Dataset):
    def __init__(self, dataset_path):
        self.data = np.load(dataset_path)
        self.total_hours = self.data.shape[0]
        self.feature_count = self.data.shape[1]
        self.past_hours = 6
        self.future_hours = 6

    def __len__(self):
        return self.total_hours - self.past_hours - self.future_hours + 1

    def __getitem__(self, idx):
        X = self.data[(idx):(idx+self.past_hours),:,:,:]
        y = self.data[(idx+self.past_hours):(idx+self.past_hours+self.future_hours),0:1,:,:]
        y_aux = self.data[(idx+self.past_hours):(idx+self.past_hours+self.future_hours),6:7,:,:]
        return X, y, y_aux, idx


test_loader = get_test_loader()



model  = cfg.test_model_class(cfg.prev_hours, cfg.input_channels)
model.load_state_dict(torch.load(model_path))
model.float()
model.to(cfg.device)




model.eval()
with torch.no_grad():
    val_calparams_epoch = Cal_params_epoch()
    val_calparams_epoch_1_hour = Cal_params_epoch()
    val_calparams_epoch_3_hour = Cal_params_epoch()
    for i, (X, y, y_aux, idx) in enumerate(test_loader):
        X = X.float().to(cfg.device)
        y = y.float().to(cfg.device)
        # print("hi"+str(i))
        predicted_frames = model(X)
        # print(predicted_frames.shape)

        # output
        pod, far, ts, ets, micro_f1, macro_f1 = val_calparams_epoch.cal_batch(y, predicted_frames,cfg.do_sigmoid_after_model_prediction)
        pod_1_hour, far_1_hour, ts_1_hour, ets_1_hour, micro_f1_1_hour, macro_f1_1_hour = val_calparams_epoch_1_hour.cal_batch(y[:,0:1,:,:,:], predicted_frames[:,0:1,:,:,:],cfg.do_sigmoid_after_model_prediction)
        pod_3_hour, far_3_hour, ts_3_hour, ets_3_hour, micro_f1_3_hour, macro_f1_3_hour = val_calparams_epoch_3_hour.cal_batch(y[:,0:3,:,:,:], predicted_frames[:,0:3,:,:,:],cfg.do_sigmoid_after_model_prediction)
        #pod_neigh, far_neigh, ts_neigh, ets_neigh, micro_f1_neigh, macro_f1_neigh = val_calparams_epoch.cal_batch_neigh(y, predicted_frames,4,4,cfg.do_sigmoid_after_model_prediction)
        sumpod, sumfar, sumts, sumets, sum_micro_f1, sum_macro_f1 = val_calparams_epoch.cal_batch_sum(y, predicted_frames,cfg.do_sigmoid_after_model_prediction)
        sumpod_neigh, sumfar_neigh, sumts_neigh, sumets_neigh, sum_micro_f1_neigh, sum_macro_f1_neigh = val_calparams_epoch.cal_batch_sum_neigh(y, predicted_frames,4,4,cfg.do_sigmoid_after_model_prediction)

        sumpod_1_hour, sumfar_1_hour, sumts_1_hour, sumets_1_hour, sum_micro_f1_1_hour, sum_macro_f1_1_hour = val_calparams_epoch_1_hour.cal_batch_sum(y[:,0:1,:,:,:], predicted_frames[:,0:1,:,:,:],cfg.do_sigmoid_after_model_prediction)
        sumpod_1_hour_neigh, sumfar_1_hour_neigh, sumts_1_hour_neigh, sumets_1_hour_neigh, sum_micro_f1_1_hour_neigh, sum_macro_f1_1_hour_neigh = val_calparams_epoch_1_hour.cal_batch_sum_neigh(y[:,0:1,:,:,:], predicted_frames[:,0:1,:,:,:],4,4,cfg.do_sigmoid_after_model_prediction,1)

        sumpod_3_hour, sumfar_3_hour, sumts_3_hour, sumets_3_hour, sum_micro_f1_3_hour, sum_macro_f1_3_hour = val_calparams_epoch_3_hour.cal_batch_sum(y[:,0:3,:,:,:], predicted_frames[:,0:3,:,:,:],cfg.do_sigmoid_after_model_prediction)
        sumpod_3_hour_neigh, sumfar_3_hour_neigh, sumts_3_hour_neigh, sumets_3_hour_neigh, sum_micro_f1_3_hour_neigh, sum_macro_f1_3_hour_neigh = val_calparams_epoch_3_hour.cal_batch_sum_neigh(y[:,0:3,:,:,:], predicted_frames[:,0:3,:,:,:],4,4,cfg.do_sigmoid_after_model_prediction,3)
        del X
        del y
        # del _
        del predicted_frames
    pod, far, ts, ets, micro_f1, macro_f1 = val_calparams_epoch.cal_epoch()
    #pod_neigh, far_neigh, ts_neigh, ets_neigh, micro_f1_neigh, macro_f1_neigh = val_calparams_epoch.cal_epoch_neigh()
    sumpod, sumfar, sumts, sumets, sum_micro_f1, sum_macro_f1 = val_calparams_epoch.cal_epoch_sum()
    sumpod_neigh, sumfar_neigh, sumts_neigh, sumets_neigh, sum_micro_f1_neigh, sum_macro_f1_neigh = val_calparams_epoch.cal_epoch_sum_neigh()

    pod_1_hour, far_1_hour, ts_1_hour, ets_1_hour, micro_f1_1_hour, macro_f1_1_hour = val_calparams_epoch_1_hour.cal_epoch()
    #pod_neigh_1_hour, far_neigh_1_hour, ts_neigh_1_hour, ets_neigh_1_hour, micro_f1_neigh_1_hour, macro_f1_neigh_1_hour = val_calparams_epoch_1_hour.cal_epoch_neigh()
    sumpod_1_hour, sumfar_1_hour, sumts_1_hour, sumets_1_hour, sum_micro_f1_1_hour, sum_macro_f1_1_hour = val_calparams_epoch_1_hour.cal_epoch_sum()
    sumpod_1_hour_neigh, sumfar_1_hour_neigh, sumts_1_hour_neigh, sumets_1_hour_neigh, sum_micro_f1_1_hour_neigh, sum_macro_f1_1_hour_neigh = val_calparams_epoch_1_hour.cal_epoch_sum_neigh()

    pod_3_hour, far_3_hour, ts_3_hour, ets_3_hour, micro_f1_3_hour, macro_f1_3_hour = val_calparams_epoch_3_hour.cal_epoch()
    #pod_neigh_3_hour, far_neigh_3_hour, ts_neigh_3_hour, ets_neigh_3_hour, micro_f1_neigh_3_hour, macro_f1_neigh_3_hour = val_calparams_epoch_3_hour.cal_epoch_neigh()
    sumpod_3_hour, sumfar_3_hour, sumts_3_hour, sumets_3_hour, sum_micro_f1_3_hour, sum_macro_f1_3_hour = val_calparams_epoch_3_hour.cal_epoch_sum()
    sumpod_3_hour_neigh, sumfar_3_hour_neigh, sumts_3_hour_neigh, sumets_3_hour_neigh, sum_micro_f1_3_hour_neigh, sum_macro_f1_3_hour_neigh = val_calparams_epoch_3_hour.cal_epoch_sum_neigh()


    


    # # spatio-temporal metrics
    # print("Spatio-temporal metrics")
    # print('TP:{:09d}  FP:{:09d} \nFN:{:09d}  TN:{:09d}'.format(val_calparams_epoch.n1, val_calparams_epoch.n2, val_calparams_epoch.n3, val_calparams_epoch.n4))
    # print('POD:{:.5f}  FAR:{:.5f}  TS:{:.5f}  ETS:{:.5f}'.format(pod, far, ts, ets))
    # print('microF1:{:.5f}  macroF1:{:.5f}'.format(micro_f1, macro_f1))

    

    # spatio-temporal metrics for neighborhood
    #print("Spatio-temporal metrics for neighborhood")
    #print('TP:{:09d}  FP:{:09d} \nFN:{:09d}  TN:{:09d}'.format(val_calparams_epoch.n1_neigh, val_calparams_epoch.n2_neigh, val_calparams_epoch.n3_neigh, val_calparams_epoch.n4_neigh))
    #print('POD:{:.5f}  FAR:{:.5f}  TS:{:.5f}  ETS:{:.5f}'.format(pod_neigh, far_neigh, ts_neigh, ets_neigh))
    #print('microF1:{:.5f}  macroF1:{:.5f}'.format(micro_f1_neigh, macro_f1_neigh))

    

    # spatio-temporal metrics for 1 hour
    # print("Spatio-temporal metrics for 1 hour")
    # print('TP:{:09d}  FP:{:09d} \nFN:{:09d}  TN:{:09d}'.format(val_calparams_epoch_1_hour.n1, val_calparams_epoch_1_hour.n2, val_calparams_epoch_1_hour.n3, val_calparams_epoch_1_hour.n4))
    # print('POD:{:.5f}  FAR:{:.5f}  TS:{:.5f}  ETS:{:.5f}'.format(pod_1_hour, far_1_hour, ts_1_hour, ets_1_hour))
    # print('microF1:{:.5f}  macroF1:{:.5f}'.format(micro_f1_1_hour, macro_f1_1_hour))

    # spatio-only metrics for 1 hour
    print("1 hour metrics")
    print('TP:{:09d}  FP:{:09d}\nFN:{:09d}  TN:{:09d}'.format(val_calparams_epoch_1_hour.n1sum, val_calparams_epoch_1_hour.n2sum, val_calparams_epoch_1_hour.n3sum, val_calparams_epoch_1_hour.n4sum))
    print('sumPOD:{:.5f}  sumFAR:{:.5f}  sumTS:{:.5f}  sumETS:{:.5f}'.format(sumpod_1_hour, sumfar_1_hour, sumts_1_hour, sumets_1_hour))
    print('summicroF1:{:.5f}  summacroF1:{:.5f}'.format(sum_micro_f1_1_hour, sum_macro_f1_1_hour))

    # spatio-temporal metrics for neighborhood 1 hour
    #print("Spatio-temporal metrics for neighborhood 1 hour")
    #print('TP:{:09d}  FP:{:09d} \nFN:{:09d}  TN:{:09d}'.format(val_calparams_epoch_1_hour.n1_neigh, val_calparams_epoch_1_hour.n2_neigh, val_calparams_epoch_1_hour.n3_neigh, val_calparams_epoch_1_hour.n4_neigh))
    #print('POD:{:.5f}  FAR:{:.5f}  TS:{:.5f}  ETS:{:.5f}'.format(pod_neigh_1_hour, far_neigh_1_hour, ts_neigh_1_hour, ets_neigh_1_hour))
    #print('microF1:{:.5f}  macroF1:{:.5f}'.format(micro_f1_neigh_1_hour, macro_f1_neigh_1_hour))


    # spatio-only metrics for neighborhood 1 hour
    print("1 hour metrics for neighborhood")
    print('TP:{:09d}  FP:{:09d}\nFN:{:09d}  TN:{:09d}'.format(val_calparams_epoch_1_hour.n1sum_neigh, val_calparams_epoch_1_hour.n2sum_neigh, val_calparams_epoch_1_hour.n3sum_neigh, val_calparams_epoch_1_hour.n4sum_neigh))
    print('sumPOD:{:.5f}  sumFAR:{:.5f}  sumTS:{:.5f}  sumETS:{:.5f}'.format(sumpod_1_hour_neigh, sumfar_1_hour_neigh, sumts_1_hour_neigh, sumets_1_hour_neigh))
    print('summicroF1:{:.5f}  summacroF1:{:.5f}'.format(sum_micro_f1_1_hour_neigh, sum_macro_f1_1_hour_neigh))
    print('')

    # # spatio-temporal metrics for 3 hour
    # print("Spatio-temporal metrics for 3 hour")
    # print('TP:{:09d}  FP:{:09d} \nFN:{:09d}  TN:{:09d}'.format(val_calparams_epoch_3_hour.n1, val_calparams_epoch_3_hour.n2, val_calparams_epoch_3_hour.n3, val_calparams_epoch_3_hour.n4))
    # print('POD:{:.5f}  FAR:{:.5f}  TS:{:.5f}  ETS:{:.5f}'.format(pod_3_hour, far_3_hour, ts_3_hour, ets_3_hour))
    # print('microF1:{:.5f}  macroF1:{:.5f}'.format(micro_f1_3_hour, macro_f1_3_hour))

    # spatio-only metrics for 3 hour
    print("3 hour metrics")
    print('TP:{:09d}  FP:{:09d}\nFN:{:09d}  TN:{:09d}'.format(val_calparams_epoch_3_hour.n1sum, val_calparams_epoch_3_hour.n2sum, val_calparams_epoch_3_hour.n3sum, val_calparams_epoch_3_hour.n4sum))
    print('sumPOD:{:.5f}  sumFAR:{:.5f}  sumTS:{:.5f}  sumETS:{:.5f}'.format(sumpod_3_hour, sumfar_3_hour, sumts_3_hour, sumets_3_hour))
    print('summicroF1:{:.5f}  summacroF1:{:.5f}'.format(sum_micro_f1_3_hour, sum_macro_f1_3_hour))

    # spatio-temporal metrics for neighborhood 3 hour
    #print("Spatio-temporal metrics for neighborhood 3 hour")
    #print('TP:{:09d}  FP:{:09d} \nFN:{:09d}  TN:{:09d}'.format(val_calparams_epoch_3_hour.n1_neigh, val_calparams_epoch_3_hour.n2_neigh, val_calparams_epoch_3_hour.n3_neigh, val_calparams_epoch_3_hour.n4_neigh))
    #print('POD:{:.5f}  FAR:{:.5f}  TS:{:.5f}  ETS:{:.5f}'.format(pod_neigh_3_hour, far_neigh_3_hour, ts_neigh_3_hour, ets_neigh_3_hour))
    #print('microF1:{:.5f}  macroF1:{:.5f}'.format(micro_f1_neigh_3_hour, macro_f1_neigh_3_hour))

    # spatio-only metrics for neighborhood 3 hour
    print("3 hour metrics for neighborhood")
    print('TP:{:09d}  FP:{:09d}\nFN:{:09d}  TN:{:09d}'.format(val_calparams_epoch_3_hour.n1sum_neigh, val_calparams_epoch_3_hour.n2sum_neigh, val_calparams_epoch_3_hour.n3sum_neigh, val_calparams_epoch_3_hour.n4sum_neigh))
    print('sumPOD:{:.5f}  sumFAR:{:.5f}  sumTS:{:.5f}  sumETS:{:.5f}'.format(sumpod_3_hour_neigh, sumfar_3_hour_neigh, sumts_3_hour_neigh, sumets_3_hour_neigh))
    print('summicroF1:{:.5f}  summacroF1:{:.5f}'.format(sum_micro_f1_3_hour_neigh, sum_macro_f1_3_hour_neigh))
    print('')


    # spatio-only metrics
    print("6 hour metrics")
    print('TP:{:09d}  FP:{:09d}\nFN:{:09d}  TN:{:09d}'.format(val_calparams_epoch.n1sum, val_calparams_epoch.n2sum, val_calparams_epoch.n3sum, val_calparams_epoch.n4sum))
    print('sumPOD:{:.5f}  sumFAR:{:.5f}  sumTS:{:.5f}  sumETS:{:.5f}'.format(sumpod, sumfar, sumts, sumets))
    print('summicroF1:{:.5f}  summacroF1:{:.5f}'.format(sum_micro_f1, sum_macro_f1))



    # spatio-only metrics for neighborhood
    print("6 hour metrics for neighborhood")
    print('TP:{:09d}  FP:{:09d}\nFN:{:09d}  TN:{:09d}'.format(val_calparams_epoch.n1sum_neigh, val_calparams_epoch.n2sum_neigh, val_calparams_epoch.n3sum_neigh, val_calparams_epoch.n4sum_neigh))
    print('sumPOD:{:.5f}  sumFAR:{:.5f}  sumTS:{:.5f}  sumETS:{:.5f}'.format(sumpod_neigh, sumfar_neigh, sumts_neigh, sumets_neigh))
    print('summicroF1:{:.5f}  summacroF1:{:.5f}'.format(sum_micro_f1_neigh, sum_macro_f1_neigh))
    print('')