import os
import numpy as np


class Config(object):
    def __init__(self):
        """Configuration class that contain attributes to set paths and network options.

        Dependencies:
            A txt file named 'profile.txt' in the same directory that matches a set of paths defined below.
        """

        # Get profile
        __location__ = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__))
        )
        with open(os.path.join(__location__, "profile.txt"), "r") as f:
            profile = f.readline()

        # Set local data directory
        if profile == "predator":
            self.data_dir = "H:\\nAge\\"
            self.tmp_dir = "H:\\nAge\\tmp\\"
            self.cfs_ds_path = "G:\\cfs\\datasets\\cfs-visit5-dataset-0.4.0.csv"
            self.cfs_ssc_path = "G:\\cfs\\polysomnography\\annotations-events-profusion"
            self.mros_ds_path = "G:\\mros\\datasets\\mros-visit1-dataset-0.3.0.csv"
            self.mros_ssc_path = (
                "G:\\mros\\polysomnography\\annotations-events-profusion\\visit1"
            )
            self.shhs_ds_path = "H:\\shhs\\datasets\\shhs1-dataset-0.15.0.csv"
            self.shhs_ssc_path = (
                "H:\\shhs\\polysomnography\\annotations-events-profusion\\shhs1"
            )
            self.wsc_ds_path = "G:\\WSC_PLM_ data_all.xlsx"
            self.wsc_ssc_path = "G:\\wsc\\polysomnography\\labels"
            self.stages_ds_path = "H:\\STAGES\\PatientDemographics.xlsx"
            self.stages_ssc_path = "H:\\STAGES\\polysomnograms"
            self.ssc_ds_path = "H:\\SSC\\ssc.xlsx"
            self.ssc_ssc_path = "H:\\SSC\\polysomnography\\labels"
            self.sof_ds_path = "H:\\sof\\datasets\\sof-visit-8-dataset-0.6.0.csv"
            self.sof_ssc_path = "H:\\sof\\polysomnography\\annotations-events-profusion"
            self.hpap_ds_path = (
                "H:\\homepap\\datasets\\homepap-baseline-dataset-0.1.0.csv"
            )
            self.hpap_ssc_path = (
                "H:\\homepap\\polysomnography\\annotations-events-profusion\\lab\\full"
            )
            self.list_split_train = "H:\\nAge\\X_train.csv"
            self.list_split_val = "H:\\nAge\\X_val.csv"
            self.list_split_test = "H:\\nAge\\X_test.csv"
        elif profile == "sherlock":
            self.data_dir = "/scratch/users/abk26/nAge/"
            self.tmp_dir = "/scratch/users/abk26/nAge/tmp/"
            self.cfs_ds_path = (
                "/oak/stanford/groups/mignot/cfs/datasets/cfs-visit5-dataset-0.4.0.csv"
            )
            self.cfs_ssc_path = "/oak/stanford/groups/mignot/cfs/polysomnography/annotations-events-profusion/"
            self.mros_ds_path = "/oak/stanford/groups/mignot/mros/datasets/mros-visit1-dataset-0.3.0.csv"
            self.mros_ssc_path = "/oak/stanford/groups/mignot/mros/polysomnography/annotations-events-profusion/visit1/"
            self.shhs_ds_path = (
                "/oak/stanford/groups/mignot/shhs/datasets/shhs1-dataset-0.14.0.csv"
            )
            self.shhs_ssc_path = (
                "/home/users/abk26/SleepAge/Scripts/data/shhs/polysomnography/shhs1/"
            )
            self.wsc_ds_path = (
                "/home/users/abk26/SleepAge/Scripts/data/WSC_PLM_ data_all.xlsx"
            )
            self.wsc_ssc_path = "/oak/stanford/groups/mignot/psg/WSC_EDF/"
            self.stages_ds_path = (
                "/home/users/abk26/SleepAge/Scripts/data/PatientDemographics.xlsx"
            )
            self.stages_ssc_path = "/oak/stanford/groups/mignot/psg/STAGES/deid/"
            self.ssc_ds_path = "/home/users/abk26/SleepAge/Scripts/data/ssc.xlsx"
            self.ssc_ssc_path = "/oak/stanford/groups/mignot/psg/SSC/APOE_deid/"
            self.list_split_train = os.path.join(self.data_dir, "X_train.csv")
            self.list_split_val = os.path.join(self.data_dir, "X_val.csv")
            self.list_split_test = os.path.join(self.data_dir, "X_test.csv")
        elif profile == "new":
            self.data_dir = os.path.join(__location__, "data")
        else:
            self.data_dir = ""
            self.tmp_dir = ""
            self.cfs_ds_path = ""
            self.mros_ds_path = ""
            self.shhs_ds_path = ""
            self.wsc_ds_path = ""
            self.stages_ds_path = ""

        # Datapaths
        self.model_dir = os.path.join(self.data_dir, "model")
        self.train_dir = os.path.join(self.data_dir, "train")
        self.val_dir = os.path.join(self.data_dir, "val")
        self.interp_dir = os.path.join(self.data_dir, "interpretation")
        self.am_dir = os.path.join(self.data_dir, "am")
        self.test_dir = os.path.join(self.data_dir, "test")
        self.train_cache_dir = os.path.join(self.data_dir, "train_cache")
        self.val_cache_dir = os.path.join(self.data_dir, "val_cache")
        self.test_cache_dir = os.path.join(self.data_dir, "test_cache")
        self.train_F_dir = os.path.join(self.data_dir, "train_F")
        self.val_F_dir = os.path.join(self.data_dir, "val_F")
        self.test_F_dir = os.path.join(self.data_dir, "test_F")
        self.pretrain_dir = os.path.join(self.data_dir, "all")
        self.F_train_dir = os.path.join(self.data_dir, "all_F")

        # Checkpoint
        self.save_dir = self.model_dir
        self.model_F_path = os.path.join(self.model_dir, "modelF")
        self.model_L_path = os.path.join(self.model_dir, "modelL")
        self.model_L_BO_path = self.model_L_path
        self.BO_expe_path = os.path.join(self.model_dir, "exp")

        self.return_only_pred = False
        self.return_pdf_shape = True

        # Pretraining
        # label-config
        self.pre_label = ["age"]  # ['age', 'bmi', 'sex']
        self.pre_label_size = [1]  # [1, 1, 2]
        self.pre_n_class = sum(self.pre_label_size)
        self.pre_only_sleep = 0
        # network-config
        self.n_channels = 12
        self.pre_model_num = 1
        # train-config
        self.pre_max_epochs = 20
        self.pre_patience = 3
        self.pre_batch_size = 32
        self.pre_lr = 1e-3
        self.pre_n_workers = 0
        self.do_f = 0.75
        self.pre_channel_drop = True
        self.pre_channel_drop_prob = 0.1
        self.loss_func = "huber"
        self.only_eeg = 1

        # Training
        # label-config
        self.label = ["age"]
        self.label_cond = []  # ['q_low', 'q_high'] #['q_low', 'q_high', 'bmi', 'sex']
        self.label_cond_size = []  # [7, 7] #[7, 7, 1, 1]
        self.n_class = 1
        # train-config
        self.do_l = 0.5
        self.max_epochs = 200
        self.patience = 20
        self.batch_size = 64
        self.lr = 5e-4
        self.l2 = 1e-5
        self.n_workers = 0
        self.pad_length = 120
        self.cond_drop = False
        self.cond_drop_prob = 0.5
        # network-config
        self.net_size_scale = 4
        self.lstm_n = 1
        self.epoch_size = 5 * 60 * 128
        self.return_att_weights = False
