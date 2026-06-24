import os


class Config:
    def __init__(self, **kwargs):
        # ==========================================
        # 1. Flags
        # ==========================================
        # Models: MCAD_Model (multi scale, cross modal), MCAD_Model_cs(cross scale, cross modal)
        self.model_type = 'MCAD_Model_SynScale2'
        # Additional Alternative Modules: 'CBAM', 'ECA', 'GAM', 'TripletAM', 'EMA', 'LSKNet', 'None'
        self.attention_type = 'None'
        self.fused_block = 'None'
        self.custom_ext = ''
        self.save_model = False

        # Run mode: 'single' (split 8:2) or 'kfold' (Cross Validation)
        self.run_mode = 'single'
        self.k_folds = 5
        self.test_size = 0.2
        self.random_seed = 999999
        self.use_cfp_only = False
        self.special_test = '' #Test Distillation model

        self.pretrained_model_oct = False
        #self.pretrained_model_oct_pth = r'utils/distilled_resnet18_bscan_bs64_lr1e4_10kAMD.pth'
        self.pretrained_model_oct_pth = r'utils/distilled_resnet18_bscan_bs64_lr1e4_10kAMD_448.pth'
        #self.pretrained_model_oct_pth = r'utils/distilled_resnet18_bscan_bs64_lr1e4_10kAMD_224.pth'

        self.pretrained_model_cfp = False
        self.pretrained_model_cfp_pth = r'utils/distilled_efficientB0_cfp_bs64_lr0.0001_wd0.0001_MultiEye_448.pth'
        #self.pretrained_model_cfp_pth = r'utils/distilled_efficientB0_cfp_bs64_lr0.0001_wd0.0001_MultiEye_224.pth'

        # 2. Loss, Bayesian Configs
        # ==========================================
        # Train Loss: 'CE' or 'UAL' (Uncertainty-Aware Loss)
        self.loss_type = 'CE'
        self.mc_passes_train = 50  # MC-Dropout on training phrase for UAL

        # Chế độ Inference lúc Test:
        self.bayesian_test = True  # True: Enable Bayesian (MC-Dropout) on test. False: Test 1 time only.
        self.mc_passes_test = 1  # MC-Dropout on testing phrase for UAL
        self.entropy_threshold = 0.5  # Threshold for High Uncertainty Samples (HUS)


        # 3. System Configs
        self.gpus = '0'
        self.embed_dim = 256
        self.num_workers = 0
        self.pin_memory = True

        # 4. Data Paths
        self.dataset_name = 'mmc-amd'
        self.csv_path_training = r'data/mmc_amd_all_unique_pairs_training.csv'
        self.csv_path_testing = r'data/mmc_amd_all_unique_pairs_testing.csv'
        self.cfp_data_dir = r'D:\Projects\Personal\Master\OCT\Dataset\mmc-amd\ImageData\cfp-clahe-448x448'
        self.oct_data_dir = r'D:\Projects\Personal\Master\OCT\Dataset\mmc-amd\ImageData\oct-median3x3-448x448'


        # 5. Model Params
        self.input_size = 448
        self.num_classes = 4
        self.embed_dim = 256
        self.dropout = 0.3
        self.dropout_test = 0.5

        self.alpha = 0.05

        # 6. Training hyperparams
        self.epochs = 60
        self.batch_size = 16
        self.learning_rate = 1e-3
        self.weight_decay = 1e-2


        # 7. Augmentation
        self.fill = 0
        self.affine = True


        self.use_supcon = False
        self.beta_supcon = 0.5
        self.supcon_temperature = 0.07

        self.optimize_metric = 'F1-score'
        self.early_stopping_patience = 10
        self.unfreeze_all = False
        self.freeze_scale1 = False

        self.lr_backbone = 1e-3
        self.lr_head = 1e-3
        self.dropout_head = 0.2
        self.dropout_mc = 0.5

        self.cfp_attention_type = None
        self.oct_attention_type = None

        # 8. Update params
        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.input_size == 224:
            self.pretrained_model_cfp_pth = r'utils/distilled_efficientB0_cfp_bs64_lr0.0001_wd0.0001_MultiEye_224.pth'
            self.pretrained_model_oct_pth = r'utils/distilled_resnet18_bscan_bs64_lr1e4_10kAMD_224.pth'
        else:
            self.pretrained_model_cfp_pth = r'utils/distilled_efficientB0_cfp_bs64_lr0.0001_wd0.0001_MultiEye_448.pth'
            self.pretrained_model_oct_pth = r'utils/distilled_resnet18_bscan_bs64_lr1e4_10kAMD_448.pth'

            if self.model_type == 'MCAD_Model_GhostNet':
                self.pretrained_model_cfp_pth = r'utils/distilled_ghostnet_100_cfp_bs64_lr0.0001_wd0.0001_MultiEye_448.pth'
                self.pretrained_model_oct_pth = r'utils/distilled_ghostnet_100_bscan_bs64_lr0.0001_wd0.0001_448.pth'

            if self.model_type == 'MCAD_Model_SynScale2_b1':
                self.pretrained_model_cfp_pth = r'utils/distilled_efficientnet_b1_cfp_448.pth'

        if self.dataset_name == 'mmc-amd':
            self.csv_path = r'data/mmc_amd_all_unique_pairs.csv'
            self.cfp_data_dir = r'D:\Projects\Personal\Master\OCT\Dataset\mmc-amd\ImageData\cfp-clahe-448x448'
            self.oct_data_dir = r'D:\Projects\Personal\Master\OCT\Dataset\mmc-amd\ImageData\oct-median3x3-448x448'
            self.num_classes = 4
        elif self.dataset_name == 'gamma':
            self.csv_path = r'E:\Project\OCT\SourceCode\DCAT\data\GAMMA\glaucoma_grading_training_GT.csv'
            self.cfp_data_dir = r'E:\Project\OCT\SourceCode\DCAT\data\GAMMA\prepro\CFP-448x448'
            self.oct_data_dir = r'E:\Project\OCT\SourceCode\DCAT\data\GAMMA\prepro\OCT-448x448'
            self.num_classes = 3
            self.pretrained_model_oct_pth = r'utils/distilled_resnet18_bscan_bs64_bs0.0001_bs0.0001_10k_glaucoma_448.pth'
        elif self.dataset_name == 'topcon-mm':
            self.csv_path = r"D:\Projects\Personal\Master\OCT\Dataset\topconmm_data\topcon_mm_valid_pairs.csv"
            self.cfp_data_dir = r"D:\Projects\Personal\Master\OCT\Dataset\topconmm_data\fundus_images"
            self.oct_data_dir = r"D:\Projects\Personal\Master\OCT\Dataset\topconmm_data\oct_images"
            self.num_classes = 11
            self.epochs = 100
            self.batch_size = 64
            self.optimize_metric = 'AP'
        elif self.dataset_name == 'HarvardFairVision-AMD':
            #self.csv_path = r'D:\Projects\Personal\Master\OCT\SourceCode\FairVision\dataset\HarvardFairVision\AMD\your_file.csv'
            self.cfp_data_dir = r'D:\Projects\Personal\Master\OCT\SourceCode\FairVision\dataset\HarvardFairVision\prepro\FairVision_AMD_slo_clahe'
            self.oct_data_dir = r'D:\Projects\Personal\Master\OCT\SourceCode\FairVision\dataset\HarvardFairVision\prepro\FairVision_AMD_oct_median3x3'
            self.num_classes = 4
        elif self.dataset_name == 'HarvardFairVision-DR':
            # self.csv_path = r'D:\Projects\Personal\Master\OCT\SourceCode\FairVision\dataset\HarvardFairVision\AMD\your_file.csv'
            #self.cfp_data_dir = r'D:\Projects\Personal\Master\OCT\SourceCode\FairVision\dataset\HarvardFairVision\prepro\FairVision_DR_slo_clahe'
            #self.oct_data_dir = r'D:\Projects\Personal\Master\OCT\SourceCode\FairVision\dataset\HarvardFairVision\prepro\FairVision_DR_oct_median3x3'
            self.cfp_data_dir = r'D:\Projects\Personal\Master\OCT\SourceCode\FairVision\dataset\HarvardFairVision\prepro\FairVision_DR_slo_clahe_b'
            self.oct_data_dir = r'D:\Projects\Personal\Master\OCT\SourceCode\FairVision\dataset\HarvardFairVision\prepro\FairVision_DR_oct_median3x3_b'
            self.num_classes = 2

        #self.model_name = f'{self.model_type}_{self.attention_type}_fb{self.fused_block}_{self.loss_type}sUnc_bs{self.batch_size}_lr{self.learning_rate}_rs{self.random_seed}_dr{self.dropout}_pt{self.mc_passes_train}_disOCT{self.pretrained_model_oct}_disCFP{self.pretrained_model_cfp}'
        self.model_name = f'{self.model_type}_cfp{self.cfp_attention_type}_oct{self.oct_attention_type}_fb{self.fused_block}_{self.loss_type}_bs{self.batch_size}_lrbb{self.lr_backbone}_rs{self.random_seed}_ptr{self.mc_passes_train}_ptr{self.mc_passes_test}_alpha{self.alpha}_dr{self.dropout}_{self.custom_ext}'

        if self.special_test != '':
            self.model_name = f'{self.special_test}_bs{self.batch_size}_lr{self.learning_rate}_rs{self.random_seed}_dr{self.dropout}'
        print(f'Config initialized. Mode: {self.run_mode} | Attention: {self.attention_type} | Loss: {self.loss_type}')

    def get_output_dir(self, fold=None):
        base_dir = './output_updateJune2026'

        model_name = f"MCAD_Res18_EffB0_{self.attention_type}_{self.loss_type}"

        param_str = f"bs{self.batch_size}_lr{self.learning_rate}_wd{self.weight_decay}_rs{self.random_seed}"

        if self.run_mode == 'kfold':
            mode_str = f"KFold{self.k_folds}"
            if fold is not None:
                mode_str += f"_F{fold}"  # exp: KFold5_F1
        else:
            mode_str = "SingleRun"

        folder_name = f"{model_name}_{mode_str}_{param_str}"
        full_path = os.path.join(base_dir, self.dataset_name, self.model_name, folder_name)

        return full_path
