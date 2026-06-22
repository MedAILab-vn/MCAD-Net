import os
import json
import copy
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split, StratifiedKFold

from config import Config
from data.dataset import MMCAmdCSVDataset, build_cfp_transform, build_oct_transform, GAMMADataset, \
    HarvardFairVisionDataset, TOPCONDataset
from models.MCAD_net import MCAD_Model, MCAD_Model_cs, MCAD_Model_MultiScale, MCAD_Model_MultiScale_CFPOnly, MCAD_Model_SupCon, \
    MCAD_Model_Asyn, MCAD_Model_SynScale2, MCAD_Model_LateFusion, MCAD_Model_DenseNet, MCAD_Model_GhostNet, MCAD_Model_SynScale2_b1, MCAD_Model_LateFusionOnly, \
    MCAD_Model_LateFusionOnlyFlat, MCAD_Model_Scale2CrossAttOnly
from utils.engine import evaluate_with_uncertainty, set_seed, enable_dropout
from models.losses import build_loss_function, MultimodalSupConLoss
import gc
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from thop import profile

def train_and_evaluate(train_df, val_df, config, device, fold=None, dataset_name='mmc-amd'):
    set_seed(config.random_seed)

    output_dir = config.get_output_dir(fold=fold)
    os.makedirs(output_dir, exist_ok=True)

    config_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}
    with open(os.path.join(output_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)

    if dataset_name == 'mmc-amd':
        train_ds = MMCAmdCSVDataset(train_df, config.cfp_data_dir, config.oct_data_dir,
                                    build_cfp_transform(config, True), build_oct_transform(config, True), "Train_Set")
        val_ds = MMCAmdCSVDataset(val_df, config.cfp_data_dir, config.oct_data_dir,
                                  build_cfp_transform(config, False), build_oct_transform(config, False), "Val_Set")
        test_loader = None

    elif dataset_name == 'gamma':
        train_ds = GAMMADataset(train_df, config.cfp_data_dir, config.oct_data_dir,
                                build_cfp_transform(config, True), build_oct_transform(config, True), "Train_Set")
        val_ds = GAMMADataset(val_df, config.cfp_data_dir, config.oct_data_dir,
                              build_cfp_transform(config, False), build_oct_transform(config, False), "Val_Set")
        test_loader = None
    elif dataset_name == 'topcon-mm':
        train_ds = TOPCONDataset(train_df, config.cfp_data_dir, config.oct_data_dir,
                                 build_cfp_transform(config, True), build_oct_transform(config, True), "Train_Set")
        val_ds = TOPCONDataset(val_df, config.cfp_data_dir, config.oct_data_dir,
                               build_cfp_transform(config, False), build_oct_transform(config, False), "Val_Set")
        test_loader = None

    elif dataset_name == 'HarvardFairVision-AMD' or dataset_name == 'HarvardFairVision-DR':
        train_ds = HarvardFairVisionDataset(config.cfp_data_dir, config.oct_data_dir, split="train",
                                            transform_cfp=build_cfp_transform(config, True),
                                            transform_oct=build_oct_transform(config, True))
        val_ds = HarvardFairVisionDataset(config.cfp_data_dir, config.oct_data_dir, split="val",
                                          transform_cfp=build_cfp_transform(config, False),
                                          transform_oct=build_oct_transform(config, False))
        test_ds = HarvardFairVisionDataset(config.cfp_data_dir, config.oct_data_dir, split="test",
                                           transform_cfp=build_cfp_transform(config, False),
                                           transform_oct=build_oct_transform(config, False))

        test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, pin_memory=config.pin_memory,
                                 num_workers=config.num_workers)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, pin_memory=config.pin_memory,
                              num_workers=config.num_workers)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, pin_memory=config.pin_memory,
                            num_workers=config.num_workers)

    criterion_supcon = None

    if config.model_type == 'MCAD_Model':
        model = MCAD_Model(num_classes=config.num_classes, embed_dim=config.embed_dim, dropout_rate=config.dropout,
                           attention_type=config.attention_type).to(device)
    elif config.model_type == 'MCAD_Model_cs':
        model = MCAD_Model_cs(num_classes=config.num_classes, embed_dim=config.embed_dim, dropout_rate=config.dropout,
                              attention_type=config.attention_type).to(device)
    elif config.model_type == 'MCAD_Model_MultiScale':
        model = MCAD_Model_MultiScale(config).to(device)
    elif config.model_type == 'MCAD_Model_Asyn':
        model = MCAD_Model_Asyn(config).to(device)
    elif config.model_type == 'MCAD_Model_SynScale2':
        model = MCAD_Model_SynScale2(config).to(device)
    elif config.model_type == 'MCAD_Model_LateFusionOnly':
        model = MCAD_Model_LateFusionOnly(config).to(device)
    elif config.model_type == 'MCAD_Model_LateFusionOnlyFlat':
        model = MCAD_Model_LateFusionOnlyFlat(config).to(device)
    elif config.model_type == 'MCAD_Model_Scale2CrossAttOnly':
        model = MCAD_Model_Scale2CrossAttOnly(config).to(device)
    elif config.model_type == 'MCAD_Model_SynScale2_b1':
        model = MCAD_Model_SynScale2_b1(config).to(device)
    elif config.model_type == 'MCAD_Model_LateFusion':
        model = MCAD_Model_LateFusion(config).to(device)
    elif config.model_type == 'MCAD_Model_DenseNet':
        model = MCAD_Model_DenseNet(config).to(device)
    elif config.model_type == 'MCAD_Model_GhostNet':
        model = MCAD_Model_GhostNet(config).to(device)
    elif config.model_type == 'MCAD_Model_MultiScale_CFPOnly':
        model = MCAD_Model_MultiScale_CFPOnly(num_classes=config.num_classes, embed_dim=config.embed_dim,
                                          dropout_rate=config.dropout,
                                          attention_type=config.attention_type).to(device)
    elif config.model_type == 'MCAD_Model_SupCon':
        model = MCAD_Model_SupCon(config).to(device)
        criterion_supcon = MultimodalSupConLoss(temperature=config.supcon_temperature).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if config.dataset_name == 'topcon-mm':
        criterion = nn.BCEWithLogitsLoss()
    else:
        alpha_val = getattr(config, 'alpha', 0.1)
        criterion = build_loss_function(config.loss_type, device, alpha=alpha_val)

    lr_backbone = getattr(config, 'lr_backbone', 1e-4)
    lr_head = getattr(config, 'lr_head', 1e-3)

    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': lr_backbone},
        {'params': head_params, 'lr': lr_head}
    ], weight_decay=config.weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)

    best_f1 = -1.0
    best_auc = -1.0
    best_bacc = -1.0
    best_ap = -1.0
    best_map = -1.0
    best_model_weights = None
    training_history = []
    patience = getattr(config, 'early_stopping_patience', 10)
    patience_counter = 0

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0

        for inputs_CFP, inputs_OCT, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs} [Train]",
                                                   leave=False):
            inputs_CFP, inputs_OCT, labels = inputs_CFP.to(device), inputs_OCT.to(device), labels.to(device)

            if getattr(config, 'use_cfp_only', False):
                inputs_OCT = inputs_CFP

            optimizer.zero_grad()

            if config.model_type == 'MCAD_Model_SupCon':
                enable_dropout(model)
                mc_passes = getattr(config, 'mc_passes_train', 50)
                out, vec_cfp, vec_oct = model(inputs_CFP, inputs_OCT, mc_passes=mc_passes)
                cls_loss = criterion(out, labels)
                supcon_loss = criterion_supcon(vec_cfp, vec_oct, labels)
                beta = getattr(config, 'beta_supcon', 0.5)
                loss = cls_loss + (beta * supcon_loss)
            elif getattr(config, 'loss_type', 'CE') == 'UAL':
                enable_dropout(model)
                mc_passes = getattr(config, 'mc_passes_train', 50)
                stacked_logits = model(inputs_CFP, inputs_OCT, mc_passes=mc_passes)
                loss = criterion(stacked_logits, labels)
            else:
                out = model(inputs_CFP, inputs_OCT)
                loss = criterion(out, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs_CFP.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        val_metrics = evaluate_with_uncertainty(model, val_loader, criterion, device, config.num_classes,
                                                f"Epoch_{epoch + 1}", mc_passes=1,
                                                use_cfp_only=getattr(config, 'use_cfp_only', False))
        val_metrics['Train_Loss'] = train_loss
        training_history.append(val_metrics)

        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch + 1:02d} | LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['Loss']:.4f} | Acc: {val_metrics['Acc']:.4f} | F1: {val_metrics['F1-score']:.4f} | AUC: {val_metrics['AUC']:.4f}")

        scheduler.step()

        metric_name = config.optimize_metric
        current_score = val_metrics[metric_name]

        if metric_name == 'F1-score':
            best_score = best_f1
        elif metric_name == 'AUC':
            best_score = best_auc
        elif metric_name == 'BAcc':
            best_score = best_bacc
        elif metric_name == 'AP':
            best_score = best_ap
        elif metric_name == 'mAP':
            best_score = best_map
        else:
            raise ValueError(f"Invalid optimization metric: {metric_name}")

        if current_score > best_score:
            if metric_name == 'F1-score':
                best_f1 = current_score
            elif metric_name == 'AUC':
                best_auc = current_score
            elif metric_name == 'BAcc':
                best_bacc = current_score
            elif metric_name == 'AP':
                best_ap = current_score
            elif metric_name == 'mAP':
                best_map = current_score

            best_score = current_score
            best_model_weights = copy.deepcopy(model.state_dict())

            best_val_metrics = copy.deepcopy(val_metrics)
            best_val_metrics['Split'] = f"Best_Val_Epoch_{epoch + 1}"

            patience_counter = 0
            print(f"Saved new weights with {metric_name}: {best_score:.4f}")
        else:
            patience_counter += 1
            print(f"{metric_name} did not improve. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"\nEarly Stop at {epoch + 1}!")
            break

    model.load_state_dict(best_model_weights)
    if config.save_model:
        torch.save(best_model_weights, os.path.join(output_dir, f"best_model_fold{fold if fold else '1'}.pth"))

    is_bayesian_test = getattr(config, 'bayesian_test', True)
    test_passes = getattr(config, 'mc_passes_test', 1) if is_bayesian_test else 1

    if test_passes != 1:
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.p = config.dropout_test

    final_eval_loader = test_loader if test_loader is not None else val_loader

    final_val_mc = evaluate_with_uncertainty(model, final_eval_loader, criterion, device, config.num_classes,
                                             "FINAL_VAL_MC_DROPOUT", mc_passes=test_passes,
                                             entropy_threshold=config.entropy_threshold,
                                             use_cfp_only=getattr(config, 'use_cfp_only', False))

    print(
        f"[{'Fold ' + str(fold) if fold else 'Single Run'} Results] F1: {final_val_mc['F1-score']:.4f} | Entropy: {final_val_mc.get('Mean_Entropy', 0.0):.4f} | HUS: {final_val_mc.get('HUS_Count', 0)}")

    pd.DataFrame(training_history).to_csv(os.path.join(output_dir, "training_history.csv"), index=False)

    important_cols = list(final_val_mc.keys())
    if 'Train_Loss' not in important_cols:
        important_cols.append('Train_Loss')
        final_val_mc['Train_Loss'] = 0.0

    for col in ['Mean_Entropy', 'HUS_Count']:
        if col not in final_val_mc:
            final_val_mc[col] = 0.0
            if col not in important_cols: important_cols.append(col)
        if col not in best_val_metrics:
            best_val_metrics[col] = 0.0

    df_final = pd.DataFrame([best_val_metrics, final_val_mc])[important_cols]
    df_final.to_csv(os.path.join(output_dir, "final_test_results.csv"), index=False)

    return final_val_mc, best_val_metrics

def main(current_seed, dname, alpha=0.01, branch_attention_type=None, dropout=0.5, loss_type="CE", batch_size=16, mc_passes_train=50):

    config = Config(
        dataset_name=dname,
        model_type='MCAD_Model_SynScale2',
        use_cfp_only=False,
        batch_size=batch_size,
        attention_type='None',
        fused_block='None',
        use_supcon=False,
        run_mode='kfold',
        loss_type=loss_type,
        bayesian_test=True,
        epochs=60,
        lr_head=1e-3,
        lr_backbone=1e-4,
        weight_decay=0.05,
        dropout=dropout,
        mc_passes_train=mc_passes_train,
        mc_passes_test=1,
        alpha=alpha,
        random_seed=current_seed,
        pretrained_model_oct=True,
        pretrained_model_cfp=True,
        optimize_metric='F1-score',
        input_size=448,
        unfreeze_all=False,
        early_stopping_patience=50,
        oct_attention_type=branch_attention_type,
        cfp_attention_type=branch_attention_type,
        freeze_scale1=False,
        custom_ext='_',
        save_model=False,
    )

    set_seed(config.random_seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.run_mode == 'train_val_test' or config.dataset_name.startswith('HarvardFairVision'):
        train_and_evaluate(None, None, config, device, fold=None, dataset_name=config.dataset_name)

    else:
        df_all = pd.read_csv(config.csv_path, sep=None, engine='python', encoding='utf-8-sig')

        if config.dataset_name == 'gamma':
            def get_gamma_label(row):
                if row.get('non', 0) == 1:
                    return 0
                elif row.get('early', 0) == 1:
                    return 1
                elif row.get('mid_advanced', 0) == 1:
                    return 2
                return 0
            df_all['label'] = df_all.apply(get_gamma_label, axis=1)

        if config.run_mode == 'single':
            train_df, val_df = train_test_split(df_all, test_size=config.test_size, stratify=df_all['label'], random_state=config.random_seed)
            print(f"Total images -> Train: {len(train_df)} | Val: {len(val_df)}")
            train_and_evaluate(train_df, val_df, config, device, fold=None, dataset_name=config.dataset_name)

        elif config.run_mode == 'kfold':

            if config.dataset_name == 'topcon-mm':
                from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

                label_cols = [c for c in df_all.columns if c not in ['core_id', 'cfp', 'oct']]
                y_multilabel = df_all[label_cols].values

                mskf = MultilabelStratifiedKFold(n_splits=config.k_folds, shuffle=True, random_state=config.random_seed)
                fold_splits = mskf.split(df_all, y_multilabel)

            else:
                from sklearn.model_selection import StratifiedKFold
                skf = StratifiedKFold(n_splits=config.k_folds, shuffle=True, random_state=config.random_seed)
                fold_splits = skf.split(df_all, df_all['label'])

            fold_results = []
            fold_best_results = []

            for fold, (train_idx, val_idx) in enumerate(fold_splits, 1):
                train_df = df_all.iloc[train_idx]
                val_df = df_all.iloc[val_idx]
                print(f"\n--- FOLD {fold}/{config.k_folds} ---")
                print(f"Train: {len(train_df)} | Val: {len(val_df)}")

                if config.dataset_name == 'topcon-mm':
                    val_disease_counts = val_df[label_cols].sum().to_dict()
                    print(f"Checking Val set label distribution: {val_disease_counts}")

                metrics, best_metrics = train_and_evaluate(train_df, val_df, config, device, fold=fold,
                                                           dataset_name=config.dataset_name)
                fold_results.append(metrics)
                fold_best_results.append(best_metrics)


            df_kfold = pd.DataFrame(fold_results)
            mean_metrics = df_kfold.mean(numeric_only=True)
            std_metrics = df_kfold.std(numeric_only=True)

            for col in ['F1-score', 'Acc', 'AUC', 'BAcc', 'Spe', 'Mean_Entropy']:
                print(f"{col}: {mean_metrics[col]:.4f} ± {std_metrics[col]:.4f}")

            mean_row = mean_metrics.to_dict()
            mean_row['Split'] = 'Average'
            std_row = std_metrics.to_dict()
            std_row['Split'] = 'Std_Dev'
            df_kfold = pd.concat([df_kfold, pd.DataFrame([mean_row, std_row])], ignore_index=True)
            base_kfold_dir = os.path.dirname(config.get_output_dir(fold=1))
            df_kfold.to_csv(os.path.join(base_kfold_dir, f"KFold_{config.k_folds}_Summary_rs{config.random_seed}.csv"),
                            index=False)

            df_kfold_best = pd.DataFrame(fold_best_results)
            mean_metrics_best = df_kfold_best.mean(numeric_only=True)
            std_metrics_best = df_kfold_best.std(numeric_only=True)
            mean_row_best = mean_metrics_best.to_dict()
            mean_row_best['Split'] = 'Average'
            std_row_best = std_metrics_best.to_dict()
            std_row_best['Split'] = 'Std_Dev'
            df_kfold_best = pd.concat([df_kfold_best, pd.DataFrame([mean_row_best, std_row_best])], ignore_index=True)
            df_kfold_best.to_csv(os.path.join(base_kfold_dir, f"KFold_{config.k_folds}_Summary_BestTrainEpoch_rs{config.random_seed}.csv"), index=False)


if __name__ == "__main__":
    dropout_list = [0.5]
    attention_list = ['None']
    dataset_list = ['gamma']
    for dname in dataset_list:
        seed_list = [247, 483, 3516, 1, 111, 222, 333, 456, 789, 1010]
        alpha_list = [0.02]
        loss_type_list = ["UAL"]
        mc_passes_train_list = [50]
        batch_size_list = [16]

        for alpha in alpha_list:
            for seed in seed_list:
                for attention in attention_list:
                    for dropout in dropout_list:
                        for loss_type in loss_type_list:
                            for batch_size in batch_size_list:
                                for mc_passes in mc_passes_train_list:
                                    main(current_seed=seed, dname=dname, alpha=alpha,
                                         branch_attention_type=attention, dropout=dropout,
                                         loss_type=loss_type, batch_size=batch_size, mc_passes_train=mc_passes)
                                    gc.collect()
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
