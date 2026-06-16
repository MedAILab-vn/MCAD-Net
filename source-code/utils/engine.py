import os
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, roc_auc_score,
                             f1_score, recall_score, precision_score, matthews_corrcoef,
                             average_precision_score, confusion_matrix, multilabel_confusion_matrix)


# =====================================================================
# 1. HÀM KHÓA SỰ NGẪU NHIÊN (REPRODUCIBILITY)
# =====================================================================
def set_seed(seed=42):
    """
    Khóa chặt tất cả các cơ chế sinh số ngẫu nhiên của Python, Numpy, PyTorch và CUDA.
    Đảm bảo mỗi lần chạy với cùng một seed sẽ cho ra kết quả y hệt nhau.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ép cuDNN tính toán theo cách xác định (deterministic)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"\n[HỆ THỐNG] Đã khóa toàn cục Random Seed ở mức: {seed}")


# =====================================================================
# 2. CÔNG CỤ HỖ TRỢ ĐÁNH GIÁ (Không dùng nữa)
# =====================================================================
def enable_dropout(model):
    """
    Bật lại các lớp Dropout trong quá trình Evaluation để phục vụ tính toán MC-Dropout.
    """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


# =====================================================================
# 3. HÀM ĐÁNH GIÁ CHÍNH & TÍNH ĐỘ BẤT ĐỊNH (UNCERTAINTY)
# =====================================================================
def evaluate_with_uncertainty(model, loader, criterion, device, num_classes, split_name, mc_passes=1,
                              entropy_threshold=0.5, use_cfp_only=False, is_multilabel=False):
    model.eval()

    # Tự động chuyển sang chế độ Đa nhãn nếu số class = 11 (TOPCON-MM)
    if num_classes == 11:
        is_multilabel = True

    if mc_passes > 1:
        enable_dropout(model)

    running_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []
    entropies = []
    hus_count = 0

    with torch.no_grad():
        for inputs_CFP, inputs_OCT, labels in tqdm(loader, desc=f"Evaluating {split_name}", leave=False):
            inputs_CFP, inputs_OCT, labels = inputs_CFP.to(device), inputs_OCT.to(device), labels.to(device)

            if mc_passes > 1:
                batch_probs = []
                for _ in range(mc_passes):
                    outputs = model(inputs_CFP, inputs_OCT)
                    if is_multilabel:
                        batch_probs.append(torch.sigmoid(outputs))
                    else:
                        batch_probs.append(F.softmax(outputs, dim=1))

                mean_probs = torch.stack(batch_probs).mean(dim=0)
                probs = mean_probs

                # Tính Loss và Entropy cho MC-Dropout
                if is_multilabel:
                    loss = F.binary_cross_entropy(probs, labels)
                    # Binary Entropy: H = -p*log(p) - (1-p)*log(1-p)
                    eps = 1e-8
                    entropy = - (probs * torch.log(probs + eps) + (1 - probs) * torch.log(1 - probs + eps))
                    entropy = torch.mean(entropy, dim=1)  # Trung bình entropy của 11 bệnh
                else:
                    log_probs = torch.log(probs + 1e-8)
                    loss = F.nll_loss(log_probs, labels)
                    entropy = -torch.sum(probs * log_probs, dim=1)

                entropies.extend(entropy.cpu().numpy())
                hus_count += torch.sum(entropy > entropy_threshold).item()

            else:
                outputs = model(inputs_CFP, inputs_OCT)
                loss = criterion(outputs, labels)
                if is_multilabel:
                    probs = torch.sigmoid(outputs)
                else:
                    probs = F.softmax(outputs, dim=1)

            running_loss += loss.item() * inputs_CFP.size(0)

            # Phân loại (Threshold)
            if is_multilabel:
                preds = (probs > 0.5).int()  # Đa nhãn: Bệnh nào > 0.5 thì lấy
            else:
                _, preds = torch.max(probs, 1)  # Đa lớp: Lấy bệnh cao điểm nhất

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    total_loss = running_loss / len(loader.dataset)
    all_probs, all_labels, all_preds = np.array(all_probs), np.array(all_labels), np.array(all_preds)

    # =====================================================================
    # 4. TÍNH TOÁN METRICS PHÂN NHÁNH: ĐA NHÃN (MULTI-LABEL) VS ĐA LỚP
    # =====================================================================
    specs, class_accs = [], {}

    if is_multilabel:
        # === ĐÁNH GIÁ MULTI-LABEL ===
        # Dùng trung bình Macro cho toàn bộ các metric cốt lõi
        acc = accuracy_score(all_labels, all_preds)  # Exact match (khá khắt khe)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        sen = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        pre = precision_score(all_labels, all_preds, average='macro', zero_division=0)

        # Confusion Matrix cho từng class (2x2)
        mcm = multilabel_confusion_matrix(all_labels, all_preds)
        for i in range(num_classes):
            tn, fp, fn, tp = mcm[i].ravel()
            spec_i = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            acc_i = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
            specs.append(spec_i)
            class_accs[f'Acc_Class_{i}'] = acc_i

        spe = np.mean(specs)
        bacc = (sen + spe) / 2.0  # Tính trung bình Sen và Spe

        # MCC trung bình
        mcc_list = [matthews_corrcoef(all_labels[:, i], all_preds[:, i]) for i in range(num_classes)]
        mcc = np.mean(mcc_list)

        try:
            auc = roc_auc_score(all_labels, all_probs, average='macro')
            ap = average_precision_score(all_labels, all_probs, average='macro')
        except Exception as e:
            print(f"\n[Cảnh báo] Không thể tính AUC/AP đa nhãn. Lỗi: {e}")
            auc, ap = 0.0, 0.0

    else:
        # === ĐÁNH GIÁ MULTI-CLASS (GIỮ NGUYÊN CODE CŨ) ===
        acc = accuracy_score(all_labels, all_preds)
        bacc = balanced_accuracy_score(all_labels, all_preds)
        mcc = matthews_corrcoef(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        sen = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        pre = precision_score(all_labels, all_preds, average='macro', zero_division=0)

        cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
        class_names = [f'Class_{i}' for i in range(num_classes)]

        for i in range(num_classes):
            tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
            fp = np.sum(cm[:, i]) - cm[i, i]
            specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
            total_class_i = np.sum(cm[i, :])
            if num_classes == 3:
                class_accs[class_names[i]] = cm[i, i] / total_class_i if total_class_i > 0 else 0.0
            else:
                class_accs[f'Acc_{class_names[i]}'] = cm[i, i] / total_class_i if total_class_i > 0 else 0.0

        spe = np.mean(specs)

        try:
            if num_classes == 2:
                auc = roc_auc_score(all_labels, all_probs[:, 1])
                ap = average_precision_score(all_labels, all_probs[:, 1])
            else:
                auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
                labels_onehot = np.eye(num_classes)[all_labels]
                ap = average_precision_score(labels_onehot, all_probs, average="macro")
        except Exception as e:
            print(f"\n[Cảnh báo] Không thể tính AUC/AP. Lỗi: {e}")
            auc, ap = 0.0, 0.0

    # =====================================================================
    # 5. ĐÓNG GÓI KẾT QUẢ ĐẦU RA
    # =====================================================================
    metrics = {'Split': split_name, 'Loss': total_loss, 'Acc': acc, 'Sen': sen, 'Pre': pre, 'Spe': spe,
               'F1-score': f1, 'BAcc': bacc, 'MCC': mcc, 'AP': ap, 'AUC': auc}
    metrics.update(class_accs)

    metrics['Mean_Entropy'] = np.mean(entropies) if mc_passes > 1 else 0.0
    metrics['HUS_Count'] = hus_count if mc_passes > 1 else 0

    return metrics