import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss cho bài toán đa lớp mất cân bằng dữ liệu.
    Dồn trọng tâm vào các mẫu khó (hard examples) bằng tham số gamma.
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Có thể là 1 tensor chứa trọng số của từng class
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [Batch, Classes] - Logits chưa qua softmax
        # targets: [Batch] - Nhãn integer

        # 1. Tính Cross Entropy Loss tiêu chuẩn (không reduction để lấy từng phần tử)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # 2. Lấy xác suất dự đoán của class đúng (pt)
        pt = torch.exp(-ce_loss)

        # 3. Tính Focal Loss: Nhân thêm hệ số phạt (1 - pt)^gamma
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        # 4. Áp dụng trọng số class (Alpha) nếu có
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        # 5. Trả về kết quả
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class MultimodalSupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, vec_cfp, vec_oct, labels):
        device = vec_cfp.device
        batch_size = vec_cfp.shape[0]

        # 1. Tạo Super Batch: Nối dọc CFP và OCT -> Kích thước: [2B, 256]
        features = torch.cat([vec_cfp, vec_oct], dim=0)

        # 2. Nhân đôi nhãn để khớp với Super Batch -> Kích thước: [2B]
        super_labels = torch.cat([labels, labels], dim=0)

        # 3. Tính ma trận tương đồng (Cosine Similarity)
        # Vì vec_cfp, vec_oct đã được L2 Normalize, tích vô hướng chính là Cosine
        sim_matrix = torch.div(torch.matmul(features, features.T), self.temperature)

        # 4. Tạo Mặt nạ Nhãn (Label Mask)
        super_labels = super_labels.contiguous().view(-1, 1)
        mask = torch.eq(super_labels, super_labels.T).float().to(device)

        # Loại bỏ đường chéo chính (Không tự đối chiếu với chính mình)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * 2).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # 5. Tính toán SupCon Loss chuẩn học thuật
        # Trừ đi max để tránh tràn số (Numerical stability)
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - sim_max.detach()

        # Tính mẫu số và Log Xác suất
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        # Tính trung bình log-prob trên các positive pairs (cùng nhãn)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)

        # Hàm Loss cuối cùng
        loss = - mean_log_prob_pos.mean()

        return loss

class UncertaintyAwareLoss(nn.Module):
    """
    Uncertainty-Aware Loss (UAL) áp dụng cho MC-Dropout lúc Train.
    """

    def __init__(self, alpha=0.1, label_smoothing=0.0):
        super(UncertaintyAwareLoss, self).__init__()
        self.alpha = alpha  # Thêm biến nhận hệ số phạt

    def forward(self, stacked_logits, labels):
        # ========================================================
        # KHẮC PHỤC LỖI CRASH DIMENSION LÚC VALIDATION
        # ========================================================
        # Nếu đầu vào chỉ là 2D [Batch, Class] (do lúc Val chỉ chạy 1 lần)
        if stacked_logits.dim() == 2:
            # Ép thành 3D [1, Batch, Class]
            stacked_logits = stacked_logits.unsqueeze(0)

            # Lúc này stacked_logits LUÔN LUÔN là 3D: [M_passes, Batch, Classes]

        # 1. Tính trung bình cộng xác suất từ M lần chạy dọc theo trục 0 (M_passes)
        # Kết quả trả về sẽ luôn là [Batch, Classes]
        mean_probs = torch.softmax(stacked_logits, dim=-1).mean(dim=0)

        # Thêm 1e-8 để tránh lỗi log(0)
        log_mean_probs = torch.log(mean_probs + 1e-8)

        # 2. Tính Standard Cross Entropy Loss (NLL)
        ce_loss = F.nll_loss(log_mean_probs, labels)

        # 3. Tính Predictive Entropy (PE)
        entropy = -torch.sum(mean_probs * log_mean_probs, dim=1)  # Trả về [Batch]
        mean_entropy = torch.mean(entropy)

        # 4. Loss tổng: Phạt sự hoang mang (Có áp dụng hệ số kìm hãm alpha)
        total_loss = ce_loss + (self.alpha * mean_entropy)
        return total_loss


class UncertaintyAwareLossSmoothing(nn.Module):
    """
    Uncertainty-Aware Loss (UAL) áp dụng MC-Dropout và Label Smoothing.
    """

    def __init__(self, alpha=0.1, label_smoothing=0.0):
        super(UncertaintyAwareLossSmoothing, self).__init__()
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, stacked_logits, labels):
        # 1. Ép chiều dữ liệu nếu đang ở mode Evaluation (chỉ có 1 pass)
        if stacked_logits.dim() == 2:
            stacked_logits = stacked_logits.unsqueeze(0)

        # 2. Tính xác suất trung bình từ M lần chạy MC-Dropout
        mean_probs = torch.softmax(stacked_logits, dim=-1).mean(dim=0)
        log_mean_probs = torch.log(mean_probs + 1e-8)

        # 3. TÍNH CROSS ENTROPY (CÓ THỂ BẬT TẮT LABEL SMOOTHING)
        if self.label_smoothing > 0.0:
            num_classes = mean_probs.size(-1)
            # Khởi tạo ma trận nhãn mềm (ví dụ 4 class, smoothing=0.1 -> các class sai nhận 0.1/3 = 0.0333)
            true_dist = torch.zeros_like(log_mean_probs)
            true_dist.fill_(self.label_smoothing / (num_classes - 1))

            # Gán xác suất cho class đúng (nhận 1.0 - 0.1 = 0.9)
            true_dist.scatter_(1, labels.unsqueeze(1), 1.0 - self.label_smoothing)

            # Tính CE Loss thủ công: Tổng( - Nhãn_mềm * log_Xác_suất )
            ce_loss = torch.mean(-torch.sum(true_dist * log_mean_probs, dim=-1))
        else:
            # Nếu smoothing = 0, quay về dùng NLL chuẩn cho nhanh
            ce_loss = F.nll_loss(log_mean_probs, labels)

        # 4. Tính Predictive Entropy (PE)
        entropy = -torch.sum(mean_probs * log_mean_probs, dim=-1)  # Trả về [Batch]
        mean_entropy = torch.mean(entropy)

        # 5. Loss tổng
        total_loss = ce_loss + (self.alpha * mean_entropy)
        return total_loss


class ScaleCS_Loss(nn.Module):
    """
    Loss mới cho MCAT_Model_ScaleCS.
    Tổng hợp: Base_Loss (CE hoặc UAL) - Alpha * Cosine_Similarity
    """

    def __init__(self, base_loss='CE', ual_alpha=0.1, cos_alpha=0.05, label_smoothing=0.0):
        super(ScaleCS_Loss, self).__init__()
        self.cos_alpha = cos_alpha
        self.base_loss = base_loss

        if base_loss == 'UAL':
            self.cls_criterion = UncertaintyAwareLossSmoothing(alpha=ual_alpha, label_smoothing=label_smoothing)
        elif base_loss == 'CE':
            self.cls_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            raise ValueError(f"Không hỗ trợ base_loss: {base_loss} trong ScaleCS_Loss")

    # Đưa labels lên trước, cho phép cos_sim=None lúc chạy Validation
    def forward(self, logits, labels, cos_sim=None):
        # 1. Tính Classification Loss gốc
        cls_loss = self.cls_criterion(logits, labels)

        # 2. Nếu không có cos_sim (lúc chạy Val/Test qua engine.py), chỉ trả về cls_loss
        if cos_sim is None:
            return cls_loss

        # 3. Lúc Train: Trừ đi phần thưởng Cosine
        mean_cos_sim = cos_sim.mean()
        total_loss = cls_loss - (self.cos_alpha * mean_cos_sim)
        return total_loss

def build_loss_function(loss_type, device, label_smoothing=0.1, alpha=0.1):
    """
    Hàm Factory tự động nạp Loss theo file Config.
    """
    if loss_type == 'CE':
        print(f"-> Kích hoạt Standard Cross Entropy Loss (Smoothing: {label_smoothing})")
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing).to(device)

    elif loss_type == 'UAL':
        print(f"-> Kích hoạt Uncertainty-Aware Loss (Yêu cầu bật MC-Dropout lúc Train | Alpha: {alpha})")
        #return UncertaintyAwareLoss(alpha=alpha).to(device)
        return UncertaintyAwareLossSmoothing(alpha=alpha, label_smoothing=label_smoothing).to(device)
    elif loss_type == 'Focal':
        # Mặc định dùng gamma=2.0 là thông số tối ưu nhất theo paper gốc
        print(f"-> Kích hoạt Focal Loss (Gamma: 2.0 để trị mất cân bằng dữ liệu)")
        return FocalLoss(gamma=2.0).to(device)
    elif loss_type == 'ScaleCS_CE':
        print(f"-> Kích hoạt ScaleCS Loss (Base: CE | Cosine_Alpha: {alpha})")
        return ScaleCS_Loss(base_loss='CE', cos_alpha=alpha, label_smoothing=label_smoothing).to(device)
    else:
        raise ValueError(f"[LỖI] Không tìm thấy kiến trúc Loss: '{loss_type}'")