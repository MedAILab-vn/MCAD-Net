import torch
import torch.nn as nn
import timm
import torchvision.models as tv_models
from models.attention_blocks import build_attention_block

class DistilledResNet18_Features(nn.Module):
    def __init__(self, checkpoint_path, unfreeze_all=False, freeze_scale1=False):
        super().__init__()
        resnet = tv_models.resnet18(pretrained=False)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        print(f"\n[System] Loading distillation weights for OCT branch from:\n{checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        self.backbone.load_state_dict(state_dict, strict=True)

        if unfreeze_all:
            print("[System] UNFREEZE ALL for OCT branch (ResNet18).")
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            fid = 5
            if freeze_scale1:
                fid = 6
            print("[System] Executing Partial Freezing for OCT branch:")
            for idx, child in enumerate(self.backbone.children()):
                if idx <= fid:
                    for param in child.parameters():
                        param.requires_grad = False
                else:
                    for param in child.parameters():
                        param.requires_grad = True

    def forward(self, x):
        features = []
        for idx, child in enumerate(self.backbone.children()):
            x = child(x)
            if idx == 6:
                features.append(x)
            elif idx == 7:
                features.append(x)
                break
        return features

class DistilledEfficientNetB0_Features(nn.Module):
    def __init__(self, checkpoint_path, unfreeze_all=False, freeze_scale1=False):
        super().__init__()
        effnet = tv_models.efficientnet_b0(pretrained=False)
        self.backbone = nn.Sequential(*list(effnet.children())[:-1])

        print(f"\n[System] Loading distillation weights for CFP branch from:\n{checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        self.backbone.load_state_dict(state_dict, strict=True)

        if unfreeze_all:
            print("[System] UNFREEZE ALL for OCT branch (ResNet18).")
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            print("[System] Executing Partial Freezing for CFP branch (EfficientNet-B0):")
            fid = 4
            if freeze_scale1:
                fid = 5
            features = self.backbone[0]
            for idx, child in enumerate(features.children()):
                if idx <= fid:
                    for param in child.parameters():
                        param.requires_grad = False
                    print(f"  -> LOCKED (Freeze) block {idx}")
                else:
                    for param in child.parameters():
                        param.requires_grad = True
                    print(f"  -> UNLOCKED (Train) block {idx}")

    def forward(self, x):
        features_out = []
        features_module = self.backbone[0]

        for idx, child in enumerate(features_module.children()):
            x = child(x)
            if idx == 5:
                features_out.append(x)
            elif idx == 7:
                features_out.append(x)
                break

        return features_out

class MCAD_CrossAtt_Block(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8):
        super().__init__()

        self.cross_cfp_to_oct = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_oct_to_cfp = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.norm_q_cfp = nn.LayerNorm(embed_dim)
        self.norm_q_oct = nn.LayerNorm(embed_dim)
        self.norm_kv_cfp = nn.LayerNorm(embed_dim)
        self.norm_kv_oct = nn.LayerNorm(embed_dim)

        self.channel_fusion = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, q_cfp, q_oct, list_kvs_cfp, list_kvs_oct):
        B, C, H_q, W_q = q_cfp.shape

        flat_q_cfp = q_cfp.flatten(2).transpose(1, 2)
        flat_q_oct = q_oct.flatten(2).transpose(1, 2)
        flat_kvs_cfp = torch.cat([kv.flatten(2).transpose(1, 2) for kv in list_kvs_cfp], dim=1)
        flat_kvs_oct = torch.cat([kv.flatten(2).transpose(1, 2) for kv in list_kvs_oct], dim=1)

        n_q_cfp = self.norm_q_cfp(flat_q_cfp)
        n_kv_oct = self.norm_kv_oct(flat_kvs_oct)
        n_q_oct = self.norm_q_oct(flat_q_oct)
        n_kv_cfp = self.norm_kv_cfp(flat_kvs_cfp)

        attn_cfp, _ = self.cross_cfp_to_oct(query=n_q_cfp, key=n_kv_oct, value=n_kv_oct)
        attn_oct, attn_weights_oct2cfp = self.cross_oct_to_cfp(query=n_q_oct, key=n_kv_cfp, value=n_kv_cfp)

        self.last_attention_map_oct2cfp = attn_weights_oct2cfp.detach()

        attn_cfp_2d = attn_cfp.transpose(1, 2).reshape(B, C, H_q, W_q)
        attn_oct_2d = attn_oct.transpose(1, 2).reshape(B, C, H_q, W_q)

        out_cfp = attn_cfp_2d + q_cfp
        out_oct = attn_oct_2d + q_oct

        concat_feat = torch.cat([out_cfp, out_oct], dim=1)

        fused = self.channel_fusion(concat_feat)

        return fused

class MCAD_Model_SynScale2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        num_classes = getattr(config, 'num_classes', 4)
        embed_dim = getattr(config, 'embed_dim', 256)
        dropout_rate = getattr(config, 'dropout', 0.5)

        pretrained_oct = getattr(config, 'pretrained_model_oct', False)
        pretrained_oct_pth = getattr(config, 'pretrained_model_oct_pth', None)
        pretrained_cfp = getattr(config, 'pretrained_model_cfp', False)
        pretrained_cfp_pth = getattr(config, 'pretrained_model_cfp_pth', None)

        unfreeze_all = getattr(config, 'unfreeze_all', False)
        freeze_scale1 = getattr(config, 'freeze_scale1', False)

        if pretrained_cfp and pretrained_cfp_pth is not None:
            print('Load pretrained model DistilledEfficientNetB0_Features')
            self.backbone_CFP = DistilledEfficientNetB0_Features(checkpoint_path=pretrained_cfp_pth,
                                                                 unfreeze_all=unfreeze_all, freeze_scale1=freeze_scale1)
        else:
            self.backbone_CFP = timm.create_model('efficientnet_b0', pretrained=True, features_only=True,
                                                  out_indices=(3, 4))

        if pretrained_oct and pretrained_oct_pth is not None:
            print('Load pretrained model DistilledResNet18_Features')
            self.backbone_OCT = DistilledResNet18_Features(checkpoint_path=pretrained_oct_pth,
                                                           unfreeze_all=unfreeze_all, freeze_scale1=freeze_scale1)
        else:
            self.backbone_OCT = timm.create_model('resnet18', pretrained=True, features_only=True, out_indices=(3, 4))

        self.proj_cfp_s1 = nn.Conv2d(112, embed_dim, 1)
        self.proj_cfp_s2 = nn.Conv2d(320, embed_dim, 1)
        self.proj_oct_s1 = nn.Conv2d(256, embed_dim, 1)
        self.proj_oct_s2 = nn.Conv2d(512, embed_dim, 1)

        cfp_att_type = getattr(config, 'cfp_attention_type', None)
        oct_att_type = getattr(config, 'oct_attention_type', None)

        self.spatial_cfp_s1 = build_attention_block(attention_type=cfp_att_type, in_planes=embed_dim)
        self.spatial_cfp_s2 = build_attention_block(attention_type=cfp_att_type, in_planes=embed_dim)
        self.spatial_oct_s1 = build_attention_block(attention_type=oct_att_type, in_planes=embed_dim)
        self.spatial_oct_s2 = build_attention_block(attention_type=oct_att_type, in_planes=embed_dim)

        self.block_s2 = MCAD_CrossAtt_Block(embed_dim=embed_dim)

        fused_block_type = getattr(config, 'fused_block', 'EMA')
        self.fusion_attention = build_attention_block(attention_type=fused_block_type, in_planes=embed_dim)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x_CFP, x_OCT, mc_passes=1):
        f_cfp = self.backbone_CFP(x_CFP)
        f_oct = self.backbone_OCT(x_OCT)

        p_cfp_s1 = self.proj_cfp_s1(f_cfp[0])
        p_cfp_s2 = self.proj_cfp_s2(f_cfp[1])
        p_oct_s1 = self.proj_oct_s1(f_oct[0])
        p_oct_s2 = self.proj_oct_s2(f_oct[1])

        clean_cfp_s1 = self.spatial_cfp_s1(p_cfp_s1)
        clean_cfp_s2 = self.spatial_cfp_s2(p_cfp_s2)
        clean_oct_s1 = self.spatial_oct_s1(p_oct_s1)
        clean_oct_s2 = self.spatial_oct_s2(p_oct_s2)

        kvs_cfp = [clean_cfp_s1, clean_cfp_s2]
        kvs_oct = [clean_oct_s1, clean_oct_s2]

        refined_s2 = self.block_s2(q_cfp=clean_cfp_s2, q_oct=clean_oct_s2, list_kvs_cfp=kvs_cfp, list_kvs_oct=kvs_oct)

        attended_fused = self.fusion_attention(refined_s2)

        fused_features = self.pool(attended_fused).flatten(1)

        if mc_passes > 1:
            B = x_CFP.size(0)
            fused_mc = fused_features.repeat(mc_passes, 1)
            out_logits = self.classifier(fused_mc).view(mc_passes, B, -1)
            return out_logits

        out = self.classifier(fused_features)
        return out