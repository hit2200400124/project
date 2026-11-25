import numpy as np
from sklearn import metrics
from torch import nn
from model.clip.clip import load
import torch
from model.adapters.adapter import Adapter
from model.adapters.conv_moe import ConvMoE
from model.adapters.frequency import FrequencyFusion
from .attn import RecAttnClip
from .layer import PostClipProcess, MaskPostXrayProcess
import torch.nn.functional as F
from trainer.metrics.base_metrics_class import calculate_metrics_for_train


def moe_route_metrics(weights):
    rates, mean_weights = torch.stack(((weights > 0).float().mean(dim=0), weights.mean(dim=0))).tolist()
    metrics = {}
    for expert, rate, weight in zip(ConvMoE.expert_names, rates, mean_weights):
        metrics[f'moe_{expert}_rate'] = rate
        metrics[f'moe_{expert}_weight'] = weight
    return metrics


class DS(nn.Module):
    def __init__(self, clip_name,
                 adapter_vit_name,
                 num_quires,
                 fusion_map,
                 mlp_dim,
                 mlp_out_dim,
                 head_num,
                 device,
                 mode='video',
                 frequency_fusion="off",
                 clip_download_root=None,
                 conv_moe_mode="off",
                 conv_moe_balance_weight=0.01,
                 conv_moe_top_k=1):
        super().__init__()
        self.device = device
        self.conv_moe_balance_weight = conv_moe_balance_weight
        self.clip_model, self.processor = load(clip_name, device=device, download_root=clip_download_root)
        self.adapter = Adapter(vit_name=adapter_vit_name, num_quires=num_quires, fusion_map=fusion_map, mlp_dim=mlp_dim,
                               mlp_out_dim=mlp_out_dim, head_num=head_num, device=self.device,
                               clip_dim=self.clip_model.visual.width, frequency_fusion=frequency_fusion,
                               conv_moe_mode=conv_moe_mode, conv_moe_top_k=conv_moe_top_k)
        self.rec_attn_clip = RecAttnClip(self.clip_model.visual, num_quires,device=self.device)  # 全部参数被冻结
        self.masked_xray_post_process = MaskPostXrayProcess(in_c=num_quires).to(self.device)
        self.clip_post_process = PostClipProcess(num_quires=num_quires, embed_dim=768)

        self.prob, self.label = [], []
        self.correct, self.total = 0, 0
        self.mode = mode
        self._freeze()


    def _freeze(self):
        for name, param in self.named_parameters():
            if 'clip_model' in name :
                param.requires_grad = False

    def get_losses(self, data_dict, pred_dict):
        label = data_dict['label']
        xray = data_dict.get('xray')
        loss_cls = F.cross_entropy(pred_dict['cls'].float(), label)
        loss_dict = {'cls': loss_cls, 'overall': loss_cls}
        if xray is not None:
            loss_mse = F.mse_loss(pred_dict['xray_pred'].float(), xray.float())
            loss_intra = pred_dict['loss_intra']
            loss_clip = pred_dict['loss_clip']
            loss_dict.update({
                'xray': loss_mse,
                'intra': loss_intra,
                'loss_clip': loss_clip,
                'overall': 10 * loss_cls + 200 * loss_mse + 20 * loss_intra + 10 * loss_clip,
            })
        if 'coverage_logits' in pred_dict:
            loss_coverage = FrequencyFusion.coverage_loss(
                pred_dict['coverage_logits'], label, data_dict.get('mask'), data_dict.get('mask_available')
            )
            loss_dict['coverage'] = loss_coverage
            loss_dict['overall'] = loss_dict['overall'] + loss_coverage
        if 'moe_balance' in pred_dict:
            loss_dict['moe_balance'] = pred_dict['moe_balance']
            loss_dict['overall'] = loss_dict['overall'] + self.conv_moe_balance_weight * loss_dict['moe_balance']
        return loss_dict

    def get_train_metrics(self, data_dict, pred_dict):
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        if 'moe_route_weights' in pred_dict:
            metric_batch_dict.update(moe_route_metrics(pred_dict['moe_route_weights']))
        return metric_batch_dict

    def get_test_metrics(self):
        y_pred = np.concatenate(self.prob)
        y_true = np.concatenate(self.label)
        # auc
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        # eer
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        # ap
        ap = metrics.average_precision_score(y_true, y_pred)
        # acc
        acc = self.correct / self.total
        # reset the prob and label
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap, 'pred': y_pred, 'label': y_true}

    def forward(self, data_dict, inference=False):
        images = data_dict['image']
        clip_images = F.interpolate(
            images,
            size=(224, 224),
            mode='bilinear',
            align_corners=False,
        )
        clip_features = self.clip_model.extract_features(clip_images, self.adapter.fusion_map.values())
        attn_biases, xray_preds, loss_adapter_intra, adapter_outputs = self.adapter(
            data_dict, clip_features, inference
        )
        clip_output, loss_clip = self.rec_attn_clip(
            data_dict, clip_features, attn_biases[-1], inference, normalize=True
        )
        if_boundary = data_dict.get('if_boundary')
        if if_boundary is not None:
            if_boundary = if_boundary.to(images.device)
        xray_pred = self.masked_xray_post_process(xray_preds[-1], if_boundary)

        clip_cls_output = self.clip_post_process(clip_output.float())
        prob = torch.softmax(clip_cls_output, dim=1)[:, 1]
        pred_dict = {
            'cls': clip_cls_output,
            'prob': prob,
            'xray_pred': xray_pred,
            'loss_intra': loss_adapter_intra,
            'loss_clip': loss_clip,
            **adapter_outputs,
        }

        if inference and data_dict.get('label') is not None:
            labels = data_dict['label']
            self.prob.append(prob.detach().cpu().numpy())
            self.label.append(labels.detach().cpu().numpy())
            self.correct += (clip_cls_output.argmax(dim=1) == labels).sum().item()
            self.total += labels.size(0)

        return pred_dict
