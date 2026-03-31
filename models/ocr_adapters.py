from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register
from .GP_LPR_arch import make_GPLPR


def align_prediction_to_gt(pred, gt):
    pred = pred.strip() if pred is not None else ''
    gt = gt.strip()
    gt_len = len(gt)

    pred_chars = list(pred[:gt_len])
    if len(pred_chars) < gt_len:
        pred_chars += [''] * (gt_len - len(pred_chars))
    return pred_chars, list(gt)


def calc_text_score_and_acc(pred, gt):
    pred_chars, gt_chars = align_prediction_to_gt(pred, gt)

    score = 0
    char_hits = 0
    for pred_ch, gt_ch in zip(pred_chars, gt_chars):
        if pred_ch == '':
            score += 0
        elif pred_ch == gt_ch:
            score += 2
            char_hits += 1
        else:
            score += -1

    exact = int(''.join(pred_chars) == gt)
    char_acc = char_hits / max(len(gt_chars), 1)
    return score, exact, char_acc


def summarize_ocr_metrics(texts, targets, confidences=None):
    metrics = {
        'ocr_score': 0.0,
        'ocr_acc': 0.0,
        'ocr_char_acc': 0.0,
        'ocr_conf': 0.0,
    }
    if not targets:
        return metrics

    scores = []
    exacts = []
    char_accs = []
    for pred, gt in zip(texts, targets):
        score, exact, char_acc = calc_text_score_and_acc(pred, gt)
        scores.append(score)
        exacts.append(exact)
        char_accs.append(char_acc)

    metrics['ocr_score'] = float(sum(scores) / len(scores))
    metrics['ocr_acc'] = float(sum(exacts) / len(exacts))
    metrics['ocr_char_acc'] = float(sum(char_accs) / len(char_accs))

    if confidences is not None:
        if torch.is_tensor(confidences):
            metrics['ocr_conf'] = float(confidences.float().mean().item())
        elif confidences:
            metrics['ocr_conf'] = float(sum(confidences) / len(confidences))

    return metrics


class BaseOCRAdapter(nn.Module):
    def __init__(self):
        super().__init__()

    def freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
        return self

    def OCR_pred(self, images):
        outputs = self.predict(images)
        return outputs['texts'], outputs['confidences']

    def predict(self, images):
        raise NotImplementedError

    def evaluate(self, images, targets=None):
        outputs = self.predict(images)
        metrics = summarize_ocr_metrics(
            outputs['texts'], targets or [], outputs.get('confidences')
        )
        outputs['metrics'] = metrics
        return outputs

    def teacher_loss(self, images, targets):
        device = images.device if torch.is_tensor(images) else 'cpu'
        return torch.zeros((), device=device)


@register('gplpr_ocr')
class GPLPROCRAdapter(BaseOCRAdapter):
    def __init__(
        self,
        load=None,
        alphabet='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',
        nc=3,
        imgW=96,
        imgH=32,
        K=7,
        isSeqModel=True,
        head=2,
        inner=256,
        isl2Norm=True,
    ):
        super().__init__()
        self.model = make_GPLPR(
            alphabet=alphabet,
            nc=nc,
            imgW=imgW,
            imgH=imgH,
            K=K,
            isSeqModel=isSeqModel,
            head=head,
            inner=inner,
            isl2Norm=isl2Norm,
        )
        self.img_size = (imgH, imgW)

        if load is not None:
            ckpt = torch.load(load, map_location='cpu')
            state_dict = ckpt
            if isinstance(ckpt, dict) and 'model' in ckpt:
                state_dict = ckpt['model'].get('sd', ckpt['model'])
            self.model.load_state_dict(state_dict, strict=False)

        self.freeze()

    def predict(self, images):
        if images.size(-2) != self.img_size[0] or images.size(-1) != self.img_size[1]:
            images = F.interpolate(
                images, size=self.img_size, mode='bilinear', align_corners=False
            )
        texts, confidences = self.model.OCR_pred(images)
        if torch.is_tensor(confidences):
            confidences = confidences.detach().cpu()
        return {
            'texts': texts,
            'confidences': confidences,
            'logits': None,
        }


@register('parseq_ocr')
class PARSeqOCRAdapter(BaseOCRAdapter):
    def __init__(
        self,
        hub_repo='baudm/parseq',
        model_name='parseq',
        pretrained=True,
        load=None,
        img_size=None,
        source='github',
    ):
        super().__init__()
        self.hub_repo = hub_repo
        self.model_name = model_name
        self.pretrained = pretrained
        self.load = load
        self.source = source
        self.model = self._build_model()
        default_size = getattr(getattr(self.model, 'hparams', None), 'img_size', None)
        self.img_size = tuple(default_size or img_size or (32, 128))
        self.freeze()

    def _build_model(self):
        hub_kwargs = {'source': 'local'} if self.source == 'local' else {}
        model = torch.hub.load(
            self.hub_repo,
            self.model_name,
            pretrained=self.pretrained if self.load is None else False,
            **hub_kwargs,
        )

        if self.load is not None:
            ckpt = torch.load(self.load, map_location='cpu')
            state_dict = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
            if isinstance(state_dict, dict):
                cleaned = {}
                for key, value in state_dict.items():
                    if key.startswith('model.'):
                        cleaned[key[len('model.'):]] = value
                    else:
                        cleaned[key] = value
                state_dict = cleaned
            model.load_state_dict(state_dict, strict=False)

        return model.eval()

    def _prepare_images(self, images):
        if images.dim() == 3:
            images = images.unsqueeze(0)
        if images.size(1) == 1:
            images = images.repeat(1, 3, 1, 1)
        if images.size(-2) != self.img_size[0] or images.size(-1) != self.img_size[1]:
            images = F.interpolate(
                images, size=self.img_size, mode='bilinear', align_corners=False
            )
        images = images.clamp(0.0, 1.0)
        images = (images - 0.5) / 0.5
        return images

    def _normalize_confidence(self, confidence) -> torch.Tensor:
        if torch.is_tensor(confidence):
            return confidence.detach().cpu().float()
        if isinstance(confidence, (list, tuple)):
            values: List[float] = []
            for item in confidence:
                if torch.is_tensor(item):
                    values.append(float(item.detach().cpu().float().mean().item()))
                else:
                    values.append(float(item))
            return torch.tensor(values, dtype=torch.float32)
        return torch.tensor([float(confidence)], dtype=torch.float32)

    def predict(self, images):
        images = self._prepare_images(images)
        logits = self.model(images)
        probs = logits.softmax(-1)
        texts, confidence = self.model.tokenizer.decode(probs)
        return {
            'texts': texts,
            'confidences': self._normalize_confidence(confidence),
            'logits': logits,
        }
