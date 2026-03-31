import os
from typing import List, Optional

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

    def forward_logits(self, images):
        raise NotImplementedError

    def predict(self, images):
        raise NotImplementedError

    def evaluate(self, images, targets=None):
        outputs = self.predict(images)
        metrics = summarize_ocr_metrics(
            outputs['texts'], targets or [], outputs.get('confidences')
        )
        outputs['metrics'] = metrics
        return outputs

    def teacher_loss(
        self,
        images,
        targets=None,
        reference_images=None,
        mode='distill',
        temperature=1.0,
        confidence_weighted=True,
    ):
        device = images.device if torch.is_tensor(images) else 'cpu'
        return torch.zeros((), device=device)

    def training_loss(self, images, targets):
        return self.teacher_loss(images, targets=targets)

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

    def _distill_from_reference(
        self,
        student_logits,
        reference_logits,
        temperature=1.0,
        confidence_weighted=True,
    ):
        teacher_prob = F.softmax(reference_logits / temperature, dim=-1)
        student_log_prob = F.log_softmax(student_logits / temperature, dim=-1)
        token_kl = F.kl_div(
            student_log_prob, teacher_prob, reduction='none'
        ).sum(dim=-1).mean(dim=-1)

        if confidence_weighted:
            weights = teacher_prob.max(dim=-1).values.mean(dim=-1).detach()
            token_kl = token_kl * weights

        return token_kl.mean() * (temperature ** 2)

    def _entropy_regularization(self, logits):
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
        return entropy.mean()


class _BaseGPLPROCRAdapter(BaseOCRAdapter):
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
        self.alphabet = alphabet
        self.K = K
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
            model_state = self.model.state_dict()
            filtered_state = {}
            for key, value in state_dict.items():
                if key in model_state and model_state[key].shape == value.shape:
                    filtered_state[key] = value
            self.model.load_state_dict(filtered_state, strict=False)

    def _prepare_images(self, images):
        if images.dim() == 3:
            images = images.unsqueeze(0)
        if images.size(1) == 1:
            images = images.repeat(1, 3, 1, 1)
        if images.size(-2) != self.img_size[0] or images.size(-1) != self.img_size[1]:
            images = F.interpolate(
                images, size=self.img_size, mode='bilinear', align_corners=False
            )
        return images.clamp(0.0, 1.0)

    def forward_logits(self, images):
        images = self._prepare_images(images)
        _, logits, _ = self.model(images)
        return logits

    def _encode_targets(self, targets, device):
        target_tensor = self.model.converter.encode_list(targets, K=self.K).to(device)
        return target_tensor

    def _cross_entropy_loss(self, logits, targets):
        target_tensor = self._encode_targets(targets, logits.device)
        bsz, steps, vocab = logits.shape
        per_token = F.cross_entropy(
            logits.reshape(bsz * steps, vocab),
            target_tensor.reshape(bsz * steps),
            reduction='none',
        ).view(bsz, steps)
        valid = (target_tensor > 0).float()
        denom = valid.sum().clamp_min(1.0)
        return (per_token * valid).sum() / denom

    def predict(self, images):
        images = self._prepare_images(images)
        texts, confidences = self.model.OCR_pred(images)
        return {
            'texts': texts,
            'confidences': self._normalize_confidence(confidences),
            'logits': self.forward_logits(images),
        }


@register('gplpr_ocr')
class GPLPROCRAdapter(_BaseGPLPROCRAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.freeze()

    def teacher_loss(
        self,
        images,
        targets=None,
        reference_images=None,
        mode='distill',
        temperature=1.0,
        confidence_weighted=True,
    ):
        student_logits = self.forward_logits(images)
        if reference_images is not None and mode == 'distill':
            with torch.no_grad():
                reference_logits = self.forward_logits(reference_images)
            return self._distill_from_reference(
                student_logits,
                reference_logits,
                temperature=temperature,
                confidence_weighted=confidence_weighted,
            )

        if targets:
            return self._cross_entropy_loss(student_logits, targets)

        return self._entropy_regularization(student_logits)


@register('gplpr_trainable_ocr')
class GPLPRTrainableOCRAdapter(_BaseGPLPROCRAdapter):
    def predict(self, images):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            outputs = super().predict(images)
        if was_training:
            self.train()
        return outputs

    def teacher_loss(
        self,
        images,
        targets=None,
        reference_images=None,
        mode='ce',
        temperature=1.0,
        confidence_weighted=True,
    ):
        logits = self.forward_logits(images)

        if mode == 'distill' and reference_images is not None:
            with torch.no_grad():
                reference_logits = self.forward_logits(reference_images)
            return self._distill_from_reference(
                logits,
                reference_logits,
                temperature=temperature,
                confidence_weighted=confidence_weighted,
            )

        if not targets:
            raise ValueError('Trainable OCR mode requires targets for supervision')

        return self._cross_entropy_loss(logits, targets)

    def training_loss(self, images, targets):
        return self.teacher_loss(images, targets=targets, mode='ce')


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

    def _get_local_repo_candidates(self):
        candidates = []
        if os.path.isdir(self.hub_repo):
            candidates.append(self.hub_repo)

        if self.hub_repo == 'baudm/parseq':
            candidates.extend(
                [
                    '/home/re6141029/.cache/torch/hub/baudm_parseq_main',
                    os.path.expanduser('~/.cache/torch/hub/baudm_parseq_main'),
                    '/tmp/torch_cache/baudm_parseq_main',
                ]
            )

        unique = []
        for path in candidates:
            if path not in unique and os.path.isdir(path):
                unique.append(path)
        return unique

    def _load_hub_model(self, repo_or_dir, source):
        kwargs = {}
        if source == 'local':
            kwargs['source'] = 'local'
        return torch.hub.load(
            repo_or_dir,
            self.model_name,
            pretrained=self.pretrained if self.load is None else False,
            **kwargs,
        )

    def _build_model(self):
        torch.hub.set_dir(os.environ.get('TORCH_HOME', '/tmp/torch_cache'))

        load_attempts = []
        if self.source == 'local':
            load_attempts.extend((path, 'local') for path in self._get_local_repo_candidates())
        else:
            load_attempts.append((self.hub_repo, self.source))
            load_attempts.extend((path, 'local') for path in self._get_local_repo_candidates())

        last_error: Optional[Exception] = None
        model = None
        for repo_or_dir, source in load_attempts:
            try:
                model = self._load_hub_model(repo_or_dir, source)
                break
            except Exception as exc:
                last_error = exc

        if model is None:
            raise RuntimeError(
                f'Unable to load PARSeq from {self.hub_repo!r} or local cache candidates.'
            ) from last_error

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

    def forward_logits(self, images):
        images = self._prepare_images(images)
        return self.model(images)

    def predict(self, images):
        logits = self.forward_logits(images)
        probs = logits.softmax(-1)
        texts, confidence = self.model.tokenizer.decode(probs)
        return {
            'texts': texts,
            'confidences': self._normalize_confidence(confidence),
            'logits': logits,
        }

    def teacher_loss(
        self,
        images,
        targets=None,
        reference_images=None,
        mode='distill',
        temperature=1.0,
        confidence_weighted=True,
    ):
        student_logits = self.forward_logits(images)

        if reference_images is not None and mode == 'distill':
            with torch.no_grad():
                reference_logits = self.forward_logits(reference_images)
            return self._distill_from_reference(
                student_logits,
                reference_logits,
                temperature=temperature,
                confidence_weighted=confidence_weighted,
            )

        # Fallback for settings without paired reference images:
        # keep the OCR distribution sharp so SR does not collapse into low-confidence text.
        return self._entropy_regularization(student_logits)
