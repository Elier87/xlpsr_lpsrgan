import argparse
import copy
import inspect
import random
import smtplib
from email.mime.text import MIMEText
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import datasets
import losses
import models
import train_funcs
import utils


torch.autograd.set_detect_anomaly(True)
if torch.cuda.is_available():
    torch.cuda.empty_cache()


def make_dataloader(spec, tag=''):
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    loader = DataLoader(
        dataset,
        batch_size=spec['batch'],
        shuffle=spec.get('shuffle', tag == 'train'),
        num_workers=spec.get('num_workers', 0),
        pin_memory=spec.get('pin_memory', False),
        collate_fn=dataset.collate_fn,
    )
    return loader


def make_dataloaders():
    train_loader = make_dataloader(config['train_dataset'], tag='train')
    val_loader = make_dataloader(config['val_dataset'], tag='val')
    return train_loader, val_loader


def _is_multi_model(model):
    return isinstance(model, (tuple, list))


def _move_model_to_cuda(model):
    if _is_multi_model(model):
        return tuple(module.cuda() for module in model)
    return model.cuda()


def _unwrap_model(module):
    return module.module if isinstance(module, nn.parallel.DataParallel) else module


def _get_model_for_save(model):
    if _is_multi_model(model):
        model_g = _unwrap_model(model[0])
        model_d = _unwrap_model(model[1]) if len(model) > 1 else None
        return model_g, model_d
    return _unwrap_model(model), None


def _make_optimizer_pair(model):
    return _make_optimizer_bundle(model)


def _get_single_optimizer_spec(config_obj):
    optimizer_spec = config_obj.get('optimizer', config_obj.get('optimizer_g'))
    if optimizer_spec is None:
        raise KeyError('Single-model training requires optimizer or optimizer_g')
    return optimizer_spec


def _make_optimizer_bundle(model, model_ocr=None):
    optimizers = []

    if _is_multi_model(model):
        optimizer_default = config.get('optimizer')
        optimizer_g_spec = config.get('optimizer_g', optimizer_default)
        optimizer_d_spec = config.get('optimizer_d', optimizer_default)
        if optimizer_g_spec is None or optimizer_d_spec is None:
            raise KeyError('Tuple model training requires optimizer_g and optimizer_d')
        optimizer_g = utils.make_optimizer(model[0].parameters(), optimizer_g_spec)
        optimizer_d = utils.make_optimizer(model[1].parameters(), optimizer_d_spec)
        optimizers.extend([optimizer_g, optimizer_d])
    else:
        optimizer = utils.make_optimizer(model.parameters(), _get_single_optimizer_spec(config))
        optimizers.append(optimizer)

    if model_ocr is not None and config.get('optimizer_ocr') is not None:
        ocr_params = [param for param in model_ocr.parameters() if param.requires_grad]
        if ocr_params:
            optimizer_ocr = utils.make_optimizer(ocr_params, config['optimizer_ocr'])
            optimizers.append(optimizer_ocr)

    if len(optimizers) == 1:
        return optimizers[0]
    return tuple(optimizers)


def _load_optimizer_pair(model, sv_file, model_ocr=None):
    optimizers = []

    if _is_multi_model(model):
        optimizer_g = utils.make_optimizer(
            model[0].parameters(), sv_file['optimizer_g'], load_optimizer=True
        )
        optimizer_d = utils.make_optimizer(
            model[1].parameters(), sv_file['optimizer_d'], load_optimizer=True
        )
        optimizers.extend([optimizer_g, optimizer_d])
    else:
        optimizer_spec = sv_file.get('optimizer', sv_file.get('optimizer_g'))
        optimizers.append(
            utils.make_optimizer(model.parameters(), optimizer_spec, load_optimizer=True)
        )

    if model_ocr is not None and 'optimizer_ocr' in sv_file:
        ocr_params = [param for param in model_ocr.parameters() if param.requires_grad]
        if ocr_params:
            optimizers.append(
                utils.make_optimizer(ocr_params, sv_file['optimizer_ocr'], load_optimizer=True)
            )

    if len(optimizers) == 1:
        return optimizers[0]
    return tuple(optimizers)


def _make_scheduler(optimizer, state_dict=None):
    if config.get('reduce_on_plateau') is None:
        return None

    scheduler_target = optimizer[0] if isinstance(optimizer, (tuple, list)) else optimizer
    scheduler_args = dict(config['reduce_on_plateau'])
    valid_keys = set(inspect.signature(ReduceLROnPlateau.__init__).parameters.keys())
    scheduler_args = {k: v for k, v in scheduler_args.items() if k in valid_keys}
    scheduler = ReduceLROnPlateau(scheduler_target, **scheduler_args)
    if state_dict is not None:
        scheduler.load_state_dict(state_dict)
    return scheduler


def _log_model_summary(model):
    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    if _is_multi_model(model):
        log('model_G: #struct={}'.format(model[0]))
        log('model_D: #struct={}'.format(model[1]))
    else:
        log('model: #struct={}'.format(model))


def _prepare_ocr_model():
    ocr_spec = config.get('model_ocr')
    if ocr_spec is None:
        return None

    model_ocr = models.make(ocr_spec)
    if model_ocr is None:
        return None
    if ocr_spec.get('freeze', True) and hasattr(model_ocr, 'freeze'):
        model_ocr.freeze()
    return model_ocr.cuda()


def _prepare_ocr_model_from_resume(sv_file=None):
    ocr_spec = config.get('model_ocr')
    if ocr_spec is None:
        return None

    if sv_file is not None and 'model_ocr' in sv_file:
        saved_ocr_spec = copy.deepcopy(sv_file['model_ocr'])
        model_ocr = models.make(saved_ocr_spec)
        if 'trainable_sd' in saved_ocr_spec and hasattr(model_ocr, 'load_trainable_state_dict'):
            model_ocr.load_trainable_state_dict(saved_ocr_spec['trainable_sd'])
        elif 'sd' in saved_ocr_spec:
            model_ocr.load_state_dict(saved_ocr_spec['sd'], strict=False)
    else:
        model_ocr = models.make(ocr_spec)

    if model_ocr is None:
        return None
    if ocr_spec.get('freeze', True) and hasattr(model_ocr, 'freeze'):
        model_ocr.freeze()
    return model_ocr.cuda()


def prepare_training():
    if config.get('resume') is not None:
        sv_file = torch.load(config['resume'], map_location='cpu')
        model = models.make(sv_file['model'], load_model=True)
        model = _move_model_to_cuda(model)
        model_ocr = _prepare_ocr_model_from_resume(sv_file)
        resume_mode = config.get('resume_mode', 'resume')
        needs_new_ocr_optimizer = (
            model_ocr is not None
            and config.get('optimizer_ocr') is not None
            and 'optimizer_ocr' not in sv_file
        )

        if resume_mode == 'finetune' or needs_new_ocr_optimizer:
            optimizer = _make_optimizer_bundle(model, model_ocr=model_ocr)
            lr_scheduler = _make_scheduler(optimizer)
            early_stopper = utils.Early_stopping(**config['early_stopper'])
            epoch_start = 1
            print(f'Initializing finetune from checkpoint: {config["resume"]}')
            log(f'Initializing finetune from checkpoint: {config["resume"]}')
        else:
            optimizer = _load_optimizer_pair(model, sv_file, model_ocr=model_ocr)
            lr_scheduler = _make_scheduler(optimizer, sv_file.get('lr_scheduler'))
            early_stopper = utils.Early_stopping(
                **sv_file.get('early_stopping', config['early_stopper'])
            )

            epoch_start = sv_file['epoch'] + 1
            state = sv_file.get('state')
            if state is not None:
                torch.set_rng_state(state)

            print(f'Resuming from epoch {epoch_start}...')
            log(f'Resuming from epoch {epoch_start}...')
    else:
        print('Training from start...')
        epoch_start = 1
        model = models.make(config['model'])
        model = _move_model_to_cuda(model)
        model_ocr = _prepare_ocr_model()
        optimizer = _make_optimizer_bundle(model, model_ocr=model_ocr)
        lr_scheduler = _make_scheduler(optimizer)
        early_stopper = utils.Early_stopping(**config['early_stopper'])

    _log_model_summary(model)
    if model_ocr is not None:
        log('model_ocr: #params={}'.format(utils.compute_num_params(model_ocr, text=True)))
        log('model_ocr: #struct={}'.format(model_ocr))
    return model, model_ocr, optimizer, epoch_start, lr_scheduler, early_stopper


def _save_checkpoint(model, model_ocr, optimizer, epoch, lr_scheduler, early_stopper, save_path, best_model):
    model_g, model_d = _get_model_for_save(model)
    model_spec = copy.deepcopy(config['model'])

    if model_d is not None:
        model_spec['sd_g'] = model_g.state_dict()
        model_spec['sd_d'] = model_d.state_dict()
    else:
        model_spec['sd'] = model_g.state_dict()

    sv_file = {
        'model': model_spec,
        'epoch': epoch,
        'state': torch.get_rng_state(),
        'early_stopping': vars(early_stopper),
        'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
    }

    if isinstance(optimizer, (tuple, list)):
        optimizer_default = config.get('optimizer')
        optimizer_g_spec = copy.deepcopy(config.get('optimizer_g', optimizer_default))
        optimizer_d_spec = copy.deepcopy(config.get('optimizer_d', optimizer_default))
        if optimizer_g_spec is None or optimizer_d_spec is None:
            raise KeyError('Tuple model checkpointing requires optimizer_g and optimizer_d')
        optimizer_g_spec['sd'] = optimizer[0].state_dict()
        optimizer_d_spec['sd'] = optimizer[1].state_dict()
        sv_file['optimizer_g'] = optimizer_g_spec
        sv_file['optimizer_d'] = optimizer_d_spec
        if len(optimizer) > 2 and config.get('optimizer_ocr') is not None:
            optimizer_ocr_spec = copy.deepcopy(config['optimizer_ocr'])
            optimizer_ocr_spec['sd'] = optimizer[2].state_dict()
            sv_file['optimizer_ocr'] = optimizer_ocr_spec
    else:
        optimizer_spec = copy.deepcopy(_get_single_optimizer_spec(config))
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file['optimizer'] = optimizer_spec

    if model_ocr is not None and config.get('optimizer_ocr') is not None:
        model_ocr_spec = copy.deepcopy(config['model_ocr'])
        model_ocr_unwrapped = _unwrap_model(model_ocr)
        if hasattr(model_ocr_unwrapped, 'get_trainable_state_dict'):
            model_ocr_spec['trainable_sd'] = model_ocr_unwrapped.get_trainable_state_dict()
        else:
            model_ocr_spec['sd'] = model_ocr_unwrapped.state_dict()
        sv_file['model_ocr'] = model_ocr_spec

    torch.save(sv_file, save_path / Path('epoch-last.pth'))
    if best_model:
        torch.save(sv_file, save_path / Path(f'best_model_Epoch_{epoch}.pth'))

    epoch_save = config.get('epoch_save')
    if epoch_save is not None and epoch % epoch_save == 0:
        torch.save(sv_file, save_path / Path(f'epoch-{epoch}.pth'))


def main(config_, save_path):
    global config, log, writer
    config = config_
    config['save_path'] = str(save_path)

    log, writer = utils.make_log_writer(save_path)
    train_loader, val_loader = make_dataloaders()
    model, model_ocr, optimizer, epoch_start, lr_scheduler, early_stopper = prepare_training()
    train = train_funcs.make(config['func_train'])
    validation = train_funcs.make(config['func_val'])
    loss_fn = losses.make(config['loss'])

    n_gpus = torch.cuda.device_count()
    print(f'Available GPUs: {n_gpus}')
    if n_gpus > 1:
        if _is_multi_model(model):
            model = [nn.parallel.DataParallel(model[0]), nn.parallel.DataParallel(model[1])]
        else:
            model = nn.parallel.DataParallel(model)

    epoch_max = config['epoch_max']
    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        print(f'epoch {epoch}/{epoch_max}')
        t_epoch_init = timer._get()
        log_info = [f'epoch {epoch}/{epoch_max}']

        if isinstance(optimizer, (tuple, list)):
            writer.add_scalar('lr_G', optimizer[0].param_groups[0]['lr'], epoch)
            writer.add_scalar('lr_D', optimizer[1].param_groups[0]['lr'], epoch)
            log_info.append(f"lr_G:{optimizer[0].param_groups[0]['lr']}")
            log_info.append(f"lr_D:{optimizer[1].param_groups[0]['lr']}")
            if len(optimizer) > 2:
                writer.add_scalar('lr_OCR', optimizer[2].param_groups[0]['lr'], epoch)
                log_info.append(f"lr_OCR:{optimizer[2].param_groups[0]['lr']}")
        else:
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            log_info.append(f"lr:{optimizer.param_groups[0]['lr']}")

        config['epoch'] = epoch
        train_out = train(train_loader, model, optimizer, loss_fn, [], config, model_ocr)
        if isinstance(train_out, tuple):
            train_loss, train_report = train_out
        else:
            train_loss, train_report = train_out, {}
        val_loss, val_report = validation(val_loader, model, loss_fn, [], config, model_ocr)

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('val_loss', val_loss, epoch)
        log_info.append(f'train: loss={train_loss:.4f}')
        log_info.append(f'val: loss={val_loss:.4f}')
        if isinstance(train_report, dict):
            for key, value in train_report.items():
                writer.add_scalar(f'train_{key}', value, epoch)
                log_info.append(f'train_{key}:{value:.4f}')
        if isinstance(val_report, dict):
            for key, value in val_report.items():
                writer.add_scalar(f'val_{key}', value, epoch)
                log_info.append(f'{key}:{value:.4f}')

        if lr_scheduler is not None:
            lr_scheduler.step(val_loss)

        t = timer._get()
        log_info.append(f"{timer.time_text(t - t_epoch_init)} / {timer.time_text(t)}")

        stop, best_model = early_stopper.early_stop(val_loss)
        log_info.append(f'Early stop {stop} / Best model {best_model}')

        _save_checkpoint(
            model, model_ocr, optimizer, epoch, lr_scheduler, early_stopper, save_path, best_model
        )

        print(', '.join(log_info))
        log(', '.join(log_info))
        writer.flush()

        if stop:
            print(f'Early stop: {stop}')
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--save', default=None)
    parser.add_argument('--tag', default=None)
    args = parser.parse_args()

    def setup_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True

    setup_seed(1996)

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    save_name = Path(args.save) if args.save is not None else Path('./save')
    if save_name is not None:
        save_name = save_name / Path(args.config.split('/')[-1][:-len('.yaml')])
    if args.tag is not None:
        save_name = Path(str(save_name) + '_' + args.tag)

    save_path = Path(save_name)
    save_path.mkdir(parents=True, exist_ok=True)
    main(config, save_path)

    if config.get('email'):
        msg = MIMEText('Your training process has completed successfully.')
        msg['Subject'] = config['email']['subject'] + '_' + config['model']['name']
        msg['From'] = config['email']['sender']
        msg['To'] = config['email']['recipient']

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(config['email']['sender'], config['email']['passwd'])
        server.sendmail(
            config['email']['sender'], config['email']['recipient'], msg.as_string()
        )
        server.quit()
