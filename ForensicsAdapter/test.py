"""
eval pretained model.
"""
import numpy as np
import random
import yaml
from pathlib import Path
from tqdm import tqdm
from trainer.metrics.utils import get_test_metrics
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from model.ds import DS, moe_route_metrics

import argparse

parser = argparse.ArgumentParser(description='Evaluate a Forensics Adapter checkpoint.')
parser.add_argument('--detector_path', type=str,
                    default=str(Path(__file__).with_name('config') / 'test.yaml'))
parser.add_argument('--test_dataset', nargs='+')
parser.add_argument('--weights_path', type=str)

def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = test_name  # specify the current test dataset
        test_set = DeepfakeAbstractBaseDataset(
                config=config,
                mode='test', 
            )
        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set, 
                batch_size=config['test_batchSize'],
                shuffle=False, 
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
                drop_last=False
            )
        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def test_one_dataset(model, data_loader):
    device = next(model.parameters()).device
    predictions, labels, reference_rates = [], [], []
    route_weights = []
    for data_dict in tqdm(data_loader, total=len(data_loader)):
        data_dict['label'] = (data_dict['label'] != 0).long()
        data_dict = {
            name: tensor.to(device) if tensor is not None else None
            for name, tensor in data_dict.items()
        }
        result = inference(model, data_dict)
        labels.extend(data_dict['label'].cpu().tolist())
        predictions.extend(result['prob'].cpu().tolist())
        if 'frequency_reference_used' in result:
            reference_rates.extend(result['frequency_reference_used'].float().mean(dim=1).cpu().tolist())
        if 'moe_route_weights' in result:
            route_weights.append(result['moe_route_weights'].cpu())

    route_metrics = moe_route_metrics(torch.cat(route_weights)) if route_weights else {}
    return np.asarray(predictions), np.asarray(labels), np.asarray(reference_rates), route_metrics
    
def test_epoch(model, test_data_loaders):
    # set model to eval mode
    model.eval()

    # define test recorder
    metrics_all_datasets = {}

    # testing for all test data
    keys = test_data_loaders.keys()
    for key in keys:
        data_dict = test_data_loaders[key].dataset.data_dict
        # compute loss for each dataset
        predictions_nps, label_nps, reference_rates, route_metrics = test_one_dataset(model, test_data_loaders[key])
        print(f'name {data_dict.keys()}')
        # compute metric for each dataset
        metric_one_dataset = get_test_metrics(y_pred=predictions_nps, y_true=label_nps,
                                              img_names=data_dict['image'])
        metric_one_dataset.update(route_metrics)
        if reference_rates.size:
            metric_one_dataset['frequency_reference_rate'] = float(reference_rates.mean())
            for label, name in ((0, 'real'), (1, 'fake')):
                rates = reference_rates[label_nps == label]
                if rates.size:
                    metric_one_dataset[f'frequency_reference_rate_{name}'] = float(rates.mean())
        metrics_all_datasets[key] = metric_one_dataset
        
        # info for each dataset
        tqdm.write(f"dataset: {key}")
        for k, v in metric_one_dataset.items():
            tqdm.write(f"{k}: {v}")

    return metrics_all_datasets

@torch.no_grad()
def inference(model, data_dict):
    predictions = model(data_dict, inference=True)
    return predictions


def main():
    args = parser.parse_args()
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)

    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    weights_path = args.weights_path or config.get('weights_path')
    if not weights_path:
        parser.error('provide a checkpoint with --weights_path or weights_path in the configuration')

    device = torch.device(config['device'])
    config['cuda'] = device.type == 'cuda'
    init_seed(config)
    if config['cudnn']:
        cudnn.benchmark = True

    test_data_loaders = prepare_testing_data(config)
    frequency_mode = config.get('frequency_fusion') or 'off'
    model = DS(clip_name=config['clip_model_name'],
               adapter_vit_name=config['vit_name'],
               num_quires=config['num_quires'],
               fusion_map=config['fusion_map'],
               mlp_dim=config['mlp_dim'],
               mlp_out_dim=config['mlp_out_dim'],
               head_num=config['head_num'],
               device=device,
               frequency_fusion=frequency_mode,
               clip_download_root=config.get('clip_download_root'),
               conv_moe_mode=config.get('conv_moe_mode') or 'off',
               conv_moe_top_k=config.get('conv_moe_top_k', 1)).to(device)

    ckpt = torch.load(weights_path, map_location=device)
    if ckpt and all(key.startswith('module.') for key in ckpt):
        consume_prefix_in_state_dict_if_present(ckpt, 'module.')
    incompatible = model.load_state_dict(ckpt, strict=frequency_mode != 'off')
    if any(key.startswith('adapter.conv_moe.') for key in incompatible.missing_keys):
        raise RuntimeError('Checkpoint lacks weights for the configured convolutional MoE.')
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(f'Checkpoint key mismatch: {incompatible}')
    print(f'===> Load {weights_path} done!')
    test_epoch(model, test_data_loaders)
    print('===> Test Done!')


if __name__ == '__main__':
    main()
