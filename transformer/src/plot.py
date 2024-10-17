import os
import re
import sys
import tomllib
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from conf import Train as TrainConfig
from loguru import logger
logger.remove()
logger.add(sys.stdout)
sns.set_theme(style='whitegrid', font_scale=1.6)


@dataclass
class Metric:
    log_name: str
    label: str
    larger_is_better: bool

@dataclass
class Result:
    label: str
    metric: Metric
    last: float
    best: float

METRICS = [
    Metric('BLEU Score', 'BLEU Score', True),
    Metric('val_loss', 'Validation Loss (CE)', False),
    Metric('train_loss', 'Training Loss (CE)', False),
    Metric('Rouge1', 'Rouge-1 Score', True),
    Metric('Rouge2', 'Rouge-2 Score', True),
    Metric('RougeL', 'Rouge-L Score', True),
    Metric('RougeLsum', 'Rouge-Lsum Score', True),
    Metric('Meteor', 'METEOR Score', True),
    Metric('Bertscore F1', 'Bertscore F1', True),
    Metric('time', 'Time (s)', False)
]

LABEL_MAP = {
    'NVIDIA A40': 'A40',
    'NVIDIA A100': 'A100',
    'alternating-exp-ring': 'AER'
}


def get_label(train_cfg: TrainConfig,
              gpu_model: str,
              num_workers: int):
    if train_cfg.backend.lower() == 'pytorchddp':
        #label = f'AllReduce - {num_workers}x{gpu_model}'
        label = 'AllReduce'
    else:
        # label = f'Decent - {num_workers}x{gpu_model} - {train_cfg.decent.topology} - {train_cfg.optim.name} - {train_cfg.optim.lr}'
        label = f'AccumAdam - {train_cfg.decent.topology}'
    for key, value in LABEL_MAP.items():
        label = label.replace(key, value)
    return label


def read_test_log(exp_dir: str):
    def _load_toml(file_path: str):
        with open(file_path, 'rb') as f:
            return tomllib.load(f)
    num_workers = 0
    test_log = pd.read_csv(os.path.join(exp_dir, 'test_log.csv'), sep=',')
    test_log['BLEU Score'] *= 100
    test_log['time'] /= 1000
    with open(os.path.join(exp_dir, 'train_cfg.dump.toml'), 'r') as f:
        dumped_train_cfg = f.read()
        matches = re.findall('world_size = .*\n', dumped_train_cfg)
        num_workers = int(matches[0].split(' ')[2])
        matches = re.findall('gpu_model = ".*"\n', dumped_train_cfg)
        gpu_model = matches[0].split('"')[1]
    train_cfg = TrainConfig(**_load_toml(os.path.join(exp_dir, 'train_cfg.dump.toml')))
    label = get_label(train_cfg, gpu_model, num_workers)
    test_log['label'] = label
    
    res = []
    for metric in METRICS:
        last = test_log[metric.log_name].to_list()[-1]
        best = test_log[metric.log_name].max() if metric.larger_is_better else test_log[metric.log_name].min()
        res.append(Result(label, metric, last, best))

    return test_log, res

def plot(logs: pd.DataFrame, x: str, y: str, img_dir: str, xscale: str='linear', yscale: str='linear'):
    plt.clf()
    plt.figure(figsize=(6, 6))
    sns.lineplot(data=logs, x=x, y=y, style='label', hue='label', errorbar=('sd', 2))
    if y == 'BLEU Score':
        # plot baseline
        plt.axhline(y=27.3, color='r', linestyle='--', label='baseline')
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.xlim(left=0)
    # if y == 'BLEU Score':
    #     plt.ylim(bottom=0.2)

    if x == 'time':
        plt.xlabel('Time (1000 s)')
    elif x == 'step':
        plt.xlabel('Step')
    
    if y == 'BLEU Score':
        plt.ylabel('BLEU Score')
    elif y == 'val_loss':
        plt.ylabel('Validation Loss (Cross Entropy)')
        ax = plt.gca()
        ax.text((10 + 25) // 2, 3.07, '~60% Speedup', horizontalalignment='center', verticalalignment='center', fontdict={'fontsize': 15, 'color': 'black'})
        ax.annotate('',
            xy=(10, 3.045),
            xytext=(25, 3.045),
            xycoords='data',
            arrowprops=dict(color='black', arrowstyle="<->", lw=1))

    plt.tight_layout()
    plt.savefig(img_dir, bbox_inches='tight')


def exclude_empty_dirs(exp_dirs: list):
    return [exp_dir for exp_dir in exp_dirs if os.path.exists(os.path.join(exp_dir, 'test_log.csv'))]


def exclude_label(test_log: pd.DataFrame):
    # if 'DDP' in test_log['label'].unique()[0]:
    #     return False
    # if ('ring' in test_log['label'].unique()[0]) or ('AER' in test_log['label'].unique()[0]):
    #     return True
    return False


def main():
    exp_dirs = sys.argv[1:]
    exp_dirs = exclude_empty_dirs(exp_dirs)

    time_values: Dict[str, List[float]] = {}
    metric_values: Dict[str, List[Result]] = {}
    for exp_dir in exp_dirs:
        test_log, _metric_values = read_test_log(exp_dir)
        if exclude_label(test_log): continue
        for label in test_log['label'].unique():
            if label not in time_values:
                time_values[label] = []
                metric_values[label] = []
            time_values[label].extend(test_log[test_log['label'] == label]['time'].unique().tolist())
            metric_values[label].extend(_metric_values)
    for label in time_values:
        time_values[label] = sorted(list(set(time_values[label])))
    
    interp_logs = pd.concat(filter(lambda x: not exclude_label(x), [read_test_log(exp_dir)[0] for exp_dir in exp_dirs]))

    for label in interp_logs['label'].unique():
        for i in interp_logs['epoch'].unique():
            mean_time = interp_logs[(interp_logs['epoch'] == i) & (interp_logs['label'] == label)]['time'].mean()
            interp_logs.loc[(interp_logs['epoch'] == i) & (interp_logs['label'] == label), 'time'] = mean_time

    # interp_log = pd.concat(interp_logs)
    # print(interp_log)
    test_log = pd.concat(filter(lambda x: not exclude_label(x), [read_test_log(exp_dir)[0] for exp_dir in exp_dirs]))

    print(interp_logs.loc[interp_logs['epoch'] == 24][['label', 'BLEU Score', 'Meteor']])

    os.makedirs('image', exist_ok=True)
    plot(interp_logs, 'time', 'BLEU Score', 'image/time_vs_bleu.svg')
    plot(test_log, 'step', 'BLEU Score', 'image/step_vs_bleu.svg')
    plot(interp_logs, 'time', 'val_loss', 'image/time_vs_valloss.svg')
    plot(test_log, 'step', 'val_loss', 'image/step_vs_valloss.svg')
    plot(interp_logs, 'time', 'Meteor', 'image/time_vs_meteor.svg')

    logger.info(f'Plots generated at image/ directory')

    for label in sorted(metric_values.keys()):
        # log the last bleu score with 2x standard deviation
        # _bleu_score = np.array(last_bleu_values[label])
        # _train_time = np.array(total_train_time_values[label]) / 60 / 60
        # logger.info(f'{label:40s} - last BLEU score     : {np.mean(_bleu_score):.4f} ± {2 * np.std(_bleu_score):.4f}')
        # logger.info(f'{"":40s} - last validation loss: {np.mean(last_val_loss_values[label]):.4f} ± {2 * np.std(last_val_loss_values[label]):.4f}')
        # logger.info(f'{"":40s} - last training loss  : {np.mean(last_train_loss_values[label]):.4f} ± {2 * np.std(last_train_loss_values[label]):.4f}')
        # logger.info(f'{"":40s} - total training time : {np.mean(_train_time):.4f} ± {2 * np.std(_train_time):.4f} hours')

        logger.info(f'----- {label:40s} -----')

        for metric in METRICS:
            _metric_values = [r.last for r in metric_values[label] if r.metric == metric]
            _metric_best_values = [r.best for r in metric_values[label] if r.metric == metric]
            logger.info(f'{metric.label:40s} - last: {np.mean(_metric_values):.4f} ± {2 * np.std(_metric_values):.4f}')
            logger.info(f'{"":40s} - best: {np.mean(_metric_best_values):.4f} ± {2 * np.std(_metric_best_values):.4f}')


if  __name__ == '__main__':
    main()
