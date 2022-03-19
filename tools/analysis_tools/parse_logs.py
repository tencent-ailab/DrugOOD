import argparse
import logging
import os.path as osp
import pathlib
from glob import glob

import numpy as np
from prettytable import PrettyTable


def read_log(log_dir, best_metric="val:accuracy"):
    # find out the logs
    logs = glob(f"{log_dir}/*.log")
    logs.sort(key=osp.getmtime)
    target_log = logs[-1]

    with open(target_log) as f:
        f = f.readlines()
    f.reverse()
    for idx, l in enumerate(f):
        if f'Best {best_metric}' in l:
            return l, f[idx - 1]


def log_metrics(info):
    group_table = PrettyTable()
    group_table.field_names = ["subset", *info[next(iter(info))].keys()]

    if info.get("seed") is None:
        group_table.title = 'exp result'
    else:
        group_table.title = f'seed {info.get("seed")}'
        info.pop("seed")

    for subset, metrics in info.items():
        metric_values = [float(v) for k, v, in metrics.items()]
        group_table.add_row([subset, *metric_values])

    print(group_table.get_string())


def parse_args():
    parser = argparse.ArgumentParser(description='None')
    parser.add_argument('--work_dir', help='work dir of exp',
                        default="/apdcephfs/share_1364275/yuanfengji/project/ood/work_dirs/erm/20210926_camelyon17_erm")
    parser.add_argument('--best_metric', default="val:accuracy")
    parser.add_argument("--ignore_metrics", help="metrics need to ignore")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.best_metric, (
        'Please specify at best metric the argument "--best_metric".')

    folder_list = glob(f"{args.work_dir}/[0-9]/")
    print(f"Exp dir : {args.work_dir}")

    if not folder_list:
        folder_list.append(args.work_dir)
    folder_list.sort()

    infos = []
    metrics = dict()
    seeds = []

    for folder in folder_list:
        _, others = read_log(folder, args.best_metric)
        others = others.split(',')
        others.__delitem__(0)

        info = {}
        for item in others:
            if "\t" in item:
                item = item.split("\t")[1]
            if '\n' in item:
                item = item.split("\n")[0]
            d, m, v = item.replace(" ", "").split(":")

            if d not in info.keys():
                info[d] = {m: v}
            else:
                info[d].update({m: v})

        path = pathlib.PurePath(folder).name

        try:
            info["seed"] = int(path)
            seeds.append(int(path))
        except ValueError as e:
            logging.exception(e)

        log_metrics(info)
        infos.append(info)

    print(f"Total {len(infos)} seed: {seeds}")

    for info in infos:
        for d, ms in info.items():
            for m, v in ms.items():
                if f'{d}:{m}' not in metrics.keys():
                    metrics[f'{d}:{m}'] = [float(v)]
                else:
                    metrics[f'{d}:{m}'].append(float(v))

    for metric, metric_value in metrics.items():
        print(f'{metric} mean: {np.asarray(metric_value).mean():.2f} '
              f'std: {np.asarray(metric_value).std():.2f} '
              f'var: {np.asarray(metric_value).var():.2f}')


if __name__ == '__main__':
    main()
