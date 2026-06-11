import argparse
import json
from collections import OrderedDict

import numpy as np

from recbole_cdr.quick_start import objective_function


CONFIG_FILES = [
    './recbole_cdr/properties/dataset/Epinions.yaml',
    './recbole_cdr/properties/experiment/no_progress.yaml',
]


def to_builtin(value):
    if isinstance(value, (dict, OrderedDict)):
        return {key: to_builtin(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(val) for val in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_model(model, seed, epochs, train_batch_size, eval_step, stopping_step, saved):
    result = objective_function(
        config_dict={
            'model': model,
            'seed': seed,
            'train_epochs': [f'BOTH:{epochs}'],
            'train_batch_size': train_batch_size,
            'eval_step': eval_step,
            'stopping_step': stopping_step,
            'compute_tgf': True,
        },
        config_file_list=CONFIG_FILES,
        saved=saved,
    )
    test_result = result['test_result']
    return to_builtin({
        'best_valid_score': result['best_valid_score'],
        'overall': test_result['overall'],
        'overlapped': test_result['overlapped'],
        'non_overlapped': test_result['non_overlapped'],
        'ugf': test_result['ugf'],
        'target_only_overall': test_result.get('target_only', {}).get('overall'),
        'transfer_gain': test_result.get('transfer_gain'),
        'tgf': test_result.get('tgf'),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['ETRCCoNet', 'ETRCDTCDR'])
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--train_batch_size', type=int, default=4096)
    parser.add_argument('--eval_step', type=int, default=5)
    parser.add_argument('--stopping_step', type=int, default=1000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[2022])
    parser.add_argument('--no_saved', action='store_true')
    parser.add_argument('--output', type=str, default='etrc_backbone_epinions_300e_bs4096_results.json')
    args = parser.parse_args()

    summary = {}
    for seed in args.seeds:
        seed_key = str(seed)
        summary[seed_key] = {}
        for model in args.models:
            summary[seed_key][model] = run_model(
                model=model,
                seed=seed,
                epochs=args.epochs,
                train_batch_size=args.train_batch_size,
                eval_step=args.eval_step,
                stopping_step=args.stopping_step,
                saved=not args.no_saved,
            )

    with open(args.output, 'w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
