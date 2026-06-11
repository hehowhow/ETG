import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from recbole_cdr.quick_start.quick_start import run_recbole_cdr


def save_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_result(model_name, seed, result, runtime_sec):
    test_result = result['test_result']
    overall = test_result['overall']
    overlapped = test_result['overlapped']
    non_overlapped = test_result['non_overlapped']
    ugf = test_result['ugf']
    tgf = test_result.get('tgf', {})

    return {
        'model': model_name,
        'seed': seed,
        'best_valid_score': round(float(result['best_valid_score']), 4),
        'runtime_sec': round(runtime_sec, 2),
        'ndcg10': overall.get('ndcg@10'),
        'ndcg20': overall.get('ndcg@20'),
        'hit10': overall.get('hit@10'),
        'hit20': overall.get('hit@20'),
        'overlap_ndcg10': overlapped.get('ndcg@10'),
        'overlap_ndcg20': overlapped.get('ndcg@20'),
        'non_overlap_ndcg10': non_overlapped.get('ndcg@10'),
        'non_overlap_ndcg20': non_overlapped.get('ndcg@20'),
        'ugf_ndcg10': ugf.get('ndcg@10'),
        'ugf_ndcg20': ugf.get('ndcg@20'),
        'ugf_hit10': ugf.get('hit@10'),
        'ugf_hit20': ugf.get('hit@20'),
        'tgf_ndcg10': tgf.get('ndcg@10'),
        'tgf_ndcg20': tgf.get('ndcg@20'),
        'tgf_hit10': tgf.get('hit@10'),
        'tgf_hit20': tgf.get('hit@20'),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_config', type=str, default='./recbole_cdr/properties/dataset/Epinions.yaml')
    parser.add_argument('--output_dir', type=Path, default=Path('etrc_vug_compare_results'))
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--train_batch_size', type=int, default=4096)
    parser.add_argument('--eval_batch_size', type=int, default=409600000)
    parser.add_argument('--eval_step', type=int, default=1)
    parser.add_argument('--stopping_step', type=int, default=10)
    parser.add_argument('--valid_metric', type=str, default='NDCG@10')
    parser.add_argument('--noharm_weight', type=float, default=0.1)
    parser.add_argument('--tgf_weight', type=float, default=0.1)
    args = parser.parse_args()

    common_config = {
        'seed': args.seed,
        'gpu_id': args.gpu_id,
        'state': 'ERROR',
        'show_progress': False,
        'save_dataset': False,
        'save_dataloaders': False,
        'delete_checkpoint_after_eval': True,
        'compute_tgf': True,
        'train_epochs': [f'BOTH:{args.epochs}'],
        'train_batch_size': args.train_batch_size,
        'eval_batch_size': args.eval_batch_size,
        'eval_step': args.eval_step,
        'stopping_step': args.stopping_step,
        'valid_metric': args.valid_metric,
        'etrc_noharm_weight': args.noharm_weight,
        'etrc_tgf_weight': args.tgf_weight,
    }

    rows = []
    payload = {}
    for model_name in ['VUG', 'ETRCVUG']:
        start_time = time.time()
        result = run_recbole_cdr(
            model=model_name,
            config_file_list=[args.dataset_config],
            config_dict=common_config,
            saved=True,
        )
        runtime_sec = time.time() - start_time
        rows.append(summarize_result(model_name, args.seed, result, runtime_sec))
        payload[model_name] = result

    csv_path = args.output_dir / f'vug_etrcvug_fair_compare_seed{args.seed}.csv'
    json_path = args.output_dir / f'vug_etrcvug_fair_compare_seed{args.seed}.json'
    save_csv(csv_path, rows, fieldnames=list(rows[0].keys()))
    with open(json_path, 'w', encoding='utf-8') as fp:
        json.dump(payload, fp, indent=2)

    print(json.dumps({'csv_path': str(csv_path), 'json_path': str(json_path), 'rows': rows}, indent=2))


if __name__ == '__main__':
    main()
