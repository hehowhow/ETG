import argparse
import csv
import json
import sys
import time
from pathlib import Path

from recbole.utils import init_logger, init_seed

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from recbole_cdr.config import CDRConfig
from recbole_cdr.data import create_dataset, data_preparation
from recbole_cdr.utils import get_model, get_trainer


def save_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_result(model_name, test_result, runtime_sec, epochs):
    overall = test_result['overall']
    overlapped = test_result['overlapped']
    non_overlapped = test_result['non_overlapped']
    ugf = test_result['ugf']
    tgf = test_result.get('tgf', {})
    return {
        'model': model_name,
        'epochs': epochs,
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


def run_one_model(model_name, dataset_config, config_dict):
    config = CDRConfig(
        model=model_name,
        config_file_list=[dataset_config],
        config_dict=config_dict,
    )
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    if test_data.overlap_users is None:
        test_data.overlap_users = train_data.overlap_users

    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    trainer.fit(
        train_data,
        valid_data=valid_data,
        verbose=True,
        saved=False,
        show_progress=config['show_progress'],
    )
    return trainer.evaluate(test_data, load_best_model=False, show_progress=config['show_progress'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_config', type=str, default='./recbole_cdr/properties/dataset/Epinions.yaml')
    parser.add_argument('--output_dir', type=Path, default=Path('etrc_vug_crossfit_results'))
    parser.add_argument('--pair_table', type=str, default='etrc_vug_crossfit_results/vug_pair_crossfit_seed2022.csv')
    parser.add_argument('--user_table', type=str, default='etrc_vug_crossfit_results/vug_user_crossfit_seed2022.csv')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--train_batch_size', type=int, default=4096)
    parser.add_argument('--eval_batch_size', type=int, default=4096)
    parser.add_argument('--noharm_weight', type=float, default=0.1)
    parser.add_argument('--tgf_weight', type=float, default=0.1)
    args = parser.parse_args()

    common_config = {
        'seed': args.seed,
        'gpu_id': args.gpu_id,
        'show_progress': False,
        'save_dataset': False,
        'save_dataloaders': False,
        'delete_checkpoint_after_eval': True,
        'train_batch_size': args.train_batch_size,
        'eval_batch_size': args.eval_batch_size,
        'train_epochs': [f'BOTH:{args.epochs}'],
        'eval_step': 0,
        'compute_tgf': True,
        'etrc_noharm_weight': args.noharm_weight,
        'etrc_tgf_weight': args.tgf_weight,
        'state': 'ERROR',
    }

    rows = []
    payload = {}
    for model_name, delta_path in [
        ('ETRCVUGPairCF', args.pair_table),
        ('ETRCVUGUserCF', args.user_table),
    ]:
        start_time = time.time()
        test_result = run_one_model(
            model_name,
            args.dataset_config,
            {**common_config, 'etrc_crossfit_delta_path': delta_path},
        )
        runtime_sec = time.time() - start_time
        rows.append(summarize_result(model_name, test_result, runtime_sec, args.epochs))
        payload[model_name] = test_result

    csv_path = args.output_dir / f'etrc_vug_crossfit_fixed_epoch_{args.epochs}_seed{args.seed}.csv'
    json_path = args.output_dir / f'etrc_vug_crossfit_fixed_epoch_{args.epochs}_seed{args.seed}.json'
    save_csv(csv_path, rows, fieldnames=list(rows[0].keys()))
    with open(json_path, 'w', encoding='utf-8') as fp:
        json.dump(payload, fp, indent=2)

    print(json.dumps({'csv_path': str(csv_path), 'json_path': str(json_path), 'rows': rows}, indent=2))


if __name__ == '__main__':
    main()
