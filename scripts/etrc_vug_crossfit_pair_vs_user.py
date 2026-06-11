import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from recbole.data.utils import create_samplers
from recbole.utils import init_logger, init_seed

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from recbole_cdr.config import CDRConfig
from recbole_cdr.data import create_dataset, data_preparation
from recbole_cdr.data.dataloader import CrossDomainDataloader
from recbole_cdr.quick_start.quick_start import run_recbole_cdr
from recbole_cdr.utils import get_model, get_trainer


def pair_logits(user_embeddings, item_embeddings, user_ids, item_ids):
    return torch.mul(user_embeddings[user_ids], item_embeddings[item_ids]).sum(dim=1)


def build_user_grouped_two_fold_indices(target_dataset, seed):
    inter_feat = target_dataset.inter_feat
    user_field = target_dataset.uid_field
    user_ids = inter_feat[user_field].cpu().numpy()

    rng = np.random.default_rng(seed)
    grouped = defaultdict(list)
    for idx, user_id in enumerate(user_ids.tolist()):
        grouped[int(user_id)].append(idx)

    fold_indices = [[], []]
    for user_id in sorted(grouped.keys()):
        indices = np.array(grouped[user_id], dtype=np.int64)
        rng.shuffle(indices)
        if len(indices) == 1:
            fold_indices[int(rng.integers(0, 2))].append(int(indices[0]))
            continue

        split = len(indices) // 2
        split = max(split, 1)
        fold_indices[0].extend(indices[:split].tolist())
        fold_indices[1].extend(indices[split:].tolist())

    fold_indices = [sorted(indices) for indices in fold_indices]
    return fold_indices


def subset_dataset(dataset, indices):
    sub_inter = dataset.inter_feat[indices]
    return dataset.copy(sub_inter)


def make_train_loader(config, dataset, source_dataset, source_sampler, target_dataset):
    target_train_sampler, _, _ = create_samplers(
        config,
        dataset.target_domain_dataset,
        [target_dataset, target_dataset, target_dataset],
    )
    return CrossDomainDataloader(
        config,
        dataset,
        source_dataset,
        source_sampler,
        target_dataset,
        target_train_sampler,
        shuffle=True,
    )


def train_reference_vug(config, dataset, source_dataset, source_sampler, target_dataset):
    train_data = make_train_loader(config, dataset, source_dataset, source_sampler, target_dataset)
    model = get_model(config['model'])(config, dataset).to(config['device'])
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    trainer.fit(train_data, valid_data=train_data, verbose=True, saved=False, show_progress=config['show_progress'])
    model.eval()
    model.set_phase('BOTH')
    return model


@torch.no_grad()
def collect_pair_rows(model, heldout_dataset, fold_id, batch_size, user_counts):
    _, _, target_user_all_embeddings, target_item_all_embeddings = model.forward()
    target_only_user_embeddings, target_only_item_embeddings = model.target_only_embeddings()

    inter_feat = heldout_dataset.inter_feat
    user_tensor = inter_feat[heldout_dataset.uid_field].to(model.device)
    item_tensor = inter_feat[heldout_dataset.iid_field].to(model.device)
    label_tensor = torch.ones_like(user_tensor, dtype=torch.float32, device=model.device)

    rows = []
    for start in range(0, len(user_tensor), batch_size):
        end = min(start + batch_size, len(user_tensor))
        batch_user = user_tensor[start:end]
        batch_item = item_tensor[start:end]
        batch_label = label_tensor[start:end]

        transfer_logits = pair_logits(
            target_user_all_embeddings,
            target_item_all_embeddings,
            batch_user,
            batch_item,
        )
        target_only_logits = pair_logits(
            target_only_user_embeddings,
            target_only_item_embeddings,
            batch_user,
            batch_item,
        )
        pair_delta = (
            F.binary_cross_entropy_with_logits(target_only_logits, batch_label, reduction='none')
            - F.binary_cross_entropy_with_logits(transfer_logits, batch_label, reduction='none')
        )
        transfer_prob = torch.sigmoid(transfer_logits)
        target_only_prob = torch.sigmoid(target_only_logits)

        for user_id, item_id, delta, cdr_prob, tgt_prob in zip(
            batch_user.detach().cpu().tolist(),
            batch_item.detach().cpu().tolist(),
            pair_delta.detach().cpu().tolist(),
            transfer_prob.detach().cpu().tolist(),
            target_only_prob.detach().cpu().tolist(),
        ):
            rows.append(
                {
                    'target_user_id': int(user_id),
                    'target_item_id': int(item_id),
                    'pair_delta': float(delta),
                    'fold_id': int(fold_id),
                    'user_interactions': int(user_counts[int(user_id)]),
                    'transfer_prob': float(cdr_prob),
                    'target_only_prob': float(tgt_prob),
                }
            )
    return rows


def save_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_user_rows(pair_rows, tau):
    grouped = defaultdict(list)
    user_counts = {}
    for row in pair_rows:
        user_id = int(row['target_user_id'])
        grouped[user_id].append(float(row['pair_delta']))
        user_counts[user_id] = int(row['user_interactions'])

    user_rows = []
    for user_id in sorted(grouped.keys()):
        delta_values = grouped[user_id]
        user_delta = float(sum(delta_values) / max(len(delta_values), 1))
        if user_delta > tau:
            label = 'helpful'
        elif user_delta < -tau:
            label = 'harmful'
        else:
            label = 'neutral'
        user_rows.append(
            {
                'target_user_id': int(user_id),
                'user_delta': user_delta,
                'user_interactions': int(user_counts[user_id]),
                'num_pairs': int(len(delta_values)),
                'user_label': label,
            }
        )
    return user_rows


def summarize_result(model_name, result, runtime_sec):
    test_result = result['test_result']
    overall = test_result['overall']
    overlapped = test_result['overlapped']
    non_overlapped = test_result['non_overlapped']
    ugf = test_result['ugf']
    tgf = test_result.get('tgf', {})

    return {
        'model': model_name,
        'runtime_sec': round(runtime_sec, 2),
        'best_valid_score': round(float(result['best_valid_score']), 4),
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


def make_reference_config(dataset_config, seed, batch_size, epochs, gpu_id):
    config_dict = {
        'seed': seed,
        'gpu_id': gpu_id,
        'show_progress': False,
        'delete_checkpoint_after_eval': True,
        'save_dataset': False,
        'save_dataloaders': False,
        'compute_tgf': False,
        'eval_step': 0,
        'train_batch_size': batch_size,
        'eval_batch_size': batch_size,
        'train_epochs': [f'BOTH:{epochs}'],
    }
    return CDRConfig(
        model='VUG',
        config_file_list=[dataset_config],
        config_dict=config_dict,
    )


def build_crossfit_tables(args):
    ref_config = make_reference_config(
        args.dataset_config,
        args.seed,
        args.train_batch_size,
        args.epochs,
        args.gpu_id,
    )
    init_seed(ref_config['seed'], ref_config['reproducibility'])
    init_logger(ref_config)

    dataset = create_dataset(ref_config)
    train_data, _, _ = data_preparation(ref_config, dataset)
    source_dataset = train_data.source_dataset
    source_sampler = train_data.source_dataloader.sampler
    target_train_dataset = train_data.target_dataset

    user_counts = defaultdict(int)
    for user_id in target_train_dataset.inter_feat[target_train_dataset.uid_field].cpu().tolist():
        user_counts[int(user_id)] += 1

    fold_indices = build_user_grouped_two_fold_indices(target_train_dataset, args.seed)

    pair_rows = []
    fold_meta = []
    for fold_id in range(2):
        heldout_indices = fold_indices[fold_id]
        train_indices = fold_indices[1 - fold_id]

        fold_train_dataset = subset_dataset(target_train_dataset, train_indices)
        fold_holdout_dataset = subset_dataset(target_train_dataset, heldout_indices)

        start_time = time.time()
        model = train_reference_vug(
            ref_config,
            dataset,
            source_dataset,
            source_sampler,
            fold_train_dataset,
        )
        runtime_sec = time.time() - start_time
        fold_rows = collect_pair_rows(
            model,
            fold_holdout_dataset,
            fold_id=fold_id,
            batch_size=args.eval_batch_size,
            user_counts=user_counts,
        )
        pair_rows.extend(fold_rows)
        fold_meta.append(
            {
                'fold_id': fold_id,
                'train_size': len(train_indices),
                'heldout_size': len(heldout_indices),
                'runtime_sec': round(runtime_sec, 2),
            }
        )

    pair_table_path = args.output_dir / f'vug_pair_crossfit_seed{args.seed}.csv'
    save_csv(
        pair_table_path,
        pair_rows,
        fieldnames=[
            'target_user_id',
            'target_item_id',
            'pair_delta',
            'fold_id',
            'user_interactions',
            'transfer_prob',
            'target_only_prob',
        ],
    )

    user_rows = aggregate_user_rows(pair_rows, tau=args.etrc_delta_tau)
    user_table_path = args.output_dir / f'vug_user_crossfit_seed{args.seed}.csv'
    save_csv(
        user_table_path,
        user_rows,
        fieldnames=[
            'target_user_id',
            'user_delta',
            'user_interactions',
            'num_pairs',
            'user_label',
        ],
    )

    metadata = {
        'seed': args.seed,
        'pair_table_path': str(pair_table_path),
        'user_table_path': str(user_table_path),
        'num_pair_rows': len(pair_rows),
        'num_user_rows': len(user_rows),
        'folds': fold_meta,
    }
    metadata_path = args.output_dir / f'vug_crossfit_metadata_seed{args.seed}.json'
    with open(metadata_path, 'w', encoding='utf-8') as fp:
        json.dump(metadata, fp, indent=2)

    return pair_table_path, user_table_path, metadata_path


def run_final_models(args, pair_table_path, user_table_path):
    common_config = {
        'seed': args.seed,
        'gpu_id': args.gpu_id,
        'show_progress': False,
        'delete_checkpoint_after_eval': True,
        'save_dataset': False,
        'save_dataloaders': False,
        'compute_tgf': True,
        'train_batch_size': args.train_batch_size,
        'eval_batch_size': args.eval_batch_size,
        'train_epochs': [f'BOTH:{args.epochs}'],
        'etrc_noharm_weight': args.noharm_weight,
        'etrc_tgf_weight': args.tgf_weight,
        'etrc_delta_tau': args.etrc_delta_tau,
        'etrc_crossfit_missing_strategy': 'skip',
        'etrc_crossfit_supervise_positive_only': True,
    }

    experiment_rows = []
    experiment_json = {}
    model_specs = [
        ('ETRCVUGPairCF', str(pair_table_path)),
        ('ETRCVUGUserCF', str(user_table_path)),
    ]

    for model_name, delta_path in model_specs:
        start_time = time.time()
        result = run_recbole_cdr(
            model=model_name,
            config_file_list=[args.dataset_config],
            config_dict={**common_config, 'etrc_crossfit_delta_path': delta_path},
            saved=True,
        )
        runtime_sec = time.time() - start_time

        experiment_rows.append(summarize_result(model_name, result, runtime_sec))
        experiment_json[model_name] = result

    summary_csv_path = args.output_dir / f'etrc_vug_crossfit_pair_vs_user_seed{args.seed}.csv'
    save_csv(
        summary_csv_path,
        experiment_rows,
        fieldnames=list(experiment_rows[0].keys()),
    )
    summary_json_path = args.output_dir / f'etrc_vug_crossfit_pair_vs_user_seed{args.seed}.json'
    with open(summary_json_path, 'w', encoding='utf-8') as fp:
        json.dump(experiment_json, fp, indent=2)

    return summary_csv_path, summary_json_path, experiment_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_config', type=str, default='./recbole_cdr/properties/dataset/Epinions.yaml')
    parser.add_argument('--output_dir', type=Path, default=Path('etrc_vug_crossfit_results'))
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--train_batch_size', type=int, default=4096)
    parser.add_argument('--eval_batch_size', type=int, default=4096)
    parser.add_argument('--noharm_weight', type=float, default=0.1)
    parser.add_argument('--tgf_weight', type=float, default=0.1)
    parser.add_argument('--etrc_delta_tau', type=float, default=0.001)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pair_table_path, user_table_path, metadata_path = build_crossfit_tables(args)
    summary_csv_path, summary_json_path, experiment_rows = run_final_models(args, pair_table_path, user_table_path)

    payload = {
        'pair_table_path': str(pair_table_path),
        'user_table_path': str(user_table_path),
        'metadata_path': str(metadata_path),
        'summary_csv_path': str(summary_csv_path),
        'summary_json_path': str(summary_json_path),
        'rows': experiment_rows,
    }
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
