import argparse
import json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--mouse_h5ad', required=True)
    p.add_argument('--perturbations', nargs='+', default=['ARID1A', 'PDCD1'])
    p.add_argument('--atac_key', default='atac_feat')
    p.add_argument('--out_json', default='cross_species_infer_metrics.json')
    args = p.parse_args()
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump({'status': 'template script; fill with dataset-specific evaluation flow'}, f, indent=2)


if __name__ == '__main__':
    main()
