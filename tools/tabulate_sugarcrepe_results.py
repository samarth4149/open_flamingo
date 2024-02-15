import os
import sys
sys.path.append(os.path.abspath('.'))
import argparse
from tabulate import tabulate
import json
from pathlib import Path
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, default=None)
    args = parser.parse_args()
    
    model_name = args.model_name or Path(args.results_dir).name
    results = json.load(open(Path(args.results_dir) / 'results.json', 'r'))
    
    cats = ['replace', 'swap', 'add']
    cat_avgs = []
    
    for cat in cats:
        cat_avgs.append(np.mean([results[split]['top-1 acc'] for split in results if cat in split]))
    
    results_tab = tabulate([[model_name] + cat_avgs], headers=['Model'] + cats, tablefmt="github")
    print(results_tab)
    with open(Path(args.results_dir) / 'results.txt', 'w') as f:
        print(results_tab, file=f)
    