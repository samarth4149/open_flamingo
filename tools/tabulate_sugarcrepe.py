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
    parser.add_argument('--output-csv', action='store_true', default=False)
    args = parser.parse_args()
    
    if ',' in args.results_dir:
        results_dirs = args.results_dir.split(',')
        results_tab = []
        for results_dir in results_dirs:
            model_name = Path(results_dir).name
            results = json.load(open(Path(results_dir) / 'results.json', 'r'))
            
            cats = ['replace', 'swap', 'add']
            cat_avgs = []
            
            for cat in cats:
                cat_avgs.append(np.mean([results[split]['top-1 acc'] for split in results if cat in split]))
            
            results_tab.append([model_name] + cat_avgs)
        
        if args.output_csv:
            results_tab = ','.join(['Model'] + cats) + '\n' + '\n'.join([','.join(map(str, row)) for row in results_tab])
        else:
            results_tab = tabulate(results_tab, headers=['Model'] + cats, tablefmt="github")
        print(results_tab)
    else:
        model_name = args.model_name or Path(args.results_dir).name
        results = json.load(open(Path(args.results_dir) / 'results.json', 'r'))
        
        cats = ['replace', 'swap', 'add']
        cat_avgs = []
        
        for cat in cats:
            cat_avgs.append(np.mean([results[split]['top-1 acc'] for split in results if cat in split]))
        
        if args.output_csv:
            results_tab = ','.join(['Model'] + cats) + '\n' + ','.join(map(str, [model_name] + cat_avgs))
        else:
            results_tab = tabulate([[model_name] + cat_avgs], headers=['Model'] + cats, tablefmt="github")
        
        print(results_tab)
        if args.output_csv:
            with open(Path(args.results_dir) / 'results.csv', 'w') as f:
                print(results_tab, file=f)
        else:
            with open(Path(args.results_dir) / 'results.txt', 'w') as f:
                print(results_tab, file=f)
    