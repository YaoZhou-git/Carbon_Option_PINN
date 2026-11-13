"""Explicit stage launcher. No stage runs unless --stage is supplied."""
from pathlib import Path
import argparse
import importlib
import os
import sys

def main():
    parser=argparse.ArgumentParser(description='Reproduce current V2 empirical stages using locally licensed inputs.')
    parser.add_argument('--stage', choices=['estimate','hac','price','fig7','fig8','external-audit'])
    parser.add_argument('--data-dir', help='Directory of locally licensed source workbooks; excluded from GitHub')
    parser.add_argument('--dec22-file', help='Explicit licensed Dec22 xlsx path')
    parser.add_argument('--dec23-file', help='Explicit licensed Dec23 xlsx path')
    parser.add_argument('--output-dir', help='Local generated-results directory')
    parser.add_argument('--all-historical-scenarios', action='store_true', help='Explicitly enable original legacy100/alpha-profile/additional rate-sensitivity computations; not needed for current paper')
    parser.add_argument('--language',choices=['en','zh','both'],default='en')
    parser.add_argument('--font',help='Optional installed plotting font')
    args=parser.parse_args()
    if args.stage is None:
        parser.print_help()
        return 0
    for value,key in [(args.data_dir,'CARBON_OPTION_DATA_DIR'),(args.dec22_file,'CARBON_OPTION_DEC22_FILE'),(args.dec23_file,'CARBON_OPTION_DEC23_FILE'),(args.output_dir,'CARBON_OPTION_RESULTS_DIR'),(args.font,'CARBON_OPTION_FONT')]:
        if value:os.environ[key]=value
    os.environ['CARBON_OPTION_FULL_RUN']='1' if args.all_historical_scenarios else '0'
    os.environ['CARBON_OPTION_LANGUAGE']=args.language
    sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
    module={'estimate':'run_pilot','hac':'robust_inference','price':'calibrated_pricing','fig7':'plot_price_history','fig8':'plot_indexed_results','external-audit':'external_kappa_audit'}[args.stage]
    try:
        importlib.import_module(module).main()
    except (FileNotFoundError,ValueError) as exc:
        parser.exit(2,str(exc)+'\n')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
