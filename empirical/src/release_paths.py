"""Local input/output configuration for the empirical release copy."""
from pathlib import Path
import os
PACKAGE_ROOT = Path(__file__).resolve().parent.parent

def results_dir():
    value = os.environ.get('CARBON_OPTION_RESULTS_DIR')
    return Path(value).expanduser().resolve() if value else PACKAGE_ROOT / 'results'

def full_run():
    return os.environ.get('CARBON_OPTION_FULL_RUN', '0') == '1'

def contract_path(year):
    short = str(year)[-2:]
    override = os.environ.get('CARBON_OPTION_DEC' + short + '_FILE')
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError('Licensed contract file does not exist: ' + str(path))
        return path
    folder = os.environ.get('CARBON_OPTION_DATA_DIR')
    if not folder:
        raise FileNotFoundError('Supply --data-dir or CARBON_OPTION_DATA_DIR for locally licensed Dec22/Dec23 workbooks. Raw data are excluded from this code release.')
    matches = sorted(Path(folder).expanduser().glob('EUA FUTURE DEC ' + short + '*.xlsx'))
    if len(matches) != 1:
        raise FileNotFoundError('Expected exactly one EUA FUTURE DEC ' + short + '*.xlsx in the licensed input directory; use --dec' + short + '-file for an explicit file.')
    return matches[0].resolve()

def figure_language():
    value = os.environ.get('CARBON_OPTION_LANGUAGE', 'en')
    if value not in ('en', 'zh', 'both'):
        raise ValueError('CARBON_OPTION_LANGUAGE must be en, zh or both')
    return value

def figure_font(zh=False):
    return os.environ.get('CARBON_OPTION_FONT') or ('Microsoft YaHei' if zh else 'DejaVu Sans')
