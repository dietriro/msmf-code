from pathlib import Path
import sys

try:
    import msmfcode
except ModuleNotFoundError:
    path_root = Path(__file__).parent.resolve().parent
    if path_root not in sys.path:
        sys.path.append(str(path_root))
    else:
        print(f'Can\'t import module msmfcode but its module path seems to be in the PYTHONPATH already:')
        print(path_root)
