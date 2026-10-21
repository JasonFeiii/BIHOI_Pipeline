"""Portable runtime settings. Credentials are read from environment only."""
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = Path(os.getenv('HOI_PIPELINE_ROOT', PROJECT_ROOT / 'pipeline')).expanduser().resolve()
MODEL_ROOT = Path(os.getenv('HOI_MODEL_ROOT', PROJECT_ROOT / '.models')).expanduser().resolve()
CALIBRATION_DIR = Path(os.getenv('HOI_CALIBRATION_DIR', PROJECT_ROOT / '.calibration')).expanduser().resolve()

def confined_path(root, *parts):
    """Resolve a child, including symlinks, without leaving its trusted root."""
    root = Path(root).expanduser().resolve()
    child = root.joinpath(*map(str, parts)).resolve()
    if not child.is_relative_to(root):
        raise ValueError('Path escapes its configured root')
    return child

def allowed_data_roots():
    values = [os.getenv('HOI_DATA_ROOT', str(PROJECT_ROOT / 'data'))]
    values += os.getenv('HOI_ALLOWED_DATA_ROOTS', '').split(os.pathsep)
    return tuple(Path(p).expanduser().resolve() for p in values if p)

def data_root_path(value):
    path = Path(value).expanduser().resolve()
    if not any(path.is_relative_to(root) for root in allowed_data_roots()):
        raise ValueError('Data root is not permitted; configure HOI_DATA_ROOT or HOI_ALLOWED_DATA_ROOTS before launch')
    return path

def export_root_path(data_root, value):
    root = data_root_path(data_root)
    allowed = [confined_path(root, 'exports')]
    allowed += [Path(p).expanduser().resolve() for p in os.getenv('HOI_ALLOWED_EXPORT_ROOTS', '').split(os.pathsep) if p]
    path = Path(value).expanduser()
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not any(path.is_relative_to(base) for base in allowed):
        raise ValueError('Export directory must be under exports/ or a configured HOI_ALLOWED_EXPORT_ROOTS directory')
    return path
def environment(name):
    key = 'HOI_ENV_' + name.upper().replace('-', '_')
    return os.getenv(key, str(Path(os.getenv('HOI_ENV_ROOT', PROJECT_ROOT / '.environments')).expanduser().resolve() / name))
def python_for(name):
    value = environment(name)
    if Path(value).is_absolute():
        return Path(value) / 'bin/python'
    # Explicit named-environment override remains supported.
    import shutil, subprocess
    conda = os.getenv('CONDA_EXE') or shutil.which('conda')
    if not conda:
        raise RuntimeError('Conda is required to resolve a named environment: ' + value)
    base = subprocess.check_output([conda, 'info', '--base'], text=True).strip()
    return Path(base) / 'envs' / value / 'bin/python'
