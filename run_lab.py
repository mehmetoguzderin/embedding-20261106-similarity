#!/usr/bin/env python3
"""Set up and run the lab using isolated uv-managed Python environments.

Run ``python run_lab.py`` from any working directory.  ``--offline-check``
replays a completed cache in a temporary directory without modifying the live
run.  uv and Python 3.12 are required; uv can provision Python if necessary.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import traceback

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / 'EmbeddingGemma2_Edit_Fidelity_Lab.ipynb'
PREVIEW = ROOT / 'EmbeddingGemma2_Edit_Fidelity_Preview.html'
OUTPUTS = ROOT / 'edit_fidelity_outputs'
UV_VERSION = '0.13.7'


def python_version() -> str:
    path = ROOT / '.python-version'
    try:
        value = path.read_text(encoding='utf-8').strip()
    except OSError as exc:
        raise RuntimeError(f'Cannot read Python version from {path}: {exc}') from exc
    if not re.fullmatch(r'3\.\d+(?:\.\d+)?', value):
        raise RuntimeError(f'{path} must contain one Python 3 major.minor or major.minor.patch version')
    return value


def notebook_requirements() -> list[str]:
    """Read the notebook's literal dependency declarations without executing it."""
    try:
        notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
        cells = notebook['cells']
        if not isinstance(cells, list):
            raise ValueError('cells is not a list')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f'Cannot read notebook dependency declarations from {NOTEBOOK}: {exc}') from exc

    definitions = {'core': [], 'NOTEBOOK_REQUIREMENTS': []}
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            raise RuntimeError(f'Malformed notebook cell {index}: expected an object')
        if cell.get('cell_type') != 'code':
            continue
        source = cell.get('source', '')
        if isinstance(source, list) and all(isinstance(part, str) for part in source):
            source = ''.join(source)
        if not isinstance(source, str):
            raise RuntimeError(f'Malformed source in notebook code cell {index}')
        try:
            tree = ast.parse(source, filename=f'{NOTEBOOK.name}:cell-{index}')
        except SyntaxError as exc:
            raise RuntimeError(f'Cannot parse notebook code cell {index}: {exc}') from exc
        parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id in definitions):
                continue
            parent = parents[node]
            simple = (isinstance(parent, ast.Assign) and len(parent.targets) == 1 and parent.targets[0] is node)
            annotated = isinstance(parent, ast.AnnAssign) and parent.target is node and parent.value is not None
            if not (parent in tree.body and (simple or annotated)):
                raise RuntimeError(f'{node.id} must be a simple top-level literal assignment (cell {index})')
            definitions[node.id].append((index, parent.value))

    values = {}
    for name, found in definitions.items():
        if len(found) != 1:
            raise RuntimeError(f'Expected exactly one {name} assignment in {NOTEBOOK.name}; found {len(found)}')
        index, expression = found[0]
        if name == 'core':
            if not isinstance(expression, ast.Dict):
                raise RuntimeError(f'core in notebook cell {index} must be a literal dictionary')
            try:
                keys = [ast.literal_eval(key) for key in expression.keys]
                if any(not isinstance(key, str) or not key.strip() for key in keys) or len(keys) != len(set(keys)):
                    raise ValueError('module keys must be unique non-empty strings')
            except (ValueError, TypeError) as exc:
                raise RuntimeError(f'Malformed core dictionary in notebook cell {index}: {exc}') from exc
        elif not isinstance(expression, ast.List):
            raise RuntimeError(f'{name} in notebook cell {index} must be a literal list')
        try:
            values[name] = ast.literal_eval(expression)
        except (ValueError, TypeError, SyntaxError) as exc:
            raise RuntimeError(f'{name} in notebook cell {index} must contain only literals: {exc}') from exc

    core, extra = values['core'], values['NOTEBOOK_REQUIREMENTS']
    if not core or not extra or any(not isinstance(v, str) or not v.strip() or '\n' in v or '\r' in v
                                    for v in [*core.values(), *extra]):
        raise RuntimeError('Notebook requirements must contain non-empty single-line requirement strings')
    return [*core.values(), *extra]


def verify_python(python: Path, requested: str) -> None:
    try:
        actual = subprocess.run([str(python), '-c', 'import sys; print(".".join(map(str, sys.version_info[:3])))'],
                                text=True, capture_output=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f'Cannot inspect existing environment interpreter {python}: {exc}') from exc
    parts = requested.split('.')
    if actual.split('.')[:len(parts)] != parts:
        raise RuntimeError(f'{python} is Python {actual}, but .python-version requests {requested}; '
                           'recreate that project-local environment with the requested Python')


def executable(env: Path, name: str) -> Path:
    if os.name == 'nt':
        return env / 'Scripts' / (name + '.exe')
    return env / 'bin' / name


def run(command, env):
    print('+', subprocess.list2cmdline([str(s) for s in command]), flush=True)
    subprocess.run([str(s) for s in command], cwd=ROOT, env=env, check=True)


def bootstrap(setup_only: bool, offline_check: bool) -> None:
    requested_python = python_version()
    requirements = notebook_requirements()
    tool_env = ROOT / '.uv-tool'
    local_uv = executable(tool_env, 'uv')
    system_uv = shutil.which('uv') or (str(local_uv) if local_uv.exists() else None)
    if not system_uv:
        raise RuntimeError('uv is required. Install it from https://docs.astral.sh/uv/getting-started/installation/ and retry.')
    env = os.environ.copy()
    env.setdefault('UV_CACHE_DIR', str(ROOT / '.uv-cache'))
    env.setdefault('UV_PYTHON_INSTALL_DIR', str(ROOT / '.uv-python'))
    env.setdefault('UV_TOOL_DIR', str(ROOT / '.uv-tools'))
    env.setdefault('UV_TOOL_BIN_DIR', str(ROOT / '.uv-bin'))
    env.setdefault('HF_HOME', str(ROOT / '.hf-cache'))
    env.setdefault('PYTHONUTF8', '1')
    if offline_check:
        env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', UV_OFFLINE='1')
    tool_python = executable(tool_env, 'python')
    if not tool_python.exists():
        run([system_uv, 'venv', '--python', requested_python, tool_env], env)
    version = None
    if local_uv.exists():
        version = subprocess.run([str(local_uv), '--version'], text=True, capture_output=True, env=env).stdout.strip()
    if not version or version.split()[:2] != ['uv', UV_VERSION]:
        run([system_uv, 'pip', 'install', '--python', tool_python, f'uv=={UV_VERSION}'], env)
    env['EDIT_LAB_UV'] = str(local_uv)
    for name in ('.venv', '.venv-model'):
        target = ROOT / name
        if not executable(target, 'python').exists():
            run([local_uv, 'venv', '--python', requested_python, target], env)
        verify_python(executable(target, 'python'), requested_python)
    core_python = executable(ROOT / '.venv', 'python')
    run([local_uv, 'pip', 'install', '--python', core_python, *requirements], env)
    if setup_only:
        print(f'Core environment ready: {core_python}\nModel environment: {ROOT / ".venv-model"}', flush=True)
        return
    command = [core_python, Path(__file__).resolve(), '--execute']
    if offline_check:
        command.append('--offline-check')
    run(command, env)


def kernel_spec(python: Path, parent: Path) -> None:
    directory = parent / 'kernels' / 'edit-fidelity-lab'
    directory.mkdir(parents=True)
    spec = {'argv': [str(python), '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': f'Edit Fidelity Lab (Python {python_version()})', 'language': 'python'}
    (directory / 'kernel.json').write_text(json.dumps(spec), encoding='utf-8')


def check_complete(output_dir: Path) -> dict:
    status = json.loads((output_dir / 'model_status.json').read_text(encoding='utf-8'))
    count = status.get('coverage')
    if count != 38 or status.get('expected') != 38:
        raise RuntimeError(f'Model evidence incomplete: {count}/{status.get("expected")} (expected 38/38); '
                           f'see {output_dir / "model_status.json"}')
    tests = json.loads((output_dir / 'test_results.json').read_text(encoding='utf-8'))
    if tests.get('passed', 0) < 32:
        raise RuntimeError(f'Expected at least 32 deterministic/model assertions, got {tests.get("passed")}')
    return status


def execute_one(offline: bool) -> None:
    import nbformat
    from jupyter_client.kernelspec import KernelSpecManager
    from jupyter_client import KernelManager
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter

    nb = nbformat.read(NOTEBOOK, as_version=4)
    with tempfile.TemporaryDirectory(prefix='edit-fidelity-') as scratch_str:
        scratch = Path(scratch_str)
        kernel_spec(Path(sys.executable), scratch)
        specs = KernelSpecManager(kernel_dirs=[str(scratch / 'kernels')], ensure_native_kernel=False)
        km = KernelManager(kernel_name='edit-fidelity-lab', kernel_spec_manager=specs)
        workdir = ROOT
        if offline:
            # Keep the replay offline even if a dependency tries to consult a hub.
            os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', UV_OFFLINE='1')
            source_cache = OUTPUTS / 'cache'
            if not source_cache.exists():
                raise RuntimeError('No model cache exists. Complete a live run before the offline check.')
            workdir = scratch / 'offline-run'
            (workdir / 'edit_fidelity_outputs').mkdir(parents=True)
            shutil.copytree(source_cache, workdir / 'edit_fidelity_outputs' / 'cache')
            settings = next(cell for cell in nb.cells if cell.cell_type == 'code' and re.search(r'^MODE\s*=', cell.source, re.M))
            settings.source, replaced = re.subn(r'(?m)^MODE\s*=\s*["\'](?:auto|live|offline)["\']', 'MODE = "offline"', settings.source, count=1)
            if replaced != 1:
                raise RuntimeError('Could not set offline mode in notebook settings.')
        else:
            # Only remove named lab figures; unrelated user files are preserved.
            figs = OUTPUTS / 'figures'
            if figs.exists():
                for path in figs.iterdir():
                    if re.match(r'^(?:0[1-9]|1[0-2])_.*\.(?:png|svg)$', path.name):
                        path.unlink()
        def started(cell, cell_index, **_kwargs):
            if cell.cell_type == 'code':
                print(f'Executing code cell {cell_index + 1}/{len(nb.cells)}', flush=True)
        client = NotebookClient(nb, km=km, timeout=None, startup_timeout=120,
                                allow_errors=False, on_cell_start=started,
                                resources={'metadata': {'path': str(workdir)}})
        try:
            client.execute(cwd=str(workdir), cleanup_kc=True)
            status = check_complete(workdir / 'edit_fidelity_outputs')
            if offline and status.get('origin') != 'validated local cache':
                raise RuntimeError(f'Offline replay did not use a validated local cache: {status.get("origin")}')
            if not offline:
                errors = [(i, output) for i, cell in enumerate(nb.cells) for output in cell.get('outputs', [])
                          if output.get('output_type') == 'error']
                if errors:
                    raise RuntimeError(f'Notebook contains {len(errors)} error outputs')
                uv = os.environ['EDIT_LAB_UV']
                for label, python in (('kernel', Path(sys.executable)),
                                      ('model', executable(ROOT / '.venv-model', 'python'))):
                    frozen = subprocess.run([uv, 'pip', 'freeze', '--python', str(python)],
                                            cwd=ROOT, text=True, capture_output=True, check=True)
                    (OUTPUTS / f'{label}_packages_freeze.txt').write_text(frozen.stdout, encoding='utf-8')
                rendered, _ = HTMLExporter().from_notebook_node(nb)
                notebook_tmp = NOTEBOOK.with_name(NOTEBOOK.name + '.tmp')
                preview_tmp = PREVIEW.with_name(PREVIEW.name + '.tmp')
                nbformat.write(nb, notebook_tmp)
                preview_tmp.write_text(rendered, encoding='utf-8')
                os.replace(notebook_tmp, NOTEBOOK)
                os.replace(preview_tmp, PREVIEW)
                print(f'Saved executed notebook and preview; {status["coverage"]}/{status["expected"]} real embeddings.', flush=True)
            else:
                print(f'Offline cache replay passed: {status["coverage"]}/{status["expected"]} real embeddings.', flush=True)
        except BaseException:
            if not offline:
                OUTPUTS.mkdir(exist_ok=True)
                nbformat.write(nb, OUTPUTS / 'failed_execution.ipynb')
                (OUTPUTS / 'execution_error.txt').write_text(traceback.format_exc(), encoding='utf-8')
                print(f'Execution failed; diagnostics: {OUTPUTS / "failed_execution.ipynb"} and execution_error.txt', file=sys.stderr, flush=True)
            raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setup-only', action='store_true', help='Create environments and install core dependencies, then stop')
    parser.add_argument('--offline-check', action='store_true', help='Replay the completed cache in an isolated temporary directory')
    parser.add_argument('--execute', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.setup_only and args.offline_check:
        parser.error('--setup-only and --offline-check cannot be combined')
    if args.execute:
        execute_one(args.offline_check)
    else:
        bootstrap(args.setup_only, args.offline_check)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)
