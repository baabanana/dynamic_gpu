"""The code shipped to instances must run on Python 3.8 with only the standard library."""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from dgpu import bundle
from test_runner_harness import old_python

SRC = Path(__file__).resolve().parent.parent / "src"
PACKAGE = SRC / "dgpu"
SHIPPED = sorted(
    [PACKAGE / "__init__.py", PACKAGE / "layout.py", PACKAGE / "states.py"]
    + list((PACKAGE / "runner").glob("*.py"))
)
RUNNER = sorted((PACKAGE / "runner").glob("*.py"))

# APIs and syntax newer than Python 3.8 that a 3.9+ compiler would not reject
NEWER_THAN_38 = [
    r"\.removeprefix\(",
    r"\.removesuffix\(",
    r"\bis_relative_to\(",
    r"\bwith_stem\(",
    r"waitstatus_to_exitcode",
    r"process_group\s*=",
    r"\brandbytes\(",
    r"\bzoneinfo\b",
    r"\bmath\.lcm\b",
    r"\bfunctools\.cache\b",
    r"^\s*@cache\b",
    r"\bpidfd_",
    r"\.bit_count\(",
    r"^\s*with\s*\(",
    r"\bpairwise\(",
    r"\bstrict\s*=\s*True",
    r"\bslots\s*=\s*True",
    r"\bkw_only\b",
    r"\bcontextlib\.chdir\b",
    r"\btomllib\b",
    r"\bdatetime\.UTC\b",
    r"\bTypeAlias\b",
    r"\bParamSpec\b",
    r"\bto_thread\(",
    r"isinstance\([^)]*\|",
]


def test_shipped_code_has_no_post_38_apis():
    problems = []
    for path in SHIPPED + [Path(bundle.__file__)]:
        lines = path.read_text().splitlines()
        if path.name == "bundle.py":
            text = bundle.BOOTSTRAP_SOURCE
            lines = text.splitlines()
        for number, line in enumerate(lines, 1):
            code = line.split("#", 1)[0]
            for pattern in NEWER_THAN_38:
                if re.search(pattern, code):
                    problems.append(f"{path.name}:{number}: {line.strip()}")
    assert problems == []


def test_runner_modules_postpone_annotations():
    for path in RUNNER:
        if path.name == "__init__.py":
            continue
        assert "from __future__ import annotations" in path.read_text(), path.name


def _imports(path: Path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, None
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            for alias in node.names:
                yield node.module, alias.name


def test_shipped_code_imports_only_stdlib_and_shipped_modules():
    allowed_dgpu = (
        {"dgpu", "dgpu.layout", "dgpu.states"}
        | {"dgpu.runner." + p.stem for p in RUNNER if p.stem != "__init__"}
        | {"dgpu.runner"}
    )
    problems = []
    for path in SHIPPED:
        for module, name in _imports(path):
            top = module.split(".")[0]
            if top == "dgpu":
                full = module if name is None or module != "dgpu" else "dgpu." + name
                if module == "dgpu.runner" and name is not None:
                    full = "dgpu.runner." + name
                if full not in allowed_dgpu:
                    problems.append(f"{path.name} imports {full}")
            elif top not in sys.stdlib_module_names:
                problems.append(f"{path.name} imports non-stdlib {module}")
    assert problems == []


@pytest.fixture(scope="module")
def python38_or_39():
    python = old_python()
    if python is None:
        pytest.skip("no Python 3.8 or 3.9 found by `uv python find 3.8` / `3.9`")
    return python


def test_shipped_code_compiles_on_old_python(python38_or_39, tmp_path):
    bootstrap = tmp_path / "bootstrap.py"
    bootstrap.write_text(bundle.BOOTSTRAP_SOURCE)
    files = [str(path) for path in SHIPPED] + [str(bootstrap)]
    script = "import sys\nfor f in sys.argv[1:]:\n    compile(open(f).read(), f, 'exec')\n"
    result = subprocess.run([python38_or_39, "-c", script] + files, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_shipped_code_imports_on_old_python(python38_or_39):
    modules = [f"dgpu.runner.{path.stem}" for path in RUNNER if path.stem != "__init__"]
    env = dict(os.environ, PYTHONPATH=str(SRC), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [
            python38_or_39,
            "-I",
            "-c",
            f"import sys; sys.path.insert(0, {str(SRC)!r})\n"
            + "\n".join(f"import {module}" for module in modules),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
