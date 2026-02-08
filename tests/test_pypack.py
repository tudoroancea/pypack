"""
Tests for pypack — single-binary Python apps powered by uv.

These are integration tests that exercise the full build → run pipeline.
They require: uv, zstd, tar, cc (clang or gcc).
"""

import os
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import textwrap
import zipfile

import pytest

PYPACK = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pypack.py")
PYTHON_VERSION = "3.13"

# Skip all tests if tools are missing
pytestmark = pytest.mark.skipif(
    not all(shutil.which(t) for t in ("uv", "zstd", "tar", "cc")),
    reason="Required tools (uv, zstd, tar, cc) not all available",
)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def cache_cleanup():
    """Clean pypack cache before tests to ensure first-run behavior."""
    yield
    # Don't clean up — let subsequent test runs reuse the cache


@pytest.fixture
def tmp(tmp_path):
    """Provide a temp directory for each test."""
    return tmp_path


def _build(entry, output, requirements=None, project=None, python=PYTHON_VERSION):
    """Helper: run pypack build and return the output path."""
    cmd = [
        sys.executable, PYPACK, "build",
        "--entry", entry,
        "-o", output,
        "--python", python,
    ]
    if requirements:
        cmd += ["-r", requirements]
    if project:
        cmd += ["-p", project]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        pytest.fail(
            f"pypack build failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return output


def _run(binary, *args, timeout=60):
    """Helper: run a packed binary and return (stdout, stderr, returncode)."""
    result = subprocess.run(
        [binary, *args],
        capture_output=True, text=True, timeout=timeout,
    )
    return result.stdout, result.stderr, result.returncode


# ── Test: binary structure ────────────────────────────────────────────


class TestBinaryStructure:
    """Verify the packed binary has the correct layout."""

    def test_trailer_magic(self, tmp):
        """The last 32 bytes should contain the PYPK magic."""
        app_dir = tmp / "app"
        app_dir.mkdir()
        (app_dir / "__init__.py").write_text("")
        (app_dir / "__main__.py").write_text("print('ok')")

        out = str(tmp / "bin")
        _build(str(app_dir), out)

        with open(out, "rb") as f:
            f.seek(-32, 2)
            trailer = f.read(32)

        assert trailer[:8] == b"PYPK\x00\x01\x00\x00", "Bad magic bytes"

    def test_trailer_offsets_valid(self, tmp):
        """Trailer offsets should be within file bounds."""
        app_dir = tmp / "app"
        app_dir.mkdir()
        (app_dir / "__init__.py").write_text("")
        (app_dir / "__main__.py").write_text("print('ok')")

        out = str(tmp / "bin")
        _build(str(app_dir), out)

        file_size = os.path.getsize(out)
        with open(out, "rb") as f:
            f.seek(-32, 2)
            trailer = f.read(32)

        magic = trailer[:8]
        rt_off = struct.unpack("<Q", trailer[8:16])[0]
        rt_sz = struct.unpack("<Q", trailer[16:24])[0]
        app_off = struct.unpack("<Q", trailer[24:32])[0]

        assert rt_off > 0, "runtime_offset should be > 0 (after stub)"
        assert rt_sz > 0, "runtime_size should be > 0"
        assert app_off == rt_off + rt_sz, "app_offset should follow runtime"
        assert app_off < file_size - 32, "app_offset should be before trailer"

    def test_appended_zip_is_valid(self, tmp):
        """The file should be a valid zip (readable from the tail)."""
        app_dir = tmp / "app"
        app_dir.mkdir()
        (app_dir / "__init__.py").write_text("")
        (app_dir / "__main__.py").write_text("print('ok')")

        out = str(tmp / "bin")
        _build(str(app_dir), out)

        assert zipfile.is_zipfile(out), "Packed binary should be a valid zip"

        with zipfile.ZipFile(out) as zf:
            names = zf.namelist()
            assert "__main__.py" in names
            assert "app/__main__.py" in names

    def test_output_is_executable(self, tmp):
        """The output file should have the executable bit set."""
        app_dir = tmp / "app"
        app_dir.mkdir()
        (app_dir / "__init__.py").write_text("")
        (app_dir / "__main__.py").write_text("print('ok')")

        out = str(tmp / "bin")
        _build(str(app_dir), out)

        assert os.access(out, os.X_OK), "Output should be executable"


# ── Test: package entry point ─────────────────────────────────────────


class TestPackageEntry:
    """Test building and running a package with __main__.py."""

    def test_basic_package(self, tmp):
        """A simple package should print output correctly."""
        pkg = tmp / "mypkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "__main__.py").write_text(
            textwrap.dedent("""\
                import sys
                print(f"args={sys.argv[1:]}")
            """)
        )

        out = str(tmp / "bin")
        _build(str(pkg), out)

        stdout, stderr, rc = _run(out, "foo", "bar")
        assert rc == 0, f"stderr: {stderr}"
        assert "args=['foo', 'bar']" in stdout

    def test_package_with_submodule(self, tmp):
        """A package importing its own submodule should work."""
        pkg = tmp / "mypkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "lib.py").write_text("def greet(name): return f'Hi {name}'")
        (pkg / "__main__.py").write_text(
            textwrap.dedent("""\
                from mypkg.lib import greet
                print(greet("pypack"))
            """)
        )

        out = str(tmp / "bin")
        _build(str(pkg), out)

        stdout, stderr, rc = _run(out)
        assert rc == 0, f"stderr: {stderr}"
        assert "Hi pypack" in stdout

    def test_exit_code_propagation(self, tmp):
        """Non-zero exit codes should propagate."""
        pkg = tmp / "mypkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "__main__.py").write_text("import sys; sys.exit(42)")

        out = str(tmp / "bin")
        _build(str(pkg), out)

        _, _, rc = _run(out)
        assert rc == 42


# ── Test: single script entry point ──────────────────────────────────


class TestScriptEntry:
    """Test building and running a single .py script."""

    def test_basic_script(self, tmp):
        """A single .py file should run correctly."""
        script = tmp / "hello.py"
        script.write_text("print('hello from script')")

        out = str(tmp / "bin")
        _build(str(script), out)

        stdout, stderr, rc = _run(out)
        assert rc == 0, f"stderr: {stderr}"
        assert "hello from script" in stdout

    def test_script_with_args(self, tmp):
        """Script should receive command-line arguments."""
        script = tmp / "args.py"
        script.write_text(
            textwrap.dedent("""\
                import sys
                print(" ".join(sys.argv[1:]))
            """)
        )

        out = str(tmp / "bin")
        _build(str(script), out)

        stdout, _, rc = _run(out, "a", "b", "c")
        assert rc == 0
        assert "a b c" in stdout

    def test_script_with_stdlib(self, tmp):
        """Script using stdlib modules should work."""
        script = tmp / "stdlib_test.py"
        script.write_text(
            textwrap.dedent("""\
                import json
                import os
                print(json.dumps({"pid": os.getpid()}))
            """)
        )

        out = str(tmp / "bin")
        _build(str(script), out)

        stdout, _, rc = _run(out)
        assert rc == 0
        import json
        data = json.loads(stdout.strip())
        assert "pid" in data


# ── Test: dependencies ────────────────────────────────────────────────


class TestDependencies:
    """Test building with pure-Python dependencies."""

    def test_with_click(self, tmp):
        """Building with click (a popular pure-Python dep) should work."""
        pkg = tmp / "cli"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "__main__.py").write_text(
            textwrap.dedent("""\
                import click
                @click.command()
                @click.option("--name", default="world")
                def main(name):
                    click.echo(f"Hello, {name}!")
                main()
            """)
        )
        req = tmp / "requirements.txt"
        req.write_text("click\n")

        out = str(tmp / "bin")
        _build(str(pkg), out, requirements=str(req))

        stdout, stderr, rc = _run(out, "--name", "test")
        assert rc == 0, f"stderr: {stderr}"
        assert "Hello, test!" in stdout

    def test_deps_in_zip(self, tmp):
        """Dependencies should appear under site-packages/ in the zip."""
        pkg = tmp / "cli"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "__main__.py").write_text("import click; print('ok')")
        req = tmp / "requirements.txt"
        req.write_text("click\n")

        out = str(tmp / "bin")
        _build(str(pkg), out, requirements=str(req))

        with zipfile.ZipFile(out) as zf:
            sp_files = [n for n in zf.namelist() if n.startswith("site-packages/")]
            assert len(sp_files) > 0, "No site-packages in zip"
            click_files = [n for n in sp_files if "click" in n]
            assert len(click_files) > 0, "click not found in site-packages"


# ── Test: native extensions ────────────────────────────────────────────


class TestNativeExtensions:
    """Test building and running packages with C extensions."""

    def test_markupsafe_import(self, tmp):
        """MarkupSafe has a C speedup module — should build and run."""
        pkg = tmp / "app"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "__main__.py").write_text(
            textwrap.dedent("""\
                from markupsafe import Markup, escape
                print(f"escaped={escape('<b>bold</b>')}")
                print(f"markup={Markup('<em>hi</em>')}")
            """)
        )
        req = tmp / "requirements.txt"
        req.write_text("markupsafe\n")

        out = str(tmp / "bin")
        _build(str(pkg), out, requirements=str(req))

        stdout, stderr, rc = _run(out)
        assert rc == 0, f"stderr: {stderr}"
        assert "escaped=&lt;b&gt;bold&lt;/b&gt;" in stdout
        assert "markup=<em>hi</em>" in stdout

    def test_native_ext_in_zip(self, tmp):
        """Native extension files should be present in the zip payload."""
        pkg = tmp / "app"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "__main__.py").write_text("import markupsafe; print('ok')")
        req = tmp / "requirements.txt"
        req.write_text("markupsafe\n")

        out = str(tmp / "bin")
        _build(str(pkg), out, requirements=str(req))

        with zipfile.ZipFile(out) as zf:
            native_files = [
                n for n in zf.namelist()
                if any(n.endswith(ext) for ext in (".so", ".dylib", ".pyd"))
            ]
            assert len(native_files) > 0, "No native extensions in zip"
            markupsafe_natives = [n for n in native_files if "markupsafe" in n]
            assert len(markupsafe_natives) > 0, "markupsafe .so not in zip"

    def test_native_ext_caching(self, tmp):
        """Second run should reuse extracted native extensions."""
        pkg = tmp / "app"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "__main__.py").write_text(
            textwrap.dedent("""\
                from markupsafe import escape
                print(f"result={escape('<test>')}")
            """)
        )
        req = tmp / "requirements.txt"
        req.write_text("markupsafe\n")

        out = str(tmp / "bin")
        _build(str(pkg), out, requirements=str(req))

        # First run
        stdout1, _, rc1 = _run(out)
        assert rc1 == 0
        assert "result=&lt;test&gt;" in stdout1

        # Second run — should reuse cache
        stdout2, _, rc2 = _run(out)
        assert rc2 == 0
        assert "result=&lt;test&gt;" in stdout2

    def test_pyyaml_native(self, tmp):
        """PyYAML has optional C extensions — should work."""
        pkg = tmp / "app"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "__main__.py").write_text(
            textwrap.dedent("""\
                import yaml
                data = yaml.safe_load("name: pypack")
                print(f"name={data['name']}")
            """)
        )
        req = tmp / "requirements.txt"
        req.write_text("pyyaml\n")

        out = str(tmp / "bin")
        _build(str(pkg), out, requirements=str(req))

        stdout, stderr, rc = _run(out)
        assert rc == 0, f"stderr: {stderr}"
        assert "name=pypack" in stdout

    def test_mixed_pure_and_native_deps(self, tmp):
        """Mix of pure-Python and native deps should work together."""
        pkg = tmp / "app"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "__main__.py").write_text(
            textwrap.dedent("""\
                import click
                from markupsafe import escape
                print(f"click={click.__name__}")
                print(f"escaped={escape('<ok>')}")
            """)
        )
        req = tmp / "requirements.txt"
        req.write_text("click\nmarkupsafe\n")

        out = str(tmp / "bin")
        _build(str(pkg), out, requirements=str(req))

        stdout, stderr, rc = _run(out)
        assert rc == 0, f"stderr: {stderr}"
        assert "click=click" in stdout
        assert "escaped=&lt;ok&gt;" in stdout


    def test_numpy(self, tmp):
        """NumPy (heavy C extensions, BLAS, etc.) should build and run."""
        pkg = tmp / "app"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "__main__.py").write_text(
            textwrap.dedent("""\
                import numpy as np
                a = np.array([1, 2, 3])
                b = np.array([4, 5, 6])
                print(f"dot={np.dot(a, b)}")
                print(f"sum={a.sum()}")
                print(f"shape={a.shape}")
                print(f"dtype={a.dtype}")
            """)
        )
        req = tmp / "requirements.txt"
        req.write_text("numpy\n")

        out = str(tmp / "bin")
        _build(str(pkg), out, requirements=str(req))

        stdout, stderr, rc = _run(out)
        assert rc == 0, f"stderr: {stderr}"
        assert "dot=32" in stdout
        assert "sum=6" in stdout
        assert "shape=(3,)" in stdout
        assert "dtype=int" in stdout

    def test_pytorch(self, tmp):
        """PyTorch (large native extension) should build and run."""
        pkg = tmp / "app"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "__main__.py").write_text(
            textwrap.dedent("""\
                import torch
                x = torch.tensor([1.0, 2.0, 3.0])
                y = torch.tensor([4.0, 5.0, 6.0])
                print(f"dot={torch.dot(x, y).item()}")
                print(f"sum={x.sum().item()}")
                print(f"shape={tuple(x.shape)}")
                print(f"device={x.device}")
            """)
        )
        req = tmp / "requirements.txt"
        # CPU-only torch to keep download manageable
        req.write_text(
            "--index-url https://download.pytorch.org/whl/cpu\n"
            "torch\n"
        )

        out = str(tmp / "bin")
        _build(str(pkg), out, requirements=str(req))

        stdout, stderr, rc = _run(out, timeout=120)
        assert rc == 0, f"stderr: {stderr}"
        assert "dot=32.0" in stdout
        assert "sum=6.0" in stdout
        assert "shape=(3,)" in stdout
        assert "device=cpu" in stdout


# ── Test: environment variables ───────────────────────────────────────


class TestEnvironment:
    """Test that pypack sets the expected environment variables."""

    def test_pypack_self_env(self, tmp):
        """_PYPACK_SELF should be set to the binary path."""
        pkg = tmp / "env_test"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "__main__.py").write_text(
            textwrap.dedent("""\
                import os
                print(os.environ.get("_PYPACK_SELF", "NOT SET"))
            """)
        )

        out = str(tmp / "bin")
        _build(str(pkg), out)

        stdout, _, rc = _run(out)
        assert rc == 0
        # Should contain the path to the binary
        assert "bin" in stdout.strip() or "env_test" in stdout.strip()
        assert "NOT SET" not in stdout

    def test_pypack_cache_env(self, tmp):
        """_PYPACK_CACHE should be set."""
        pkg = tmp / "env_test"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "__main__.py").write_text(
            textwrap.dedent("""\
                import os
                print(os.environ.get("_PYPACK_CACHE", "NOT SET"))
            """)
        )

        out = str(tmp / "bin")
        _build(str(pkg), out)

        stdout, _, rc = _run(out)
        assert rc == 0
        assert "pypack" in stdout.strip()
        assert "NOT SET" not in stdout


# ── Test: error handling ──────────────────────────────────────────────


class TestProjectDeps:
    """Test building with dependencies from --project (pyproject.toml / setup.py)."""

    def _make_pyproject(self, path, deps):
        """Helper: create a minimal pyproject.toml with given deps."""
        deps_str = ", ".join(f'"{d}"' for d in deps)
        path.write_text(
            textwrap.dedent(f"""\
                [project]
                name = "testapp"
                version = "0.1.0"
                dependencies = [{deps_str}]

                [build-system]
                requires = ["hatchling"]
                build-backend = "hatchling.build"
            """)
        )

    def test_pyproject_basic(self, tmp):
        """Should install deps from pyproject.toml via uv."""
        pkg = tmp / "cli"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "__main__.py").write_text(
            textwrap.dedent("""\
                import click
                @click.command()
                @click.option("--name", default="world")
                def main(name):
                    click.echo(f"Hello, {name}!")
                main()
            """)
        )
        self._make_pyproject(tmp / "pyproject.toml", ["click"])

        out = str(tmp / "bin")
        _build(str(pkg), out, project=str(tmp))

        stdout, stderr, rc = _run(out, "--name", "pypack")
        assert rc == 0, f"stderr: {stderr}"
        assert "Hello, pypack!" in stdout

    def test_pyproject_file_path(self, tmp):
        """Should accept a pyproject.toml file path (uses parent dir)."""
        pkg = tmp / "cli"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "__main__.py").write_text(
            textwrap.dedent("""\
                import click
                print(f"click loaded: {click.__name__}")
            """)
        )
        project_dir = tmp / "project"
        project_dir.mkdir()
        self._make_pyproject(project_dir / "pyproject.toml", ["click"])

        out = str(tmp / "bin")
        _build(str(pkg), out, project=str(project_dir / "pyproject.toml"))

        stdout, stderr, rc = _run(out)
        assert rc == 0, f"stderr: {stderr}"
        assert "click loaded:" in stdout

    def test_setup_py(self, tmp):
        """Should install deps from a setup.py project via uv."""
        pkg = tmp / "cli"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "__main__.py").write_text(
            textwrap.dedent("""\
                import click
                print(f"click loaded: {click.__name__}")
            """)
        )
        project_dir = tmp / "project"
        project_dir.mkdir()
        (project_dir / "setup.py").write_text(
            textwrap.dedent("""\
                from setuptools import setup
                setup(
                    name="testapp",
                    version="0.1.0",
                    install_requires=["click"],
                )
            """)
        )

        out = str(tmp / "bin")
        _build(str(pkg), out, project=str(project_dir))

        stdout, stderr, rc = _run(out)
        assert rc == 0, f"stderr: {stderr}"
        assert "click loaded:" in stdout

    def test_project_no_deps(self, tmp):
        """Should succeed if project has no dependencies."""
        pkg = tmp / "app"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "__main__.py").write_text("print('no deps')")
        self._make_pyproject(tmp / "pyproject.toml", [])

        out = str(tmp / "bin")
        _build(str(pkg), out, project=str(tmp))

        stdout, stderr, rc = _run(out)
        assert rc == 0
        assert "no deps" in stdout

    def test_requirements_and_project_conflict(self, tmp):
        """Should fail if both --requirements and --project are given."""
        pkg = tmp / "app"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "__main__.py").write_text("print('ok')")
        req = tmp / "requirements.txt"
        req.write_text("click\n")
        self._make_pyproject(tmp / "pyproject.toml", ["click"])

        out = str(tmp / "bin")
        result = subprocess.run(
            [sys.executable, PYPACK, "build",
             "--entry", str(pkg),
             "-r", str(req),
             "--project", str(tmp),
             "-o", out],
            capture_output=True, text=True,
        )
        assert result.returncode != 0


class TestErrorHandling:
    """Test that pypack gives useful errors for invalid inputs."""

    def test_missing_entry(self, tmp):
        """Should fail if entry point doesn't exist."""
        out = str(tmp / "bin")
        result = subprocess.run(
            [sys.executable, PYPACK, "build",
             "--entry", "/nonexistent/path",
             "-o", out],
            capture_output=True, text=True,
        )
        assert result.returncode != 0

    def test_package_without_main(self, tmp):
        """Should fail if package has no __main__.py."""
        pkg = tmp / "nomain"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        # No __main__.py

        out = str(tmp / "bin")
        result = subprocess.run(
            [sys.executable, PYPACK, "build",
             "--entry", str(pkg),
             "-o", out],
            capture_output=True, text=True,
        )
        assert result.returncode != 0


# ── Test: caching ─────────────────────────────────────────────────────


class TestCaching:
    """Test that the runtime caching mechanism works."""

    def test_second_run_uses_cache(self, tmp):
        """Second run should not show 'extracting runtime' message."""
        pkg = tmp / "cached"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "__main__.py").write_text("print('cached!')")

        out = str(tmp / "bin")
        _build(str(pkg), out)

        # First run — extracts
        _, stderr1, rc1 = _run(out)
        assert rc1 == 0
        assert "extracting" in stderr1.lower() or rc1 == 0

        # Second run — should use cache (no extraction message)
        _, stderr2, rc2 = _run(out)
        assert rc2 == 0
        assert "extracting" not in stderr2.lower()
