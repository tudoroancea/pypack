#!/usr/bin/env python3
"""
pypack — Build single-binary Python executables powered by uv.

Bundles a python-build-standalone runtime (via uv) with your code and
pure-Python dependencies into a single executable file.

Prerequisites:
  - uv    (https://docs.astral.sh/uv/)
  - zstd  (apt install zstd / brew install zstd)
  - cc    (any C compiler: gcc, clang, etc.)

Usage:
  python pypack.py build --entry myapp/ -r requirements.txt -o dist/myapp
  python pypack.py build --entry script.py -o dist/script --python 3.12
"""

import argparse
import io
import os
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile

MAGIC = b"PYPK\x00\x01\x00\x00"
TRAILER_SZ = 32
STUB_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stub.c")

# Stdlib modules/directories that are safe to strip — large and rarely needed
# at runtime by typical CLI/server applications.
STRIP_STDLIB_DIRS = [
    # Test infrastructure (huge, never needed at runtime)
    "test",
    # GUI / Tk
    "tkinter",
    "turtledemo",
    "idlelib",
    # Package management (not needed inside a packed binary)
    "ensurepip",
    # Documentation
    "pydoc_data",
    # Deprecated / compat
    "lib2to3",
    "distutils",
]

STRIP_STDLIB_FILES = [
    "turtle.py",
]

# Top-level dirs outside lib/pythonX.Y that are safe to remove
STRIP_TOPLEVEL_DIRS = [
    "include",      # C headers — not needed at runtime
    "share",        # man pages, etc.
]

# Lib-level dirs (siblings to lib/pythonX.Y) that are tk/tcl related
STRIP_LIB_PATTERNS = [
    "tcl",          # matches tcl8.6, tcl8, etc.
    "tk",           # matches tk8.6
    "itcl",         # Tcl extension
    "tdbc",         # Tcl extension
    "thread",       # Tcl threading extension
    "libtcl",       # shared libraries
    "libtk",        # shared libraries
    "Tix",          # Tk extension
]


# ── Utility ───────────────────────────────────────────────────────────

def die(msg):
    print(f"\n  ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def require_tool(name, install_hint=None):
    if shutil.which(name) is None:
        hint = f" ({install_hint})" if install_hint else ""
        die(f"'{name}' is not installed{hint}")


# ── uv integration ───────────────────────────────────────────────────

def uv_ensure_python(version):
    """Install and locate a Python interpreter via uv."""
    print(f"[1/7] Acquiring Python {version} via uv...")
    subprocess.run(["uv", "python", "install", version],
                   check=True, capture_output=True)
    r = subprocess.run(
        ["uv", "python", "find", version],
        capture_output=True, text=True, check=True,
    )
    py = r.stdout.strip()
    if not os.path.isfile(py):
        die(f"uv python find returned non-existent path: {py}")
    return py


def uv_install_root(python_path):
    """
    Walk up from the interpreter binary to find the PBS installation root.

    uv stores them at paths like:
      ~/.local/share/uv/python/cpython-3.13.1-linux-x86_64-none/
    The interpreter binary is somewhere like:
      .../cpython-3.13.1-linux-x86_64-none/bin/python3
    or:
      .../cpython-3.13.1-linux-x86_64-none/install/bin/python3
    """
    parts = python_path.split(os.sep)
    for i, part in enumerate(parts):
        if part.startswith("cpython-") or part.startswith("pypy-"):
            return os.sep + os.path.join(*parts[1:i + 1])

    # Fallback: assume layout is .../bin/python3, walk up
    bin_dir = os.path.dirname(python_path)
    parent = os.path.dirname(bin_dir)
    if os.path.isdir(os.path.join(parent, "lib")):
        return parent
    return os.path.dirname(parent)


def _check_native_extensions(target_dir):
    """Report native extensions found in target_dir (informational)."""
    native_files = []
    for root, _, files in os.walk(target_dir):
        for f in files:
            if f.endswith((".so", ".dylib", ".pyd")):
                native_files.append(
                    os.path.relpath(os.path.join(root, f), target_dir)
                )

    if native_files:
        print(f"       Native extensions found ({len(native_files)} files):")
        for nf in native_files[:5]:
            print(f"         • {nf}")
        if len(native_files) > 5:
            print(f"         ... and {len(native_files) - 5} more")
        print("       These will be extracted to cache at first run.")


def uv_install_deps(python_path, req_file, target_dir):
    """Install pure-Python dependencies via 'uv pip install --target'."""
    print(f"[2/7] Installing dependencies via uv pip...")
    os.makedirs(target_dir, exist_ok=True)
    subprocess.run([
        "uv", "pip", "install",
        "--python", python_path,
        "--target", target_dir,
        "-r", req_file,
        "--no-compile",
    ], check=True)

    _check_native_extensions(target_dir)


def uv_install_project_deps(python_path, project_dir, target_dir):
    """Install a project's dependencies via uv.

    Works with any project format uv understands: pyproject.toml,
    setup.py, setup.cfg. Uses 'uv pip compile' to resolve deps from
    the project metadata, then 'uv pip install' to install them.
    """
    print(f"[2/7] Installing project dependencies via uv pip...")

    # Find the project metadata file
    for name in ("pyproject.toml", "setup.cfg", "setup.py"):
        src = os.path.join(project_dir, name)
        if os.path.isfile(src):
            break
    else:
        die(f"No pyproject.toml, setup.cfg, or setup.py found in {project_dir}")

    # Resolve dependencies to a pinned requirements list
    r = subprocess.run(
        ["uv", "pip", "compile", "--python", python_path, src],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        die(f"uv pip compile failed:\n{r.stderr}")

    # Filter out comments and blank lines
    lines = [
        l.strip() for l in r.stdout.splitlines()
        if l.strip() and not l.strip().startswith("#")
    ]
    if not lines:
        print("       No dependencies found.")
        return

    # Write to a temp file and install
    os.makedirs(target_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", prefix="pypack-reqs-", delete=False
    ) as tmp:
        tmp.write("\n".join(lines) + "\n")
        tmp_path = tmp.name

    try:
        subprocess.run([
            "uv", "pip", "install",
            "--python", python_path,
            "--target", target_dir,
            "-r", tmp_path,
            "--no-compile",
        ], check=True)
    finally:
        os.unlink(tmp_path)

    _check_native_extensions(target_dir)


# ── Stdlib tree-shaking ───────────────────────────────────────────────

def _find_lib_python(install_dir):
    """Find the lib/pythonX.Y directory inside a PBS installation."""
    import glob
    patterns = [
        os.path.join(install_dir, "lib", "python3.*"),
        os.path.join(install_dir, "install", "lib", "python3.*"),
        os.path.join(install_dir, "python", "install", "lib", "python3.*"),
    ]
    for pat in patterns:
        matches = glob.glob(pat)
        if matches:
            return matches[0]
    return None


def strip_runtime(install_dir):
    """Strip unused stdlib modules and support files from a runtime copy.

    Removes large modules that are almost never needed at runtime:
    tkinter, idlelib, test, ensurepip, turtledemo, pydoc_data, etc.
    Also removes Tcl/Tk shared libraries and C headers.

    Returns the total bytes saved.
    """
    saved = 0

    def _rm(path):
        nonlocal saved
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for f in files:
                    saved += os.path.getsize(os.path.join(root, f))
            shutil.rmtree(path)
        elif os.path.isfile(path):
            saved += os.path.getsize(path)
            os.unlink(path)

    # 1. Strip top-level dirs (include/, share/)
    for d in STRIP_TOPLEVEL_DIRS:
        p = os.path.join(install_dir, d)
        _rm(p)

    # 2. Find lib/pythonX.Y and strip stdlib modules
    lib_python = _find_lib_python(install_dir)
    if lib_python:
        for d in STRIP_STDLIB_DIRS:
            _rm(os.path.join(lib_python, d))
        for f in STRIP_STDLIB_FILES:
            _rm(os.path.join(lib_python, f))

        # Strip __pycache__ dirs (pyc files, can be regenerated)
        for root, dirs, files in os.walk(lib_python):
            for d in dirs[:]:
                if d == "__pycache__":
                    _rm(os.path.join(root, d))
                    dirs.remove(d)

        # Strip site-packages (PBS ships pip, not needed)
        _rm(os.path.join(lib_python, "site-packages"))

    # 3. Strip Tcl/Tk libraries from lib/
    lib_dir = os.path.dirname(lib_python) if lib_python else os.path.join(install_dir, "lib")
    if os.path.isdir(lib_dir):
        for entry in os.listdir(lib_dir):
            entry_lower = entry.lower()
            for pattern in STRIP_LIB_PATTERNS:
                if entry_lower.startswith(pattern):
                    _rm(os.path.join(lib_dir, entry))
                    break

    return saved


# ── Stub compilation ─────────────────────────────────────────────────

def compile_stub(work_dir):
    """Compile the C stub for the current platform."""
    print("[3/7] Compiling C stub...")

    # Check for pre-built stubs first
    plat = f"{platform.system().lower()}-{platform.machine().lower()}"
    prebuilt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stubs")
    prebuilt = os.path.join(prebuilt_dir, f"stub-{plat}")
    if os.path.isfile(prebuilt):
        out = os.path.join(work_dir, "stub")
        shutil.copy2(prebuilt, out)
        print(f"       Using pre-built stub for {plat}")
        return out

    # Compile from source
    if not os.path.isfile(STUB_SRC):
        die(f"stub.c not found at {STUB_SRC}")

    require_tool("cc", "install gcc or clang")

    out = os.path.join(work_dir, "stub")
    cc = os.environ.get("CC", "cc")

    # Try static linking on Linux for maximum portability
    if platform.system() == "Linux":
        r = subprocess.run(
            [cc, "-O2", "-s", "-static", "-o", out, STUB_SRC],
            capture_output=True,
        )
        if r.returncode == 0:
            print("       Compiled (static)")
            return out
        print("       Static linking failed, falling back to dynamic")
        subprocess.run([cc, "-O2", "-s", "-o", out, STUB_SRC], check=True)
    else:
        # macOS: -s is not supported by Apple clang linker
        subprocess.run([cc, "-O2", "-o", out, STUB_SRC], check=True)

    print("       Compiled (dynamic)")
    return out


# ── App ZIP payload ──────────────────────────────────────────────────

def _make_bootstrap(entry_path):
    """Generate the __main__.py bootstrap that goes inside the zip."""
    is_package = os.path.isdir(entry_path)

    if is_package:
        module_name = os.path.basename(os.path.normpath(entry_path))
        run_stmt = (
            f'runpy.run_module("{module_name}", '
            f'run_name="__main__", alter_sys=True)'
        )
    else:
        # For single scripts, we store them as _pypack_entry.py in the zip
        # to avoid name collisions and use run_module to execute them
        run_stmt = (
            'runpy.run_module("_pypack_entry", '
            'run_name="__main__", alter_sys=True)'
        )

    return f'''"""pypack bootstrap — auto-generated, do not edit."""
import sys
import os
import runpy


_NATIVE_EXTS = (".so", ".dylib", ".pyd")


def _extract_native_extensions(self_path, cache_dir):
    """Extract packages containing native extensions from the zip to disk.

    zipimport cannot load .so/.dylib/.pyd files, so we extract the entire
    top-level package for any package that contains native extensions.
    Returns the path to the extracted site-packages dir, or None.
    """
    import zipfile
    import hashlib

    if not zipfile.is_zipfile(self_path):
        return None

    with zipfile.ZipFile(self_path) as zf:
        names = zf.namelist()

        # Find native extensions under site-packages/
        native_files = [
            n for n in names
            if n.startswith("site-packages/")
            and any(n.endswith(ext) for ext in _NATIVE_EXTS)
        ]

        if not native_files:
            return None

        # Determine which top-level packages contain native extensions
        native_packages = set()
        for nf in native_files:
            parts = nf.split("/")
            if len(parts) >= 2:
                native_packages.add(parts[1])

        # Compute cache key from native file names + sizes
        hasher = hashlib.sha256()
        for nf in sorted(native_files):
            info = zf.getinfo(nf)
            hasher.update(f"{{nf}}:{{info.file_size}}:{{info.CRC}}".encode())
        cache_key = hasher.hexdigest()[:16]

        extract_dir = os.path.join(cache_dir, "native", cache_key)
        sp_dir = os.path.join(extract_dir, "site-packages")
        marker = os.path.join(extract_dir, ".done")

        # Already extracted?
        if os.path.exists(marker):
            return sp_dir

        # Extract all files belonging to native packages
        os.makedirs(sp_dir, exist_ok=True)
        for name in names:
            if not name.startswith("site-packages/"):
                continue
            parts = name.split("/")
            if len(parts) < 2 or parts[1] not in native_packages:
                continue
            target = os.path.join(extract_dir, name)
            if name.endswith("/"):
                os.makedirs(target, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, "wb") as f:
                    f.write(zf.read(name))

        # Mark extraction complete
        with open(marker, "w") as f:
            f.write("ok\\n")

        return sp_dir


def _pypack_main():
    self_path = os.environ.get("_PYPACK_SELF", os.path.abspath(sys.argv[0]))
    cache_dir = os.environ.get("_PYPACK_CACHE", "")

    # Extract native extensions to disk if needed
    native_sp = None
    if cache_dir:
        native_sp = _extract_native_extensions(self_path, cache_dir)

    # Build sys.path in priority order:
    #   1. self_path       — user code (__main__.py, package modules)
    #   2. native_sp       — extracted native packages (filesystem, can os.listdir)
    #   3. zip_site        — pure-Python deps (via zipimport)
    zip_site = os.path.join(self_path, "site-packages")

    # Insert in reverse priority order at position 0
    if zip_site not in sys.path:
        sys.path.insert(0, zip_site)
    if native_sp and native_sp not in sys.path:
        sys.path.insert(0, native_sp)
    if self_path not in sys.path:
        sys.path.insert(0, self_path)

    # Present the binary path as argv[0] to user code
    sys.argv[0] = self_path

    # Run the entry point
    {run_stmt}


_pypack_main()
'''


def make_app_zip(entry_path, deps_dir=None):
    """Create the in-memory ZIP payload containing bootstrap + code + deps."""
    print("[4/7] Creating app payload...")
    buf = io.BytesIO()

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        # 1. Bootstrap __main__.py
        zf.writestr("__main__.py", _make_bootstrap(entry_path))

        # 2. User code
        if os.path.isdir(entry_path):
            base = os.path.dirname(os.path.normpath(entry_path))
            for root, dirs, files in os.walk(entry_path):
                dirs[:] = [
                    d for d in dirs
                    if d != "__pycache__" and not d.startswith(".")
                ]
                for f in files:
                    if f.endswith((".pyc", ".pyo")):
                        continue
                    filepath = os.path.join(root, f)
                    arcname = os.path.relpath(filepath, base)
                    zf.write(filepath, arcname)
        else:
            # Store single scripts as _pypack_entry.py so run_module can find them
            zf.write(entry_path, "_pypack_entry.py")

        # 3. Dependencies (from uv pip install --target)
        if deps_dir and os.path.isdir(deps_dir):
            for root, dirs, files in os.walk(deps_dir):
                dirs[:] = [d for d in dirs if d != "__pycache__"]
                for f in files:
                    if f.endswith((".pyc", ".pyo")):
                        continue
                    filepath = os.path.join(root, f)
                    arcname = os.path.join(
                        "site-packages",
                        os.path.relpath(filepath, deps_dir),
                    )
                    zf.write(filepath, arcname)

    data = buf.getvalue()
    print(f"       {len(data) / 1024:.0f} KB")
    return data


# ── Runtime compression ──────────────────────────────────────────────

def prepare_runtime(install_dir, work_dir, do_strip=True):
    """Copy the runtime to work_dir and optionally strip unused stdlib modules.

    Returns the path to the (possibly stripped) runtime directory.
    """
    runtime_copy = os.path.join(work_dir, "runtime")
    print(f"[5/7] Preparing Python runtime...")
    shutil.copytree(install_dir, runtime_copy, symlinks=True)

    if do_strip:
        saved = strip_runtime(runtime_copy)
        mb_saved = saved / 1024 / 1024
        print(f"       Stripped {mb_saved:.1f} MB of unused stdlib modules")
    else:
        print("       Skipping stdlib stripping (--no-strip)")

    return runtime_copy


def compress_runtime(install_dir, output_path):
    """Tar + zstd-compress the Python installation directory."""
    print(f"[6/7] Compressing Python runtime...")
    cmd = f"tar -cf - -C '{install_dir}' . | zstd -19 -q -o '{output_path}'"
    subprocess.run(cmd, shell=True, check=True)
    mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"       {mb:.1f} MB compressed")


# ── Final assembly ───────────────────────────────────────────────────

def assemble(stub_path, runtime_path, app_zip_bytes, output_path):
    """Concatenate stub + runtime + app_zip + trailer → final binary."""
    print("[7/7] Assembling binary...")

    with open(output_path, "wb") as out:
        # 1. Stub
        with open(stub_path, "rb") as f:
            out.write(f.read())
        runtime_offset = out.tell()

        # 2. Runtime blob
        with open(runtime_path, "rb") as f:
            runtime_data = f.read()
        out.write(runtime_data)
        runtime_size = len(runtime_data)
        app_offset = out.tell()

        # 3. App ZIP
        out.write(app_zip_bytes)

        # 4. Trailer (32 bytes)
        out.write(MAGIC)                                     # 8 bytes
        out.write(struct.pack("<Q", runtime_offset))         # 8 bytes
        out.write(struct.pack("<Q", runtime_size))           # 8 bytes
        out.write(struct.pack("<Q", app_offset))             # 8 bytes

    os.chmod(output_path, 0o755)

    total_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"       Total: {total_mb:.1f} MB → {output_path}")


# ── Build command ────────────────────────────────────────────────────

def cmd_build(args):
    """Execute the full build pipeline."""
    # Preflight checks
    require_tool("uv", "curl -LsSf https://astral.sh/uv/install.sh | sh")
    require_tool("zstd", "apt install zstd / brew install zstd")
    require_tool("tar")

    entry = os.path.normpath(args.entry)
    if not os.path.exists(entry):
        die(f"Entry point not found: {entry}")

    if os.path.isdir(entry):
        main_py = os.path.join(entry, "__main__.py")
        if not os.path.isfile(main_py):
            die(f"Package '{entry}' has no __main__.py")

    if args.requirements and args.project:
        die("Cannot use both --requirements and --project at the same time.")

    do_strip = not args.no_strip

    python_version = args.python or "3.13"

    # Resolve --project to a directory
    project_dir = None
    if args.project:
        project_dir = os.path.normpath(args.project)
        if os.path.isfile(project_dir):
            # If a file was given (e.g. pyproject.toml), use its parent dir
            project_dir = os.path.dirname(project_dir)
        if not os.path.isdir(project_dir):
            die(f"Project directory not found: {project_dir}")

    print(f"\n  pypack build")
    print(f"  entry:   {entry}")
    print(f"  python:  {python_version}")
    print(f"  output:  {args.output}")
    if args.requirements:
        print(f"  deps:    {args.requirements}")
    if project_dir:
        print(f"  project: {project_dir}")
    print(f"  strip:   {'yes' if do_strip else 'no'}")
    print()

    with tempfile.TemporaryDirectory(prefix="pypack-") as work_dir:
        # 1. Get Python via uv
        python_path = uv_ensure_python(python_version)
        install_root = uv_install_root(python_path)
        print(f"       interpreter:  {python_path}")
        print(f"       install root: {install_root}")

        # 2. Install deps
        deps_dir = None
        if args.requirements:
            deps_dir = os.path.join(work_dir, "deps")
            uv_install_deps(python_path, args.requirements, deps_dir)
        elif project_dir:
            deps_dir = os.path.join(work_dir, "deps")
            uv_install_project_deps(python_path, project_dir, deps_dir)

        # 3. Compile C stub
        stub_path = compile_stub(work_dir)

        # 4. Create app ZIP
        app_zip = make_app_zip(entry, deps_dir)

        # 5. Copy & strip runtime
        runtime_dir = prepare_runtime(install_root, work_dir, do_strip=do_strip)

        # 6. Compress runtime
        runtime_path = os.path.join(work_dir, "runtime.tar.zst")
        compress_runtime(runtime_dir, runtime_path)

        # 7. Assemble final binary
        output = args.output
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        assemble(stub_path, runtime_path, app_zip, output)

    print(f"\n  [✓] Build complete!")
    print(f"      Run with: ./{output}")
    print(f"      First run extracts the runtime to ~/.cache/pypack/")
    print(f"      Subsequent runs start in ~100ms.\n")


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="pypack",
        description="Single-binary Python executables, powered by uv.",
        epilog="https://github.com/user/pypack",  # placeholder
    )
    subparsers = parser.add_subparsers(dest="command")

    # build subcommand
    build_parser = subparsers.add_parser(
        "build",
        help="Pack a Python app into a single executable",
    )
    build_parser.add_argument(
        "--entry", required=True,
        help="Entry point: a .py script or a package directory with __main__.py",
    )
    build_parser.add_argument(
        "-o", "--output", required=True,
        help="Path for the output binary",
    )
    build_parser.add_argument(
        "--python", default="3.13",
        help="Python version to bundle (default: 3.13)",
    )
    build_parser.add_argument(
        "-r", "--requirements",
        help="Path to requirements.txt for dependencies",
    )
    build_parser.add_argument(
        "-p", "--project",
        help="Path to a project directory (or its pyproject.toml/setup.py); "
             "dependencies are resolved by uv",
    )
    build_parser.add_argument(
        "--no-strip", action="store_true", default=False,
        help="Don't strip unused stdlib modules from the runtime "
             "(keeps tkinter, idlelib, test, etc.)",
    )

    args = parser.parse_args()

    if args.command == "build":
        cmd_build(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
