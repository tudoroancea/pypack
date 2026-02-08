# pypack

Single-binary Python apps, powered by [uv](https://docs.astral.sh/uv/).

Turn any pure-Python project into a single executable. No Python installed on the target. No virtualenv. Just `./myapp`.

```
$ python pypack.py build --entry myapp/ -r requirements.txt -o dist/myapp
[✓] Build complete!

$ scp dist/myapp server:~/
$ ssh server ./myapp   # just works — nothing else needed
```

## How it works

1. **uv** fetches a standalone CPython build from [python-build-standalone](https://github.com/astral-sh/python-build-standalone).
2. Your code + dependencies are zipped into a payload (Python's `zipimport` imports directly from zip files).
3. A small C stub (~200 lines) is compiled for the target platform. At runtime, it reads itself, extracts the Python runtime to a cache, and execs Python pointing at the appended zip.
4. Everything is concatenated into one file: `[stub] + [compressed runtime] + [app zip] + [trailer]`.

The result is a single file that is simultaneously a native executable (read from the head) and a valid zip (read from the tail) — the same trick used by Java JAR files.

### Binary layout

```
┌──────────────────────────────────┐  offset 0
│  C stub (ELF / Mach-O)          │  ~50 KB compiled
│  - reads trailer at own EOF     │
│  - extracts runtime to cache    │
│  - execs: python <self>         │
├──────────────────────────────────┤  offset A
│  Python runtime (tar.zst)       │  ~15 MB compressed
│  (python-build-standalone via uv)│
│  - interpreter + stripped stdlib │
├──────────────────────────────────┤  offset B
│  App payload (ZIP)              │  variable
│  - __main__.py  (bootstrap)     │
│  - app/         (user code)     │
│  - site-packages/ (pure deps)   │
├──────────────────────────────────┤  offset C
│  Trailer (32 bytes)             │
│  - magic: b"PYPK\x00\x01"      │  8 bytes
│  - runtime_offset: u64 LE       │  8 bytes
│  - runtime_size: u64 LE         │  8 bytes
│  - app_offset: u64 LE           │  8 bytes
└──────────────────────────────────┘
```

### Runtime flow

```
User runs: ./myapp --flag1 --flag2
              │
              ▼
   ┌─────────────────────────┐
   │  C stub reads itself    │
   │  via /proc/self/exe (L) │
   │  or _NSGetExecPath (M)  │
   └────────────┬────────────┘
                │
                ▼
   ┌─────────────────────────┐
   │  Read trailer (last 32B)│
   │  Validate magic bytes   │
   └────────────┬────────────┘
                │
                ▼
   ┌──────────────────────────────────┐
   │  Hash the runtime blob (SHA-256) │
   │  Cache key: first 16 hex chars   │
   │  Cache dir: ~/.cache/pypack/<h>  │
   └────────────┬─────────────────────┘
                │
           ┌────┴────┐
           │ Cached?  │
           └────┬────┘
         no     │     yes
         ▼      │      ▼
   ┌──────────┐ │ ┌───────────────┐
   │ zstd -d  │ │ │ Skip extract  │
   │ | tar -x │ │ └──────┬────────┘
   │ to cache │ │        │
   └────┬─────┘ │        │
        └────┬──┘────────┘
             ▼
   ┌──────────────────────────────┐
   │  exec(cache/python, self,   │
   │       argv[1], argv[2], ...)│
   │                              │
   │  Python sees self is a zip   │
   │  (zipimport), finds and runs │
   │  __main__.py from the tail   │
   └──────────────────────────────┘
```

**Key insight:** ZIP central directories live at the end of a file. You can prepend
arbitrary bytes and it remains a valid zip. So `./myapp` is both an ELF/Mach-O executable
AND a zip archive. When Python is invoked as `python ./myapp`, zipimport finds
`__main__.py` inside the appended zip and runs it. No temp files for app code.

## Prerequisites

**Build machine:**

| Tool | Purpose | Install |
|------|---------|---------|
| [uv](https://docs.astral.sh/uv/) | Python runtime + deps | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| zstd | Runtime compression | `apt install zstd` / `brew install zstd` |
| cc | Compile C stub | Any C compiler (gcc, clang) |
| tar | Runtime archiving | Pre-installed on Linux/macOS |

**Target machine** (where the binary runs):

| Tool | Purpose |
|------|---------|
| zstd | Decompress runtime (first run only) |
| tar | Extract runtime (first run only) |
| sha256sum / shasum | Cache key computation |

## Usage

### Package with `__main__.py`

```bash
python pypack.py build \
    --entry myapp/ \
    -r requirements.txt \
    -o dist/myapp \
    --python 3.13
```

### Single script

```bash
python pypack.py build \
    --entry script.py \
    -o dist/script
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--entry` | `.py` file or package directory with `__main__.py` | (required) |
| `-o, --output` | Output binary path | (required) |
| `--python` | Python version to bundle | `3.13` |
| `-r, --requirements` | Path to `requirements.txt` | (none) |

## Examples

### Hello world (package)

```python
# myapp/__init__.py
# myapp/__main__.py
import sys

def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "world"
    print(f"Hello, {name}!")

main()
```

```bash
python pypack.py build --entry myapp/ -o dist/hello
./dist/hello pypack
# Hello, pypack!
```

### CLI with dependencies

```python
# cli/__main__.py
import click

@click.command()
@click.option("--name", default="world")
def main(name):
    click.echo(f"Hello, {name}!")

main()
```

```bash
echo "click" > requirements.txt
python pypack.py build --entry cli/ -r requirements.txt -o dist/cli
./dist/cli --name pypack
# Hello, pypack!
```

## Runtime behavior

- **First run:** The stub extracts the Python runtime to `~/.cache/pypack/<hash>/`. This takes a few seconds.
- **Subsequent runs:** The cached runtime is reused instantly. Startup time is ~100ms.
- **Cache key:** Based on a SHA-256 hash of the runtime blob — different Python versions or builds get separate cache entries.
- **Isolation:** `PYTHONNOUSERSITE=1` is set to prevent interference from the host's site-packages.

## Stub compilation notes

The C stub can be compiled manually for different platforms:

| Platform | Command | Notes |
|----------|---------|-------|
| Linux x86_64 | `cc -O2 -s -o stub stub.c` | `-s` strips symbols |
| Linux x86_64 (static) | `cc -O2 -s -static -o stub stub.c` | No glibc dep |
| macOS arm64 | `cc -O2 -o stub stub.c` | macOS doesn't support `-static` |
| macOS universal | `cc -O2 -arch arm64 -arch x86_64 -o stub stub.c` | Fat binary |

The stub shells out to `zstd`, `tar`, and `sha256sum`/`shasum` at runtime. These are
present on essentially all Linux distros and macOS. See the roadmap for embedding zstd
directly to remove this dependency.

## Comparison with existing tools

| | pypack | PyInstaller | PyOxidizer | Nuitka |
|---|---|---|---|---|
| **Build deps** | uv + cc + zstd | Python + hooks | Rust toolchain | C compiler + Python |
| **Complexity** | ~500 lines | Very large | Large | Very large (compiler) |
| **Runtime source** | python-build-standalone (via uv) | Host Python (frozen) | Embedded in Rust | Host Python (compiled to C) |
| **Deterministic** | ✅ pinned PBS version | ❌ uses host | ✅ | ❌ uses host |
| **Mental model** | "zip on a runtime" | "freeze the world" | "embed in Rust" | "compile to C" |
| **Build speed** | ~10s (mostly compression) | 30–120s | 60–300s | 60–600s |

pypack trades features for simplicity. The entire tool is ~200 lines of C + ~300 lines
of Python, with no framework, no hooks system, no hidden imports database, and no
compilation step for user code. It just glues a known-good Python runtime to a zip of
your code using mechanisms (`zipimport`, zip-tail-append) that Python already has built in.

## Running the tests

```bash
uv run pytest tests/ -v
```

## Supported platforms

| Platform | Status |
|----------|--------|
| Linux x86_64 | ✅ |
| Linux aarch64 | ✅ |
| macOS x86_64 | ✅ |
| macOS arm64 | ✅ |
| Windows | ❌ (planned) |

## Limitations (v1)

- **Pure Python only.** Packages with C extensions (numpy, pandas, etc.) are rejected at build time. Native extension support is planned for v2.
- **`requirements.txt` only.** Other dependency formats (`pyproject.toml`, `setup.py`, etc.) are not yet supported.
- **No cross-compilation.** The binary is built for the current platform only.
- **Target needs zstd + tar.** The first run extracts the runtime using these tools.
- **~15 MB minimum size.** The compressed Python runtime is the floor.

## Roadmap

| Version | Feature | Description |
|---------|---------|-------------|
| **v2** | **`pyproject.toml` deps** | Read dependencies from `[project.dependencies]` via `uv pip install .` or `uv export` |
| **v3** | **`setup.py` / `setup.cfg` deps** | Support legacy packaging formats via `uv pip install .` |
| **v4** | **Native extensions** | Extract `.so`/`.dylib` to cache at first run; use target Python for ABI-correct wheels |
| **v5** | **Stdlib tree-shaking** | Analyze imports → strip unused stdlib modules (`tkinter`, `test`, `idlelib`, etc.) to cut ~5–10 MB |
| **v6** | **Layered caching** | Hash runtime/deps/app independently — skip re-extracting unchanged layers |
| **v7** | **Windows** | Port stub to Win32 (`GetModuleFileName`, `CreateProcess`), produce `.exe` |
| **v8** | **Cross-compilation** | `--target linux-x86_64` from macOS, using uv to fetch the target PBS release |
| **v9** | **Embedded zstd** | Statically link zstd (~100 KB) into the C stub → zero runtime deps on the target |
| **v10** | **`.pyc` pre-compilation** | `compileall` at build time, ship only bytecode for faster startup |
| **v11** | **`memfd_create`** | Load interpreter from memory on Linux — no disk extraction, instant first run |

## License

MIT
