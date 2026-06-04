#!/usr/bin/env bash
# Build a macOS-arm64 wheel for omikuji 0.5.1.
#
# Why this exists: omikuji has no usable macOS-arm64 distribution.
#   - The PyPI cp312 "universal2" wheel ships an x86_64-only .so (won't dlopen on
#     Apple Silicon), and there is no cp313 wheel at all.
#   - The sdist's c-api/Cargo.lock pins `time 0.3.30`, which fails to compile on
#     Rust >= 1.80 (rustc error E0282 in time's format_description parser).
#
# Fix: bump the `time` crate in BOTH lockfiles to a patched 0.3.x, then build the
# wheel with the project's arm64 interpreter (milksnake follows the interpreter
# arch — building with an x86_64 python silently yields an x86_64 .so).
#
# The resulting wheel is committed to wheels/ and referenced from pyproject.toml
# via [tool.uv.sources] (macOS-arm64 only). Re-run this if the omikuji pin changes.
#
# Prereqs: Rust (`brew install rust`), Xcode Command Line Tools (clang), uv.
set -euo pipefail

VERSION="0.5.1"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PY="$REPO_ROOT/.venv/bin/python"   # must be the arm64 project interpreter
WORK="$(mktemp -d)"
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.cargo/bin:$PATH"

echo ">> downloading omikuji $VERSION sdist"
URL=$(python3 -c "import urllib.request,json; d=json.load(urllib.request.urlopen('https://pypi.org/pypi/omikuji/$VERSION/json')); print([f['url'] for f in d['urls'] if f['filename'].endswith('.tar.gz')][0])")
curl -sL "$URL" -o "$WORK/omikuji.tar.gz"
tar -xzf "$WORK/omikuji.tar.gz" -C "$WORK"
SRC="$WORK/omikuji-$VERSION"

echo ">> bumping time crate in both lockfiles (root + c-api)"
( cd "$SRC"        && cargo update -p time --precise 0.3.41 )
( cd "$SRC/c-api"  && cargo update -p time --precise 0.3.41 )

echo ">> building wheel with arm64 interpreter ($VENV_PY)"
"$REPO_ROOT/.venv/bin/python" -m pip --version >/dev/null 2>&1 || \
  uv pip install --python "$VENV_PY" milksnake cffi setuptools wheel
( cd "$SRC" && ARCHFLAGS="-arch arm64" _PYTHON_HOST_PLATFORM="macosx-11.0-arm64" \
    "$VENV_PY" setup.py bdist_wheel )

WHEEL=$(ls "$SRC"/dist/*.whl)
echo ">> verifying arm64 binary"
unzip -p "$WHEEL" 'omikuji/_libomikuji__lib.so' > "$WORK/check.so"
file "$WORK/check.so" | grep -q arm64 || { echo "ERROR: built .so is not arm64"; exit 1; }

mkdir -p "$REPO_ROOT/wheels"
cp "$WHEEL" "$REPO_ROOT/wheels/"
echo ">> done: $(basename "$WHEEL") -> wheels/"
