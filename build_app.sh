#!/usr/bin/env bash
# Builds or updates the self-contained De-limiter.app in this directory.
#
# - Fast incremental build (default when Contents/Resources/python already exists):
#     ./build_app.sh
#   Recompiles DelimiterApp.swift, syncs delimiter_tool.py + De-limiter/, and prunes unused runtime bloat.
#
# - Full runtime rebuild (bundles a fresh standalone Python 3.11 + packages from requirements-app.txt):
#     ./build_app.sh --full
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$SCRIPT_DIR/De-limiter.app"
CONTENTS_DIR="$APP_DIR/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"
PY_VERSION=3.11
export MACOSX_DEPLOYMENT_TARGET=14.0

FULL_REBUILD=0
MAKE_ZIP=0
for arg in "$@"; do
    case "$arg" in
        --full) FULL_REBUILD=1 ;;
        --zip)  MAKE_ZIP=1 ;;
    esac
done
if [[ ! -x "$RESOURCES_DIR/python/bin/python$PY_VERSION" ]]; then
    FULL_REBUILD=1
fi

mkdir -p "$MACOS_DIR" "$RESOURCES_DIR"

cat > "$CONTENTS_DIR/Info.plist" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>De-limiter</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundleIdentifier</key>
    <string>com.antigravity.delimiter.studio</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>De-limiter Studio</string>
    <key>CFBundleDisplayName</key>
    <string>De-limiter Studio</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.1.0</string>
    <key>CFBundleVersion</key>
    <string>2</string>
    <key>LSMinimumSystemVersion</key>
    <string>14.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSPrincipalClass</key>
    <string>NSApplication</string>
</dict>
</plist>
EOF

printf "APPL????" > "$CONTENTS_DIR/PkgInfo"

PY="$RESOURCES_DIR/python/bin/python$PY_VERSION"
SITE="$RESOURCES_DIR/python/lib/python$PY_VERSION/site-packages"

if [[ "$FULL_REBUILD" -eq 1 ]]; then
    if command -v uv >/dev/null 2>&1; then
        UV="$(command -v uv)"
    elif [[ -x "$HOME/.local/bin/uv" ]]; then
        UV="$HOME/.local/bin/uv"
    else
        echo "uv not found. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
        exit 1
    fi

    echo "==> Bundling standalone Python $PY_VERSION into De-limiter.app..."
    rm -rf "$RESOURCES_DIR/python"
    "$UV" python install --no-bin "$PY_VERSION"
    PY_BIN="$("$UV" python find --managed-python "$PY_VERSION")"
    PY_HOME="$(cd "$(dirname "$(realpath "$PY_BIN")")/.." && pwd)"
    ditto "$PY_HOME" "$RESOURCES_DIR/python"

    echo "==> Installing Python packages from requirements-app.txt..."
    "$UV" pip sync \
        --python "$PY" \
        --break-system-packages \
        --python-platform aarch64-apple-darwin \
        "$SCRIPT_DIR/requirements-app.txt"
fi

echo "==> Pruning unused packages and build artifacts from bundled Python..."
"$PY" - "$SITE" <<'PYEOF'
import os, shutil, sys

site = sys.argv[1]
keep_prefixes = (
    "_cffi_backend",
    "_soundfile",
    "asteroid_filterbanks",
    "cffi",
    "dotmap",
    "einops",
    "filelock",
    "fsspec",
    "numpy",
    "packaging",
    "pycparser",
    "pyloudnorm",
    "scipy",
    "soundfile",
    "soxr",
    "torch-",
    "tqdm",
    "typing_extensions",
)
keep_exact = {"torch", "torchgen"}
if os.path.isdir(site):
    for entry in os.listdir(site):
        if entry not in keep_exact and not entry.startswith(keep_prefixes):
            full = os.path.join(site, entry)
            if os.path.isdir(full) and not os.path.islink(full):
                shutil.rmtree(full)
            else:
                os.remove(full)
PYEOF

rm -rf \
    "$SITE/torch/include" \
    "$SITE/torch/bin/protoc"* \
    "$SITE/torch/_inductor" \
    "$RESOURCES_DIR/python/include" \
    "$RESOURCES_DIR/python/share" \
    "$RESOURCES_DIR/python/lib/tcl"* \
    "$RESOURCES_DIR/python/lib/tk"* \
    "$RESOURCES_DIR/python/lib/itcl"* \
    "$RESOURCES_DIR/python/lib/thread"* \
    "$RESOURCES_DIR/python/lib/libtcl"* \
    "$RESOURCES_DIR/python/lib/libtk"* \
    "$RESOURCES_DIR/python/lib/pkgconfig" \
    "$RESOURCES_DIR/python/lib/python$PY_VERSION/ensurepip" \
    "$RESOURCES_DIR/python/lib/python$PY_VERSION/idlelib" \
    "$RESOURCES_DIR/python/lib/python$PY_VERSION/tkinter" \
    "$RESOURCES_DIR/python/lib/python$PY_VERSION/turtle"* \
    "$RESOURCES_DIR/python/lib/python$PY_VERSION/pydoc_data" \
    "$RESOURCES_DIR/python/lib/python$PY_VERSION/lib2to3" \
    "$RESOURCES_DIR/python/lib/python$PY_VERSION/distutils" \
    "$RESOURCES_DIR/python/lib/python$PY_VERSION/config-"*

find "$SITE" -type d -name tests -prune -exec rm -rf {} +
find "$RESOURCES_DIR/python/bin" -mindepth 1 ! -name python ! -name python3 ! -name "python$PY_VERSION" -delete

if [[ -d "$SCRIPT_DIR/De-limiter" ]]; then
    rsync -a --delete --delete-excluded \
        --exclude '/.venv' --exclude '/.git' --exclude '/input_data' --exclude '/output' \
        --exclude '__pycache__' --exclude '.DS_Store' \
        "$SCRIPT_DIR/De-limiter/" "$RESOURCES_DIR/De-limiter/"
fi

rm -f "$RESOURCES_DIR/audio_tool.py"
cp "$SCRIPT_DIR/delimiter_tool.py" "$RESOURCES_DIR/delimiter_tool.py"

find "$RESOURCES_DIR" -type d -name __pycache__ -prune -exec rm -rf {} +
"$PY" -I -m compileall -q -j 0 --invalidation-mode unchecked-hash \
    "$RESOURCES_DIR/python/lib" "$RESOURCES_DIR/De-limiter" >/dev/null 2>&1 || true

echo "==> Compiling De-limiter.app..."
/usr/bin/swiftc \
    -O \
    -target arm64-apple-macos14.0 \
    -parse-as-library \
    -module-cache-path /tmp/swift-module-cache \
    -framework SwiftUI \
    -framework AppKit \
    -framework AVFoundation \
    -framework UniformTypeIdentifiers \
    "$SCRIPT_DIR/DelimiterApp.swift" \
    -o "$MACOS_DIR/De-limiter"

chmod +x "$MACOS_DIR/De-limiter"
touch "$APP_DIR"
echo "Successfully built self-contained app: $APP_DIR"
du -sh "$APP_DIR"

if [[ "$MAKE_ZIP" -eq 1 ]]; then
    ZIP_PATH="$SCRIPT_DIR/De-limiter-Studio-v1.1.0-macOS-arm64.zip"
    echo "==> Creating release archive: $ZIP_PATH..."
    rm -f "$ZIP_PATH"
    ditto -c -k --keepParent "$APP_DIR" "$ZIP_PATH"
    du -sh "$ZIP_PATH"
fi
