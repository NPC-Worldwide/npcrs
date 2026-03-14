#!/bin/bash
# Build npcrs for Linux desktop and install for Flutter testing.
#
# Usage:
#   ./build_linux.sh [--release]
#
# This copies libnpcrs.so to:
#   1. eazy-phone/linux/lib/ (for Flutter Linux desktop)
#   2. /usr/local/lib/ or ~/.local/lib/ (for system-wide access)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EAZY_PHONE_DIR="$SCRIPT_DIR/../../mindful/eazy-phone"
PROFILE="debug"

if [[ "${1:-}" == "--release" ]]; then
    PROFILE="release"
fi

source "$HOME/.cargo/env"

echo "Building npcrs for Linux ($PROFILE)..."
if [ "$PROFILE" == "release" ]; then
    cargo build --release
else
    cargo build
fi

SO_PATH="$SCRIPT_DIR/target/$PROFILE/libnpcrs.so"

if [ ! -f "$SO_PATH" ]; then
    echo "ERROR: $SO_PATH not found"
    exit 1
fi

echo "Built: $(ls -lh "$SO_PATH" | awk '{print $5}')"

# Copy to eazy-phone for Flutter Linux desktop
FLUTTER_LIB_DIR="$EAZY_PHONE_DIR/linux/lib"
mkdir -p "$FLUTTER_LIB_DIR"
cp "$SO_PATH" "$FLUTTER_LIB_DIR/libnpcrs.so"
echo "Installed to $FLUTTER_LIB_DIR/libnpcrs.so"

# Also copy to home .local for LD_LIBRARY_PATH access
LOCAL_LIB="$HOME/.local/lib"
mkdir -p "$LOCAL_LIB"
cp "$SO_PATH" "$LOCAL_LIB/libnpcrs.so"
echo "Installed to $LOCAL_LIB/libnpcrs.so"

echo ""
echo "Done! To use with Flutter Linux:"
echo "  export LD_LIBRARY_PATH=$LOCAL_LIB:\$LD_LIBRARY_PATH"
echo "  cd $EAZY_PHONE_DIR && flutter run -d linux"
