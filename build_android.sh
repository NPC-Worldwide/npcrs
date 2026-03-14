#!/bin/bash
# Build npcrs for Android (arm64-v8a)
#
# Prerequisites:
#   rustup target add aarch64-linux-android
#   Install Android NDK (via Android Studio SDK Manager)
#
# Usage:
#   ./build_android.sh [--release]
#
# Output:
#   target/aarch64-linux-android/{debug|release}/libnpcrs.so
#   Also copies to eazy-phone/android/app/src/main/jniLibs/arm64-v8a/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EAZY_PHONE_DIR="$SCRIPT_DIR/../../mindful/eazy-phone"
PROFILE="debug"

if [[ "${1:-}" == "--release" ]]; then
    PROFILE="release"
fi

# Find Android NDK
if [ -z "${ANDROID_NDK_HOME:-}" ]; then
    # Common locations
    for ndk_dir in \
        "$HOME/Android/Sdk/ndk/"* \
        "$HOME/Library/Android/sdk/ndk/"* \
        "/usr/local/lib/android/sdk/ndk/"*; do
        if [ -d "$ndk_dir" ]; then
            export ANDROID_NDK_HOME="$ndk_dir"
            break
        fi
    done
fi

if [ -z "${ANDROID_NDK_HOME:-}" ]; then
    echo "ERROR: Android NDK not found."
    echo "Set ANDROID_NDK_HOME or install via Android Studio SDK Manager."
    exit 1
fi

echo "Using NDK: $ANDROID_NDK_HOME"
echo "Profile: $PROFILE"

# Set up the linker for Android cross-compilation
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android24-clang"
export CC_aarch64_linux_android="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android24-clang"
export AR_aarch64_linux_android="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-ar"

# Build
cd "$SCRIPT_DIR"

CARGO_ARGS="--lib --target aarch64-linux-android"
if [ "$PROFILE" == "release" ]; then
    CARGO_ARGS="$CARGO_ARGS --release"
fi

echo "Building npcrs for aarch64-linux-android..."
cargo build $CARGO_ARGS

# Copy .so to eazy-phone jniLibs
JNILIB_DIR="$EAZY_PHONE_DIR/android/app/src/main/jniLibs/arm64-v8a"
mkdir -p "$JNILIB_DIR"

SO_PATH="$SCRIPT_DIR/target/aarch64-linux-android/$PROFILE/libnpcrs.so"
if [ -f "$SO_PATH" ]; then
    cp "$SO_PATH" "$JNILIB_DIR/libnpcrs.so"
    echo "Copied to $JNILIB_DIR/libnpcrs.so"
    ls -lh "$JNILIB_DIR/libnpcrs.so"
else
    echo "ERROR: $SO_PATH not found"
    exit 1
fi

echo "Done! Build npcrs for Android arm64-v8a ($PROFILE)"
