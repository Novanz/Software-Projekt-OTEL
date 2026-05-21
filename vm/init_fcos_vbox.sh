#!/usr/bin/env bash
set -euo pipefail

### CONFIGURABLE SETTINGS #######################################################

# Fedora CoreOS stream and architecture
STREAM="stable" # e.g. stable | testing | next
ARCH="x86_64"

# VirtualBox VM settings
BASEFOLDER="$HOME/VirtualBox VMs"
VM_NAME="fcos-ova"
HOSTONLY_IF="vboxnet0" # existing host-only network name

# Ignition / Butane files
BU_FILE="fcos.bu"   # your Butane config
IGN_FILE="fcos.ign" # generated Ignition config

# Resources
CPUS=4
RAM_MB=16384
DISK_GB=40 # desired disk size in GB
DISK_MB=$((DISK_GB * 1024))

###############################################################################

# Requirements: curl, jq, butane, VBoxManage
command -v curl >/dev/null 2>&1 || {
    echo "curl is required"
    exit 1
}
command -v jq >/dev/null 2>&1 || {
    echo "jq is required"
    exit 1
}
command -v butane >/dev/null 2>&1 || {
    echo "butane is required"
    exit 1
}
command -v VBoxManage >/dev/null 2>&1 || {
    echo "VBoxManage is required"
    exit 1
}

echo "==> Fetching Fedora CoreOS stream metadata for '$STREAM'..."
META_URL="https://builds.coreos.fedoraproject.org/streams/${STREAM}.json"
META_JSON="$(curl -fsSL "$META_URL")" # Stream metadata is the canonical source for artifacts. [web:99]

echo "==> Resolving latest VirtualBox OVA URL for arch=$ARCH..."
OVA_URL="$(printf '%s' "$META_JSON" |
    jq -r --arg arch "$ARCH" '.architectures[$arch].artifacts.virtualbox.formats.ova.disk.location')"

if [ -z "$OVA_URL" ] || [ "$OVA_URL" = "null" ]; then
    echo "Could not find VirtualBox OVA for arch=$ARCH in stream=$STREAM."
    exit 1
fi

OVA="$(basename "$OVA_URL")"
echo "    OVA URL: $OVA_URL"
echo "    OVA file: $OVA"

echo "==> Downloading OVA (if not already present)..."
if [ -f "$OVA" ]; then
    echo "    $OVA already exists, skipping download."
else
    curl -fSL -o "$OVA" "$OVA_URL"
fi

echo "==> Generating Ignition config from Butane..."
butane --pretty --strict --files-dir . "$BU_FILE" >"$IGN_FILE"

echo "==> Importing OVA as VM '$VM_NAME'..."
VBoxManage import "$OVA" --vsys 0 --vmname "$VM_NAME" --basefolder "$BASEFOLDER"

echo "==> Configuring VM (CPUs, RAM, NICs)..."
VBoxManage modifyvm "$VM_NAME" \
    --memory "$RAM_MB" \
    --cpus "$CPUS" \
    --nic1 nat \
    --nic2 hostonly \
    --hostonlyadapter2 "$HOSTONLY_IF"

echo "==> Detecting attached VDI path..."
DISK_PATH="$(VBoxManage showvminfo "$VM_NAME" --machinereadable |
    sed -n 's/^".*-[0-9]-[0-9]"="\([^"]*\)"$/\1/p' |
    head -n 1)"

if [ -z "$DISK_PATH" ]; then
    echo "Could not auto-detect VDI path. Run:"
    echo "  VBoxManage showvminfo \"$VM_NAME\""
    echo "and update the script with the correct VDI path."
    exit 1
fi

echo "    VDI path: $DISK_PATH"

echo "==> Resizing disk to ${DISK_GB}GB (${DISK_MB}MB)..."
VBoxManage modifymedium disk "$DISK_PATH" --resize "$DISK_MB" # Resize in MB. [web:94]

echo "==> Injecting Ignition via /Ignition/Config guest property..."
VBoxManage guestproperty set "$VM_NAME" /Ignition/Config "$(cat "$IGN_FILE")" # Ignition provider for VirtualBox. [web:9][web:20]

echo "==> Verifying Ignition property..."
VBoxManage guestproperty get "$VM_NAME" /Ignition/Config

echo "==> Starting VM headless..."
VBoxManage startvm "$VM_NAME" --type headless

echo "All done."
echo "VM '$VM_NAME' is booting. Use the host-only NIC ($HOSTONLY_IF) IP to SSH as 'core'."
