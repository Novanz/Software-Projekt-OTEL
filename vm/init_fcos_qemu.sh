#!/usr/bin/env bash
set -euo pipefail

### CONFIGURABLE SETTINGS #######################################################

# Fedora CoreOS stream and architecture
STREAM="stable" # stable | testing | next
ARCH="x86_64"

# libvirt / VM settings
VM_NAME="fcos-libvirt"
LIBVIRT_URI="qemu:///system"

# libvirt networks
NET1="default" # NAT / outbound internet

# Ignition / Butane files
BU_FILE="fcos.bu"
IGN_FILE="fcos.ign"
IGN_DIR="/var/lib/libvirt/boot"

# Resources
CPUS=4
# RAM_MB=16384
RAM_MB=12288
DISK_GB=40

# Storage
IMAGE_DIR="/var/lib/libvirt/images"

###############################################################################

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

command -v virt-install >/dev/null 2>&1 || {
    echo "virt-install is required"
    exit 1
}

command -v coreos-installer >/dev/null 2>&1 || {
    echo "coreos-installer is required"
    exit 1
}

command -v restorecon >/dev/null 2>&1 || {
    echo "restorecon is required"
    exit 1
}

sudo mkdir -p "$IMAGE_DIR" "$IGN_DIR"
echo "==> Looking for an existing Fedora CoreOS QEMU image..."
IMAGE="$(sudo find "$IMAGE_DIR" -maxdepth 1 -type f -name "fedora-coreos-*-qemu.${ARCH}.qcow2" | sort | tail -n 1)"

if [ -n "$IMAGE" ]; then
    echo " Reusing existing image: $IMAGE"
else
    echo "==> No local FCOS image found, downloading latest for stream '$STREAM'..."

    sudo coreos-installer download \
        -s "$STREAM" \
        -p qemu \
        -f qcow2.xz \
        --decompress \
        -C "$IMAGE_DIR"

    echo "==> Locating downloaded qcow2 image..."
    IMAGE="$(sudo find "$IMAGE_DIR" -maxdepth 1 -type f -name "fedora-coreos-*-qemu.${ARCH}.qcow2" | sort | tail -n 1)"
fi

if [ -z "$IMAGE" ]; then
    echo "Could not find downloaded FCOS qcow2 image in $IMAGE_DIR"
    exit 1
fi

echo " Image: $IMAGE"

echo "==> Generating Ignition config from Butane..."
butane --pretty --strict --files-dir . "$BU_FILE" >"$IGN_FILE"

IGN_PATH="${IGN_DIR}/${VM_NAME}.ign"
sudo install -m 0644 "$IGN_FILE" "$IGN_PATH"

echo "==> Restoring SELinux labels for libvirt paths..."
sudo restorecon -Rv "$IMAGE_DIR" "$IGN_DIR"

echo "==> Removing any existing VM with the same name..."
if virsh --connect "$LIBVIRT_URI" dominfo "$VM_NAME" >/dev/null 2>&1; then
    virsh --connect "$LIBVIRT_URI" destroy "$VM_NAME" >/dev/null 2>&1 || true
    virsh --connect "$LIBVIRT_URI" undefine "$VM_NAME" --nvram >/dev/null 2>&1 || true
fi

echo "==> Creating and starting FCOS VM with virt-install..."
virt-install \
    --connect "$LIBVIRT_URI" \
    --name "$VM_NAME" \
    --vcpus "$CPUS" \
    --memory "$RAM_MB" \
    --os-variant "fedora-coreos-$STREAM" \
    --import \
    --graphics none \
    --network network="$NET1" \
    --disk "size=${DISK_GB},backing_store=${IMAGE}" \
    --qemu-commandline="-fw_cfg name=opt/com.coreos/config,file=${IGN_PATH}"

echo "All done."
echo "VM '$VM_NAME' is booting under libvirt."
