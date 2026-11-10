#!/usr/bin/env bash
# Download the organoid image data (https://doi.org/10.25532/OPARA-598) from
# OPARA, verify the MD5 checksums and unpack it into ./data.
#
# Usage:
#   ./download_data.sh                 # all four sets (~433 MB)
#   ./download_data.sh train-100 test-100 test-30   # only the given sets
#
# Afterwards the image sets are also linked into code-pqap/data/images/image_sets
# under the names expected by the PQAP code.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-${REPO_DIR}/data}"
PQAP_IMAGE_SETS_DIR="${REPO_DIR}/code-pqap/data/images/image_sets"
OPARA_API="https://opara.zih.tu-dresden.de/server/api/core/bitstreams"

# name  bitstream-uuid  md5
DATASETS=(
    "train-100       1d027870-e617-4d9b-b05c-13c2c98e808b 5bd1eb4843a962788800025174c2d004"
    "test-100        70a945bc-710c-4461-8ec4-b23b1bd24b2d 3e30ae5acd7b4c4a7a2e40cdbe3e2c7e"
    "test-30         8f6d86b9-d2de-474c-a94e-d71759cad38d 3a53f731edeb73252dbef2d056bd8c7e"
    "test-unlabeled  4d66d0a3-e2c7-40ed-bc78-a615a704c013 ec775c89a3abcdfaf02a985f9f851fff"
)

for tool in curl unzip md5sum; do
    command -v "$tool" >/dev/null || { echo "Error: '$tool' is required." >&2; exit 1; }
done

requested=("$@")
is_requested() {
    [ ${#requested[@]} -eq 0 ] && return 0
    local r
    for r in "${requested[@]}"; do [ "$r" = "$1" ] && return 0; done
    return 1
}

mkdir -p "$DATA_DIR"

for entry in "${DATASETS[@]}"; do
    read -r name uuid md5 <<<"$entry"
    is_requested "$name" || continue

    if [ -d "${DATA_DIR}/${name}" ]; then
        echo "[${name}] already present in ${DATA_DIR}/${name}, skipping."
        continue
    fi

    archive="${DATA_DIR}/${name}.zip"
    if [ ! -f "$archive" ] || ! echo "${md5}  ${archive}" | md5sum --check --status; then
        echo "[${name}] downloading ..."
        curl --fail --location --retry 3 --output "$archive" "${OPARA_API}/${uuid}/content"
    fi

    echo "[${name}] verifying checksum ..."
    if ! echo "${md5}  ${archive}" | md5sum --check --status; then
        echo "Error: checksum mismatch for ${archive}. Delete it and run the script again." >&2
        exit 1
    fi

    echo "[${name}] unpacking ..."
    unzip -q -o "$archive" -d "$DATA_DIR"
    rm "$archive"
done

# Link the image sets into the PQAP data directory under the names used there.
link_set() {
    local source="$1" target="$2"
    [ -d "${DATA_DIR}/${source}" ] || return 0
    [ -e "${PQAP_IMAGE_SETS_DIR}/${target}" ] && return 0
    ln -s "$(realpath "${DATA_DIR}/${source}")" "${PQAP_IMAGE_SETS_DIR}/${target}"
    echo "Linked ${PQAP_IMAGE_SETS_DIR}/${target} -> ${DATA_DIR}/${source}"
}
mkdir -p "$PQAP_IMAGE_SETS_DIR"
link_set train-100 organoid_dataset_100_split_1
link_set test-100 organoid_dataset_100_split_2
link_set test-30 organoid_dataset_unseen_classes

# Test-100 and Test-30 together.
combined="${PQAP_IMAGE_SETS_DIR}/organoid_dataset_with_unseen_classes"
if [ -d "${DATA_DIR}/test-100" ] && [ -d "${DATA_DIR}/test-30" ] && [ ! -e "$combined" ]; then
    mkdir -p "$combined"
    for f in "${DATA_DIR}"/test-100/* "${DATA_DIR}"/test-30/*; do
        ln -s "$(realpath "$f")" "${combined}/$(basename "$f")"
    done
    echo "Created ${combined} (Test-100 + Test-30)"
fi

echo "Done. Data is in ${DATA_DIR}."
