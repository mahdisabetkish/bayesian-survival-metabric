#!/usr/bin/env bash
#
# Fetch the METABRIC breast cancer study (Curtis et al. 2012; Pereira et al. 2016)
# from the cBioPortal Data Hub and unpack it into this directory.
#
# Produces:
#   data/brca_metabric/data_clinical_patient.txt
#   data/brca_metabric/data_clinical_sample.txt
#   data/brca_metabric/data_mrna_illumina_microarray.txt
#
set -euo pipefail

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
URL="https://cbioportal-datahub.s3.amazonaws.com/brca_metabric.tar.gz"
ARCHIVE="${DATA_DIR}/brca_metabric.tar.gz"
TARGET="${DATA_DIR}/brca_metabric"

if [[ -f "${TARGET}/data_mrna_illumina_microarray.txt" ]]; then
    echo "[download] METABRIC already unpacked at ${TARGET} — nothing to do."
    exit 0
fi

if [[ ! -f "${ARCHIVE}" ]]; then
    echo "[download] Fetching METABRIC from cBioPortal (~500 MB, this takes a few minutes)..."
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 3 --retry-delay 5 -o "${ARCHIVE}.part" "${URL}"
    elif command -v wget >/dev/null 2>&1; then
        wget -c -O "${ARCHIVE}.part" "${URL}"
    else
        echo "[download] ERROR: neither curl nor wget is available." >&2
        exit 1
    fi
    mv "${ARCHIVE}.part" "${ARCHIVE}"
else
    echo "[download] Using cached archive ${ARCHIVE}"
fi

echo "[download] Unpacking..."
tar -xzf "${ARCHIVE}" -C "${DATA_DIR}"

# The expression matrix is the only file we strictly need at full size; the rest
# of the study (CNA, methylation, mutations) is left in place for future work.
if [[ ! -f "${TARGET}/data_mrna_illumina_microarray.txt" ]]; then
    echo "[download] ERROR: expected expression matrix missing after extraction." >&2
    exit 1
fi

echo "[download] Done. Study unpacked to ${TARGET}"
