#!/usr/bin/env bash
set -euo pipefail

# Build config.json from --<key> <value> flags, then run ./main with it.
#
# Every "--foo bar" pair is written into config.json as "foo": bar, with the
# value type auto-detected (true/false -> boolean, numeric -> number, else
# string). This mirrors the keys read by ./main, e.g.:
#
#   ./main_cli.sh \
#     --freesurfer /input/freesurfer/output \
#     --maps "polarAngle,eccentricity,pRFsize" \
#     --dataset hcp \
#     --subject_id sub-01
#
# Pass --dry-run to only generate and print config.json without executing.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_OUT="config.json"
CONFIG_OUT_SET=0
OUTDIR=""
DRY_RUN=0
NO_CONTAINER=0
declare -a JSON_FRAGMENTS=()

usage() {
  cat <<'EOF'
Usage: ./main_cli.sh --<key> <value> [--<key> <value> ...] [--config PATH] [--dry-run]

Any --<key> <value> pair becomes a "<key>": <value> entry in config.json.
A bare --<key> (no value, or followed by another --flag) is treated as
--<key> true, e.g. `--use_gpu` sets it to true.
By default, ./main is then executed with the generated config.

Keys (see README.md for details):
  --freesurfer     path to a single subject's FreeSurfer output directory [required]
  --maps           comma-separated maps, e.g. "polarAngle,eccentricity,pRFsize"
                   (default: polarAngle,eccentricity,pRFsize)
  --dataset        dataset name (default: brainlife)
  --subject_id     synthetic subject id used internally (default: subject)
  --fast           yes/no fast midthickness generation (default: yes)
  --use_gpu        pass --nv to singularity (requires a GPU-enabled host)
  --container_version  vnmd/deepretinotopy_<version> tag to pull (default: 1.0.18;
                        1.0.19 currently fails to build via Singularity)
  --polar_angle_benson  fold polar angle from 0-360° to 0-180° (Benson/neuropythy
                        convention; hemisphere encodes left vs right visual field)

Options:
  --config PATH     write generated config to PATH instead of ./config.json
                    (default becomes <OUTDIR>/config.json when --output_dir is set)
  --output_dir DIR  CLI-only: run ./main from inside DIR, so output/,
                    prf_surfaces/ and product.json land there instead of the
                    repo root.
  --dry-run         only generate and print config.json; do not run ./main
  --no-container    skip Singularity entirely and run deepRetinotopy directly
                    (requires a local install of the toolbox on PATH)
  -h, --help        show this help
EOF
}

to_json_fragment() {
  local key="$1" value="$2"
  # Resolve file/dir path values to absolute paths so they keep resolving
  # correctly even if we cd into --output_dir before running ./main.
  if [[ -e "$value" ]]; then
    value="$(realpath "$value")"
  fi
  jq -n --arg k "$key" --arg v "$value" '
    if ($v == "true") then {($k): true}
    elif ($v == "false") then {($k): false}
    elif ($v | test("^-?[0-9]+$")) then {($k): ($v | tonumber)}
    elif ($v | test("^-?[0-9]*\\.[0-9]+$")) then {($k): ($v | tonumber)}
    else {($k): $v}
    end
  '
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --config)
      [[ $# -ge 2 ]] || { echo "ERROR: missing value for --config" >&2; exit 1; }
      CONFIG_OUT="$2"
      CONFIG_OUT_SET=1
      shift 2
      ;;
    --output_dir)
      [[ $# -ge 2 ]] || { echo "ERROR: missing value for --output_dir" >&2; exit 1; }
      OUTDIR="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --no-container)
      NO_CONTAINER=1
      shift
      ;;
    --*=*)
      key="${1#--}"
      value="${key#*=}"
      key="${key%%=*}"
      JSON_FRAGMENTS+=("$(to_json_fragment "$key" "$value")")
      shift
      ;;
    --*)
      key="${1#--}"
      # Bare boolean flag (no value, or immediately followed by another
      # --flag) defaults to true, e.g. `--use_gpu --fast`.
      if [[ $# -ge 2 && "$2" != --* ]]; then
        value="$2"
        shift 2
      else
        value="true"
        shift
      fi
      JSON_FRAGMENTS+=("$(to_json_fragment "$key" "$value")")
      ;;
    *)
      echo "ERROR: unrecognized argument '$1'" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ${#JSON_FRAGMENTS[@]} -eq 0 ]]; then
  echo "ERROR: no --<key> <value> config options provided" >&2
  usage
  exit 1
fi

if [[ -n "${OUTDIR}" ]]; then
  mkdir -p "${OUTDIR}"
  OUTDIR="$(cd "${OUTDIR}" && pwd)"
  if [[ "${CONFIG_OUT_SET}" -eq 0 ]]; then
    CONFIG_OUT="${OUTDIR}/config.json"
  fi
fi

printf '%s\n' "${JSON_FRAGMENTS[@]}" | jq -s 'add' > "${CONFIG_OUT}"
CONFIG_OUT="$(realpath "${CONFIG_OUT}")"

echo "Wrote ${CONFIG_OUT}:"
cat "${CONFIG_OUT}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo
  echo "Dry run: not executing ./main."
  exit 0
fi

if [[ -n "${OUTDIR}" ]]; then
  echo
  echo "Running ./main in ${OUTDIR} (output/, prf_surfaces/, product.json will land there) ..."
  ( cd "${OUTDIR}" && CONFIG="${CONFIG_OUT}" NO_CONTAINER="${NO_CONTAINER}" bash "${SCRIPT_DIR}/main" )
else
  echo
  echo "Running ./main with CONFIG=${CONFIG_OUT} ..."
  CONFIG="${CONFIG_OUT}" NO_CONTAINER="${NO_CONTAINER}" bash "${SCRIPT_DIR}/main"
fi
