#!/bin/bash
# One GPU + configurable CPUs. All account-specific paths come from the run config.
set -euo pipefail
LF_PACKAGE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
LF_RUN=$(cd -- "${1:?Usage: job.sh RUN_DIRECTORY}" && pwd -P)
LF_MODE=${2:-search}
mapfile -t LF_SETTINGS < <(/usr/bin/python3 - "$LF_RUN/config.json" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]));s=c['site']
for key in ('cpu_container','cpu_python','netmhcpan','af3_image','af3_model_dir'):
    value=s[key]
    if '\n' in value or '\r' in value:raise ValueError('Invalid newline in path')
    print(value)
print(s['cpus'])
PY
)
[[ ${#LF_SETTINGS[@]} == 6 ]] || { echo 'Cannot read dependency settings'; exit 2; }
LF_CPU_IMAGE=${LF_SETTINGS[0]}
LF_CPU_PYTHON=${LF_SETTINGS[1]}
LF_NETMHCPAN=${LF_SETTINGS[2]}
export AF3_IMAGE=${LF_SETTINGS[3]} AF3_MODEL_DIR=${LF_SETTINGS[4]}
if [[ -n ${SLURM_CPUS_PER_TASK:-} ]]; then
  [[ "$SLURM_CPUS_PER_TASK" -ge "${LF_SETTINGS[5]}" ]] || { echo 'Allocation has fewer CPUs than the run configuration'; exit 2; }
fi
LF_TMP=$(mktemp -d /tmp/lf-portable.XXXXXX)
export TMPDIR="$LF_TMP" FRESH_AF3_SOCKET="$LF_TMP/af3.sock" AF3_JAX_CACHE_DIR="$LF_TMP/jax"
export MPLCONFIGDIR="$LF_TMP/matplotlib" XDG_CACHE_HOME="$LF_TMP/cache" TORCH_HOME="$LF_TMP/torch"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export APPTAINER_BIND_PATHS="$LF_PACKAGE:$LF_PACKAGE;$LF_RUN:$LF_RUN;/tmp:/tmp"
unset LD_PRELOAD || true
mkdir -p "$AF3_JAX_CACHE_DIR"
LF_DAEMON=''
LF_MONITOR=''
LF_SEARCH=''
cleanup() {
  LF_CLEANUP_EXIT=$?
  if [[ -n "$LF_DAEMON" ]]; then
    /usr/bin/python3 - <<'PY' || true
import os,socket
try:
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as s:
        s.settimeout(5);s.connect(os.environ['FRESH_AF3_SOCKET'])
        s.sendall(b'{"cmd":"shutdown"}\n');s.recv(1024)
except OSError:
    pass
PY
    kill -TERM "$LF_DAEMON" 2>/dev/null || true
    wait "$LF_DAEMON" 2>/dev/null || true
  fi
  rm -rf -- "$LF_TMP"
  if [[ "$LF_MODE" == search && "$LF_CLEANUP_EXIT" == 0 ]]; then
    /usr/bin/python3 -B "$LF_PACKAGE/archive_results.py" --run "$LF_RUN" || LF_CLEANUP_EXIT=$?
  fi
  if [[ -n "$LF_MONITOR" ]]; then kill -TERM "$LF_MONITOR" 2>/dev/null || true; wait "$LF_MONITOR" 2>/dev/null || true; fi
  trap - EXIT
  exit "$LF_CLEANUP_EXIT"
}
trap cleanup EXIT
if [[ "$LF_MODE" == search && -n ${LF_EXPERIMENT_BASE:-} ]]; then
  /usr/bin/python3 -B "$LF_EXPERIMENT_BASE/monitor.py" "$$" lf "$(basename "$LF_RUN")" &
  LF_MONITOR=$!
fi
# Slurm's advance signal requests the existing LF graceful stop/checkpoint path.
trap '[[ -z "$LF_SEARCH" ]] || kill -USR1 "$LF_SEARCH" 2>/dev/null || true' USR1
if [[ "$LF_MODE" == search ]]; then
  LF_GPU=${CUDA_VISIBLE_DEVICES:?A Slurm allocation with one visible GPU is required}
  [[ "$LF_GPU" =~ ^[0-9]+$ ]] || { echo 'Expected one numeric Slurm GPU device'; exit 2; }
  /usr/bin/python3 "$LF_PACKAGE/vendor/af3/af3_daemon.py" serve \
    --socket-path "$FRESH_AF3_SOCKET" --gpu-device "$LF_GPU" \
    >"$LF_RUN/af3_daemon.log" 2>&1 &
  LF_DAEMON=$!
  for (( LF_TRY=0; LF_TRY<120; LF_TRY++ )); do
    [[ -S "$FRESH_AF3_SOCKET" ]] && break
    kill -0 "$LF_DAEMON"
    sleep 1
  done
  test -S "$FRESH_AF3_SOCKET"
  LF_SCRIPT="$LF_PACKAGE/run120.py"
  LF_ARGS=("$LF_RUN")
elif [[ "$LF_MODE" == cpu-smoke ]]; then
  LF_SCRIPT="$LF_PACKAGE/tests/cpu_smoke.py"
  LF_ARGS=("$LF_RUN")
elif [[ "$LF_MODE" == campaign-tests ]]; then
  LF_SCRIPT="$LF_PACKAGE/tests/test_campaign.py"
  LF_ARGS=()
elif [[ "$LF_MODE" == unit-tests ]]; then
  LF_SCRIPT="$LF_PACKAGE/tests/test_portable.py"
  LF_ARGS=()
else
  echo "Unknown mode: $LF_MODE"; exit 2
fi
if [[ -n "$LF_CPU_PYTHON" ]]; then
  CUDA_VISIBLE_DEVICES='' PYTHONPATH="$LF_PACKAGE" "$LF_CPU_PYTHON" -B "$LF_SCRIPT" "${LF_ARGS[@]}" &
else
  apptainer exec --no-mount /mnt \
    -B /tmp:/tmp -B "$LF_PACKAGE:$LF_PACKAGE" -B "$LF_RUN:$LF_RUN" \
    -B "$LF_NETMHCPAN:$LF_NETMHCPAN" -B "$LF_NETMHCPAN:/mnt/netmhcpan" \
    --env "CUDA_VISIBLE_DEVICES=,PYTHONPATH=$LF_PACKAGE" \
    "$LF_CPU_IMAGE" python3.9 -B "$LF_SCRIPT" "${LF_ARGS[@]}" &
fi
LF_SEARCH=$!
set +e
wait "$LF_SEARCH"
LF_RESULT=$?
# wait can return on USR1 while the child is still completing its checkpoint.
if kill -0 "$LF_SEARCH" 2>/dev/null; then wait "$LF_SEARCH"; LF_RESULT=$?; fi
exit "$LF_RESULT"
