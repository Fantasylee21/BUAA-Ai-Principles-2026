#!/usr/bin/env bash
# Launch in tmux; keep credentials out of arguments, files and logs.
set -euo pipefail

rl_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
rl_repo_root="$(cd -- "$rl_script_dir/.." && pwd)"
cd -- "$rl_repo_root"

rl_foreground=0
if [[ "${1:-}" == "--foreground" ]]; then
  rl_foreground=1
  shift
fi
rl_python="${EMOTION_PYTHON:-$(command -v python)}"
rl_run_dir=""
rl_previous=""
for rl_argument in "$@"; do
  if [[ "$rl_previous" == "--run-dir" || "$rl_previous" == "--output-dir" ]]; then
    rl_run_dir="$rl_argument"
  fi
  rl_previous="$rl_argument"
done
if [[ -z "$rl_run_dir" ]]; then
  printf '%s\n' 'Pass --run-dir PATH and the model/data arguments; see rl/README.md.' >&2
  exit 2
fi

if [[ "$rl_foreground" == 0 ]]; then
  command -v tmux >/dev/null
  rl_session="emotion-$(date -u +%Y%m%dT%H%M%S)-$$"
  printf -v rl_command '%q ' env "EMOTION_PYTHON=$rl_python" "PATH=$PATH" \
    bash "$rl_script_dir/run_rl.sh" --foreground "$@"
  tmux new-session -d -s "$rl_session" -c "$rl_repo_root" "$rl_command"
  printf 'Session: %s\nAttach: tmux attach -t %s\nLog: %s.console.log\n' \
    "$rl_session" "$rl_session" "$rl_run_dir"
  exit 0
fi

mkdir -p -- "$(dirname -- "$rl_run_dir")"
# The console log is a sibling, so it does not make a fresh run-dir nonempty.
set +e
"$rl_python" -u -m rl.train_grpo "$@" 2>&1 | tee -a -- "$rl_run_dir.console.log"
rl_status="${PIPESTATUS[0]}"
set -e
printf 'Runner exit code: %s\n' "$rl_status" | tee -a -- "$rl_run_dir.console.log"
exit "$rl_status"
