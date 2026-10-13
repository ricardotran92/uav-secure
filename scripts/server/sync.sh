#!/usr/bin/env bash
# Copy code to the GPU server, or fetch experiment outputs back.
# Uses tar over ssh, so it also works from Git Bash on Windows (no rsync needed).
#
#   scripts/server/sync.sh push    # local code   -> server:~/uav-secure
#   scripts/server/sync.sh pull    # server outputs -> local outputs/
set -euo pipefail

HOST="${UAV_HOST:-ricardo@140.123.104.113}"
REMOTE_DIR="${UAV_REMOTE_DIR:-uav-secure}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

case "${1:-}" in
  push)
    tar -C "$ROOT" -czf - --exclude=./outputs --exclude=./source_materials --exclude=./.git \
        --exclude='__pycache__' --exclude='*.egg-info' --exclude=./.pytest_cache . \
      | ssh "$HOST" "mkdir -p $REMOTE_DIR && tar -xzf - -C $REMOTE_DIR"
    echo "pushed code to $HOST:$REMOTE_DIR"
    ;;
  pull)
    ssh "$HOST" "cd $REMOTE_DIR && tar -czf - outputs" | tar -xzf - -C "$ROOT"
    echo "pulled outputs into $ROOT/outputs"
    ;;
  *)
    echo "usage: $0 push|pull" >&2
    exit 1
    ;;
esac
