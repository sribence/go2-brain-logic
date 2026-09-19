#!/bin/bash
# (Re)start the follow executor on the Go2 Jetson. The code is bind-mounted
# from $ROOT, so copying executor.py there and running this is the deploy.
# It restarts after a reboot (unless-stopped) so the console can use it
# without a manual step; it never arms mc_motion, the operator does.
set -e
ROOT=/home/unitree/nero_go2_dev/follow_executor
docker rm -f nero_go2_follow_executor >/dev/null 2>&1 || true
docker run -d --name nero_go2_follow_executor --network host --restart unless-stopped \
  -e EXEC_MAX_VX="${EXEC_MAX_VX:-0.4}" -e EXEC_MAX_VX_BACK="${EXEC_MAX_VX_BACK:-0.2}" \
  -e EXEC_MAX_VYAW="${EXEC_MAX_VYAW:-0.8}" -e MAX_RESULT_AGE_S="${MAX_RESULT_AGE_S:-0.4}" \
  -v "$ROOT:/exec:ro" nero_go2/web_dashboard:latest python /exec/executor.py
echo "EXECUTOR_DONE $(date)"
