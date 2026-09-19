#!/bin/bash
# Build and (re)start the perception container on the Go2 Jetson.
# Expects the build context at $ROOT/ctx (mission_control layout: ctx/perception/...).
set -e
ROOT=/home/unitree/nero_go2_dev/perception
cd "$ROOT"
# Wait if a base image pull is still running (the first pull takes hours on slow WiFi).
while pgrep -f "docker pull ultralytics" >/dev/null; do sleep 10; done
docker build -t nero_go2/perception:latest -f ctx/perception/Dockerfile ctx 2>&1 | tail -15
docker rm -f nero_go2_perception 2>/dev/null || true
mkdir -p logs
docker run -d --name nero_go2_perception --runtime nvidia --network host --restart unless-stopped \
  -e RS_SOURCE=rosbridge -e YOLO_DEVICE=0 -e PERCEPTION_ALLOW_LIVE="${PERCEPTION_ALLOW_LIVE:-0}" \
  -v "$ROOT/logs:/app/perception/logs" nero_go2/perception:latest
echo "DEPLOY_DONE $(date)"
