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
RS_SOURCE="${RS_SOURCE:-realsense}"
# The camera has a single owner. With RS_SOURCE=realsense the perception
# container reads it directly, so the ROS bridge must not hold it.
# Undo: RS_SOURCE=rosbridge ./deploy.sh (it starts the bridge again).
if [ "$RS_SOURCE" = realsense ]; then
  docker stop nero_go2_realsense_bridge >/dev/null 2>&1 || true
else
  docker start nero_go2_realsense_bridge >/dev/null 2>&1 || true
fi
mkdir -p logs models
docker run -d --name nero_go2_perception --runtime nvidia --network host --restart unless-stopped \
  --privileged -v /dev/bus/usb:/dev/bus/usb \
  -e OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" -e OPENBLAS_NUM_THREADS=2 -e MKL_NUM_THREADS=2 \
  -e RS_SOURCE="$RS_SOURCE" -e RS_FPS="${RS_FPS:-15}" -e RS_USB_KEEP_AWAKE="${RS_USB_KEEP_AWAKE:-1}" -e YOLO_DEVICE=0 -e PERCEPTION_ALLOW_LIVE="${PERCEPTION_ALLOW_LIVE:-0}" \
  -v "$ROOT/logs:/app/perception/logs" -v "$ROOT/models:/app/perception/models" \
  -v /var/lib/nvpmodel:/host/nvpmodel:ro -v /etc/nvpmodel.conf:/host/nvpmodel.conf:ro \
  nero_go2/perception:latest
echo "DEPLOY_DONE $(date)"
