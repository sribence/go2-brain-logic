# omni pillar (OmniVision 360) -- build context: mission_control/ (CONVENTIONS.md).
#
# Dev PC / CI (CPU, mock cameras):
#   docker build -f omni/Dockerfile -t nero_go2/omni .
#   docker run --rm -p 9114:9114 -e OMNI_MOCK=1 nero_go2/omni
#
# Robot (Jetson Orin NX, JetPack 5, Python 3.8) -- same base as perception,
# ships torch+CUDA+ultralytics and an OpenCV with GStreamer (nvv4l2decoder):
#   docker build -f omni/Dockerfile -t nero_go2/omni \
#     --build-arg BASE=ultralytics/ultralytics:latest-jetson-jetpack5 --build-arg EXTRA_REQS="" .
#   docker run -d --name nero_go2_omni --runtime nvidia --network host \
#     --device /dev/video0 --device /dev/video2 --device /dev/video4 --device /dev/video6 \
#     -v /dev/v4l:/dev/v4l:ro -e REDIS_HOST=127.0.0.1 nero_go2/omni
ARG BASE=python:3.8-slim
FROM ${BASE}
ARG EXTRA_REQS="numpy opencv-python-headless"

WORKDIR /app
COPY core/ /app/core/
COPY omni/requirements.txt /app/omni/requirements.txt
RUN pip install --no-cache-dir -r /app/core/requirements.txt -r /app/omni/requirements.txt ${EXTRA_REQS}
COPY omni/ /app/omni/
# /app on the path: modules import each other as `omni.<module>`
ENV PYTHONPATH=/app:/app/core:/app/omni
WORKDIR /app/omni
EXPOSE 9114
CMD ["python3", "app.py"]
