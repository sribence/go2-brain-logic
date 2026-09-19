# perception pillar -- runs ON THE ROBOT (Jetson Orin NX, L4T R35.3.1 /
# JetPack 5.1.1) with CUDA. Base image ships torch+CUDA+ultralytics for
# JetPack 5 (Python 3.8). Build context: mission_control/ (CONVENTIONS.md).
#
#   docker build -f perception/Dockerfile -t nero_go2/perception .
#   docker run -d --name nero_go2_perception --runtime nvidia --network host \
#     -e RS_SOURCE=rosbridge -e YOLO_DEVICE=0 nero_go2/perception
#
# Dev PC (no GPU): --build-arg BASE=ultralytics/ultralytics:latest-cpu
ARG BASE=ultralytics/ultralytics:latest-jetson-jetpack5
FROM ${BASE}

WORKDIR /app
COPY perception/requirements.txt /app/perception/requirements.txt
RUN pip install --no-cache-dir -r /app/perception/requirements.txt
COPY perception/ /app/perception/
ENV PYTHONPATH=/app/perception
WORKDIR /app/perception
# Model weights are baked in so the robot needs no internet at start.
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
EXPOSE 9112
CMD ["python3", "app.py"]
