FROM pytorch/pytorch:latest
RUN apt update -y
RUN apt install -y build-essential ffmpeg cmake git wget pkg-config vim
RUN apt install -y libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev libgl1-mesa-glx

RUN apt install -y python3-pip graphviz
RUN apt install -y libsm6 libxext6 libfontconfig1 libxrender1 libgl1-mesa-dev
RUN apt install gcc -y
COPY $PWD /opt/project
ENV PYTHONPATH=/opt/project
WORKDIR /opt/project
RUN pip3 install -r requirements.txt
