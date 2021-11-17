FROM pytorch/pytorch:latest
RUN apt update -y
RUN apt install git -y
RUN apt install gcc -y
COPY $PWD /opt/project
ENV PYTHONPATH=/opt/project
WORKDIR /opt/project
RUN pip3 install -r requirements.txt
