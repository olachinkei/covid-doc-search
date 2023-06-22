FROM python:3.11
USER root

RUN mkdir -p /root/src
COPY requirements.txt /root/src
WORKDIR /root/src


RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install -r requirements.txt