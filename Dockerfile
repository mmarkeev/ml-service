FROM python:3.7-slim

COPY . /root

WORKDIR /root

RUN pip install -r requirements.txt