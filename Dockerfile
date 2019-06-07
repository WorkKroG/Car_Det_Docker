FROM ubuntu:18.04

RUN apt-get update && apt-get install -y \
    python3-pip
# We copy this file first to leverage docker cache
COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN python3 -m pip install -r requirements.txt

COPY . /app

ENV PYTHONPATH =.
ENV PYTHONBUFFERED = 1

CMD [ "python", "./MainBot.py"]