FROM python:3.9.11

RUN /usr/local/bin/python -m pip install --upgrade pip
COPY ./requirements.txt ./app/requirements.txt
RUN pip install -r ./app/requirements.txt

COPY ./templates ./app/templates
COPY ./static ./app/static
COPY ./backbones ./app/backbones
COPY utils.py ./app/utils.py
COPY app.py ./app/app.py
COPY sentinel.py ./app/sentinel.py

WORKDIR /app 
CMD ["python3", "./app.py"]