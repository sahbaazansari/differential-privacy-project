FROM python:3.9-slim

RUN pip install python-dp==1.1.4 pandas scikit-learn matplotlib seaborn requests

WORKDIR /app
COPY . /app

CMD ["python", "main.py"]
