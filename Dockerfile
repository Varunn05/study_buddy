FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install pipenv

COPY Pipfile Pipfile.lock ./

RUN pipenv install --system --deploy

COPY . .

RUN mkdir -p vectorstore temp

EXPOSE 8501 9999

CMD ["sh", "-c", "python setup.py & streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]