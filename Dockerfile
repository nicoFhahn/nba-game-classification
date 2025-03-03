FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

ARG CACHEBUST=1
RUN git clone https://github.com/nicoFhahn/nba-game-classification.git .
RUN pip3 install -U pip
RUN pip3 install -r requirements.txt
RUN pip3 uninstall streamlit
RUN pip3 install streamlit-nightly --upgrade
EXPOSE 8080

HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit/Home.py", "--server.port=8080", "--server.address=0.0.0.0"]