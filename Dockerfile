# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.10.0-slim

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /app
COPY . /app

# Download the punkt and averaged_perceptron_tagger packages to the NLTK data directory
RUN python -c "import nltk;nltk.download('punkt', download_dir='/usr/share/nltk_data');nltk.download('averaged_perceptron_tagger', download_dir='/usr/share/nltk_data')"

# Creates a non-root user with an explicit UID and adds permission to access the /app folder and NLTK data directory
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app /usr/share/nltk_data
USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["streamlit", "run", "ocsvm.py","--server.port","8080","--server.headless","true"]
