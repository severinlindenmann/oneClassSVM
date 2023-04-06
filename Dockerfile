FROM python:3.9-slim-buster

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code into the container
COPY . ./app

# Set the command to run the app when the container starts
CMD ["streamlit", "run", "ocsvm.py"]