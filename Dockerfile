FROM python:3.6

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Copy the requirements file
COPY /new_requirements.txt /app/requirements.txt
COPY . /app/
# Install dependencies
#RUN pip install -r requirements.txt

# Install CMake and clone ANTsPy repository from GitHub
RUN apt-get update && apt-get install -y cmake git 
#    && git clone https://github.com/ANTsX/ANTsPy /app/ANTsPy \
 #   && python3 /app/ANTsPy/setup.py install

RUN pip install -r requirements.txt

