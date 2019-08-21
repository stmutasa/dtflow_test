# Import tensorflow 1.13.2 python 3 image
FROM tensorflow/tensorflow:1.14.0-py3

# Change container working directory for all following commands
WORKDIR /app

# Copy the contents of the current directory to the working directory
COPY . /app

# Now install the dependencies
RUN pip3 install -r requirements.txt

# Default command will be running the parameter server
CMD python3 Distributed.py --task ps
