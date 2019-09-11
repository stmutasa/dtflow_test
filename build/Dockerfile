# Import tensorflow python 3 gpu image
FROM tensorflow/tensorflow:1.14.0-gpu-py3

# Change container working directory for all following commands
WORKDIR /app

# Copy the contents of the current directory to the working directory
COPY . /app

# Now install the dependencies
RUN pip3 install -r requirements.txt
RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN pip3 install opencv-python

# Default command will be running the parameter server
CMD ["python3", "/dtflow_test/Distributed2.py", "--ps_hosts=192.168.0.12:2222", "--worker_hosts=192.168.0.33:2223,192.168.0.12:2224", "--job_name=worker", "--task_index=0"]
