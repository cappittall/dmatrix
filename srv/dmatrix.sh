#!/bin/bash

# Change the working directory to /home/mendel/dmatrix/
cd /home/mendel/dmatrix 

source /opt/myenv/bin/activate

sleep 0.1

# Start the uvicorn server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload >> /home/cappittall/Documents/Farmakod/dmatrix/logs.txt 2>&1
