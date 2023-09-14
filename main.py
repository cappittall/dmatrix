import subprocess
import time
import os
from pathlib import Path

from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.responses import JSONResponse
import cv2
import numpy as np
from io import BytesIO
import base64
import logging

from tools.model_processor import ModelProcessor
from tools.decoder import DataMatrixDecoder
from tflite_runtime.interpreter import load_delegate
logging.basicConfig(level=logging.INFO)

app = FastAPI()

templates = Jinja2Templates(directory="templates")
static_dir = os.path.join(os.path.dirname(__file__), 'static')
app.mount("/static", StaticFiles(directory=static_dir), name="static")

import platform

    
system_name = platform.system()
delegate = 'libedgetpu.so.1' if system_name == "Linux" else "libedgetpu.dll" \
                                if system_name =="Windows" else None

try: 
    EDGE = True
    load_delegate(load_delegate(delegate))
except: 
    EDGE = False
    
    
# set static vars.
models_path = 'models/all/'
model = "data-matrix-v0_edgetpu.tflite" if EDGE else "data-matrix-v0.tflite"
message=''
class_names = ['D.Matrix', '???']
threshold_file = Path("data/system/threshold.txt")

# set initial detector
# Read the threshold value from the file
if threshold_file.is_file():
    with open(threshold_file, "r") as file:
        threshold = float(file.read())
else:
    threshold = 0.30  # Default value if the file does not exist
    
# create detector Instace
detector = ModelProcessor(os.path.join(models_path, model), threshold = threshold)
detector.interpreter.allocate_tensors()

# Create an instance of DataMatrixDecoder
decoder = DataMatrixDecoder()

def object_detection(image: np.ndarray):
    """
    Object detection 
    """
    start_t = time.monotonic()
    objs = detector.run_prediction(image.copy())
    end_t = (time.monotonic() - start_t) * 1000 
    height, width, _ = image.shape
    
    # Create a list to store the final object details
    final_objects = []
    DataMatrix = {}
    for i, obj in enumerate(objs):
        # Convert coordinates from relative to absolute
        x_min, y_min, x_max, y_max = obj.bbox 
        x1, y1, x2, y2 = ( int(xy) for xy in [(x_min * width), 
                                              (y_min * height), 
                                              (x_max * width), 
                                              (y_max * height)])
        
        confidence = obj.score       
         # Draw a number on each bounding box
        cv2.putText(image, str(i+1), (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        
        # Crop the image (10% wider)
        dx = int(0.1 * (x2 - x1))
        dy = int(0.1 * (y2 - y1))
        cropped_image = image[max(y1-dy,0):min(y2+dy,height), max(x1-dx,0):min(x2+dx,width)]
        
        # Decode data matrix in the cropped image
        decoded_data = decoder.decode_image(cropped_image)
        (color, data) = ((0, 255, 0), str(decoded_data[0].get('data',None))) if decoded_data else ((0,0,255), [])
     
        print('decoded_data', data) 
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        DataMatrix[str(i+1)] = {"bbox": [x1,y1,x2,y2], "datamatrix":data, "dy": f"{int(confidence*100)}%" }
        
    return image, DataMatrix

# Update the main page endpoint
@app.get("/")
async def main(request: Request,  message: str = ''):
    global threshold
    models = os.listdir(models_path)
     
    # Read the threshold value from the file
    if threshold_file.is_file():
        with open(threshold_file, "r") as file:
            threshold = float(file.read())
    else:
        threshold = 0.3  # Default value if the file does not exist

    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request, 
            'models': models, 
            'message': message, 
            'threshold': threshold,  # Pass the threshold value to your template
            'model': model
        }
    )

@app.post('/update/')
async def update_setting(selected_model: str = Form(...), threshold: float = Form(...)):
    global detector, model
    model = selected_model
     # Save the threshold value to a file
    with open(threshold_file, "w") as file:
        file.write(str(threshold))
        
    # Update the model and threshold of the existing detector instance
    detector.update_model(os.path.join(models_path, selected_model))
    detector.update_threshold(threshold)
    
    url = '/?message=Değişiklikler başarılı bir şekilde yapıldı...'
    return RedirectResponse(url=url, status_code=302)

@app.post('/gitpull')
async def git_pull_and_restart():
    
    try:
        # Execute the git pull command
        subprocess.run(['git', 'pull'], check=True)

        # Execute the . kur command (source the kur script)
        subprocess.run(['. kur'], shell=True, check=True)

        url = '/?message=Githup tan yükleme başarılı bir şekilde yapıldı...'
        return RedirectResponse(url=url, status_code=302)
    except Exception as e:
        url = '/?message=HATA OLUŞTU TEKRAR DENEYİNİZ...'
        return RedirectResponse(url=url, status_code=302)

@app.post("/upload/")
async def upload_image(image: UploadFile = Form(...)):
    global detector
    image_content = await image.read()
    
    img_name = image.filename
    print('img_name',img_name)
    # Convert bytes to OpenCV image
    nparr = np.frombuffer(image_content, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    img_np, data_matrix = object_detection(img_np.copy())
    
    imname = img_name.split('.')[0]
    cv2.imwrite(f'data/results/{imname}.jpg', img_np)
    
    # Convert the image with bbox back to bytes
    _, buffer = cv2.imencode(".tif", img_np)
    
    io_buf = BytesIO(buffer)
    encoded_img = base64.b64encode(io_buf.getvalue()).decode("utf-8")
    content = {"DataMatrix": data_matrix, "file": encoded_img}
    return JSONResponse(content= content)

