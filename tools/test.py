import requests
import time
import glob
import base64
from pathlib import Path

files = glob.glob('data/dms/*.*')

def start():
    for file in files[:]:
        fn = file.split('/')[-1]
        data = {'image': (fn, open(file, 'rb'))}
        start_t= time.monotonic()
        response = requests.post('http://127.0.0.1:8000/upload/', files=data)
               

        try:
            json_data = response.json()  # Try to decode the response content as JSON

            print(f'Time {(time.monotonic() - start_t)*1000:.0f} ',json_data['DataMatrix'])
            
            # Save the response to the disk
            image_data = base64.b64decode(json_data['file'])
            
            output_filepath = Path('data/rsts') / Path(file).name
            output_filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_filepath, 'wb') as f:
                f.write(image_data) 
                
        except requests.exceptions.JSONDecodeError:
            print("Failed to decode response content as JSON")  # 
                
   
if __name__=="__main__":
    start()