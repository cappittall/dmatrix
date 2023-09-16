import time
from pylibdmtx.pylibdmtx import decode
import cv2
import numpy as np
 
class DataMatrixDecoder:
    @staticmethod
    def decode_image(args):
        index, image = args
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        barcodes = decode(gray_image, max_count=1)
        if barcodes:
            rect = barcodes[0].rect
            barcode_data = barcodes[0].data.decode("utf-8")
            return {"index": index, "box_number": 0, "rect": rect, "data": barcode_data}
        else:
            return {"index": index}
