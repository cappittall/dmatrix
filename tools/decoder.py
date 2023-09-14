from pylibdmtx.pylibdmtx import decode
import cv2

class DataMatrixDecoder:
    def __init__(self):
        pass

    @staticmethod
    def decode_image(image):
        """Decode Data Matrixes in the given image."""
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Decode the Data Matrixes
        barcodes = decode(gray_image)

        decoded_objects = []
        for i, barcode in enumerate(barcodes):
            rect = barcode.rect
            barcode_data = barcode.data.decode("utf-8")
            decoded_objects.append({"box_number": i, "rect": rect, "data": barcode_data})

        return decoded_objects