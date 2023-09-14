
import cv2
from pylibdmtx.pylibdmtx import decode

# Load the image
image = cv2.imread('data/dms/4.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Decode the Datamatrix barcodes
barcodes = decode(gray_image)
print(barcodes)
# Loop over the detected barcodes
for barcode in barcodes:

    # Extract the bounding box of the barcode and draw it on the image
    rect = barcode.rect
    # cv2.rectangle(image, (rect.left, rect.top), (rect.left + rect.width, rect.top + rect.height), (0, 255, 0), 2)

    # Get the barcode data
    barcode_data = barcode.data.decode("utf-8")

    # Print the barcode data
    print(f"Detected Datamatrix barcode with data: {barcode_data}")

# Save and display the output image
cv2.imwrite('data/result/output.jpg', image)

