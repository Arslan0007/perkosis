import cv2
import pytesseract
import numpy as np

# NOTE: You MUST configure pytesseract to know where your installed Tesseract executable is located.
# Example for Windows (adjust path as needed):
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

def find_and_read_bib(image_path):
    """
    Simulates the finish line process: load image, pre-process, find number, read number.
    """
    print(f"--- Processing Image: {image_path} ---")

    # 1. Load the Image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image. Check path.")
        return

    # Convert to grayscale for simpler processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Simple Image Pre-processing (Enhance contrast/clarity)
    # Applying a bilateral filter to reduce noise while keeping edges sharp
    # The best pre-processing steps will depend heavily on your specific race footage.
    processed_img = cv2.bilateralFilter(gray, 9, 75, 75)

    # Apply adaptive thresholding to create a binary image (good for OCR)
    _, binary_img = cv2.threshold(processed_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Detect and Isolate the Bib (Placeholder - This is the hard part!)
    # In a real solution, you'd use a deep learning model (YOLO) here to get the bounding box.
    # For this simple example, we'll try to let Tesseract find *any* text.

    # 4. Use Tesseract to read text from the entire image
    # We specify configuration for reading digits only
    custom_config = r'--oem 3 --psm 6 outputbase digits'
    
    # You might want to try different PSM (Page Segmentation Modes)
    # PSM 6: Assume a single uniform block of text.
    # PSM 8: Assume a single word.
    
    text = pytesseract.image_to_string(binary_img, config=custom_config)
    
    # Clean the result (remove non-numeric, non-whitespace characters)
    bib_number = ''.join(filter(str.isdigit, text))

    print(f"Detected Raw Text: '{text.strip()}'")
    print(f"Extracted Bib Number: {bib_number}")

    # Display the processed image for visual inspection
    cv2.imshow('Processed Image', binary_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# --- Execution ---
# CREATE A TEST IMAGE: Find a photo of a runner with a clear bib number
# and save it in the same directory as this script as 'runner_test.jpg'.

find_and_read_bib('C:/dev/perkosis/dataset/test.jpg')