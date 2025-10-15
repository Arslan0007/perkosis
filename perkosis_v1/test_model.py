import cv2
from ultralytics import YOLO
import pytesseract
import numpy as np
from database import BIB_ROSTER
from PIL import Image, ImageDraw, ImageFont

# Assuming your database file is named 'database.py' and contains the BIB_ROSTER dictionary
# Example: from database import BIB_ROSTER
# Since I don't know the exact name of the list/dictionary in your database.py,
# I will define a placeholder here. **You must replace this with the actual import.**

# --- START: Placeholder Roster (REPLACE THIS WITH YOUR ACTUAL IMPORT) ---
# Assuming your database.py looks like: BIB_ROSTER = {"7": "Naci Arslan", ...}
BIB_ROSTER = {
    "7": "Naci Arslan",
    "179": "Şükrü",
    "499": "Rüştü Düzer",
    # Add all your actual numbers and names here
}
# --- END: Placeholder Roster ---

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'


# 1. Load your trained model
model = YOLO('runs/detect/bib_detector_v18/weights/best.pt')
# runs/detect/bib_detector_v1/weights/
# 2. Define the path to the test image
image_path = 'dataset/image_11.jpg' \
''

# 3. Run inference on the test image
results = model(image_path, verbose=False)

# Load the image using OpenCV to draw the results and for cropping
frame = cv2.imread(image_path)

if frame is None:
    print(f"Error: Could not load image from {image_path}")
else:
    # Process the results
    for r in results:
        # Check if any bounding boxes were detected
        if r.boxes:
            for box in r.boxes:
                # Get bounding box coordinates (x1, y1, x2, y2)
                # Convert to integer and list for easy cropping
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()

                # Ensure coordinates are valid for cropping
                if x1 < 0: x1 = 0
                if y1 < 0: y1 = 0

                # 4. Crop the image to just the detected number
                # Note: We use y1:y2 first for rows (height), then x1:x2 for columns (width)
                cropped_number_img = frame[y1:y2, x1:x2]

                # ------------------------------------------------------------------
                # --- STEP 1: Aggressive Upscaling (CRUCIAL for small, distant bibs) ---
                # Scale by a factor of 4 to give Tesseract more pixels to work with.
                scale_factor = 5
                h, w, _ = cropped_number_img.shape
                if w > 0 and h > 0: # Safety check
                    cropped_number_img = cv2.resize(
                        cropped_number_img, 
                        (w * scale_factor, h * scale_factor), 
                        interpolation=cv2.INTER_CUBIC
                    )

                # --- Image Pre-processing for better OCR (Crucial for accuracy) ---
                # 1. Convert to grayscale (Always helpful)
                gray = cv2.cvtColor(cropped_number_img, cv2.COLOR_BGR2GRAY)

                blurred = cv2.medianBlur(gray, 3)

                # 2. Apply a slight blur (for noise reduction)
                contrast_img = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
                
                # 3. Apply ADAPTIVE THRESHOLDING (better for images with varying light)
                # This makes the numbers black on a clean white background.
                _, ocr_image = cv2.threshold(
                    blurred,
                    0,
                    255,
                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                )

                kernel = np.ones((2, 2), np.uint8)
                ocr_image = cv2.morphologyEx(ocr_image, cv2.MORPH_OPEN, kernel)
                ocr_image = np.ascontiguousarray(ocr_image, dtype=np.uint8)
                # Apply thresholding (simple binary threshold)
                # _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Use the pre-processed image for OCR
                # ocr_image = thresh
                
                # 5. Perform OCR to read the number
                # Use --psm 8 config for reading a single line of text (like a number)
                # Use --oem 3 for default Tesseract OCR engine mode
                try:
                    bib_number_str = pytesseract.image_to_string(
                        ocr_image, 
                        config='--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789'
                    )
                except pytesseract.TesseractNotFoundError:
                    print("Tesseract is not installed or not in your PATH. Cannot perform OCR.")
                    bib_number_str = "OCR_Error"
                
                # 6. Clean and format the extracted string
                # Remove all non-digit characters (spaces, newlines, noise)
                clean_number = ''.join(filter(str.isdigit, bib_number_str)).strip()

                # 7. Lookup the name in the roster
                if clean_number and clean_number in BIB_ROSTER:
                    name = BIB_ROSTER[clean_number]
                    display_text = f"{clean_number} - {name}"
                    color = (0, 255, 0) # Green for success
                else:
                    name = "UNKNOWN"
                    display_text = f"Detected: {clean_number} - UNKNOWN"
                    color = (0, 0, 255) # Red for unknown

                # 8. Draw the bounding box and the name on the original frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # --- Unicode Text Drawing using PIL/Pillow ---
                try:
                    # 1. Convert the OpenCV image (NumPy arrray) to a Pillow image
                    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_img)

                    # 2. Define a font that supports Turkish characters
                    # NOTE: You must change "arial.ttf" to the actual path of a font file on your system
                    # For Windows, C:\Windows\Fonts\arial.ttf
                    try:
                        font_path = "C:/Windows/Fonts/arial.ttf"
                        font_size = 30
                        font = ImageFont.truetype(font_path, font_size)
                    except IOError:
                        # Fallback if the font path is wrong or the font doesn't exist
                        print("Warning: Could not load Arial font. Falling back to default Pillow font.")
                        font = ImageFont.load_default()

                    # Convert BGR color tuple to an RGB color tuple for Pillow
                    # OpenCV uses BGR Pillow uses RGB
                    rgb_color = (color[2], color[1], color[0])

                    # 3. Draw the text onto the Pillow image
                    # use x1, y1 - (font_size * 1.5) to position the text slightly above the box
                    text_position = (x1, y1 - int(font_size * 2.5))
                    draw.text(text_position, display_text, font=font, fill=rgb_color)

                    # 4. Convert the Pillow image back to an OpenCV NumPy array
                    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                except Exception as e:
                    print(f"Error drawing Unicode text with PIL: {e}")
                    cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                print(f"Result: {display_text}")

        else:
            print("No bib numbers detected in the image.")

    # Show the final results image
    # Note: cv2.imshow requires a loop and cv2.waitKey() to be stable, 
    # but for a single image, this should work.
    cv2.imshow("Bib Number Recognition", frame)
    cv2.waitKey(0) # Wait infinitely for a key press
    cv2.destroyAllWindows()