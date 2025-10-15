import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
import pytesseract
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont 
import time

# -------------------------------------------------------------
# --- 1. CONFIGURATION AND DATABASE ---
# -------------------------------------------------------------
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Database Roster (ADD YOUR NAMES HERE)
BIB_ROSTER = {
    "125": "Naci Arslan",
    "58": "Sukru Yilmaz", # Example of a full name
    "3": "A",
    "65": "B",
    "31": "Rustu Duzer",
    "30": "Diyar",
    "86": "Ali",
    "91": "Veli",
    "96": "Mehmet Can",
    "102": "Kazim Kaya",
    "114": "C",
    "101": "D",
    "129": "E",
    # Add all your actual numbers and names here
}

# -------------------------------------------------------------
# --- 2. MODEL AND VIDEO SETUP ---
# -------------------------------------------------------------
model = YOLO('C:/dev/perkosis/runs/detect/bib_detector_v18/weights/best.pt') 

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('Kosu1.mp4')

# Get video frame rate for accurate time calculation
fps = cap.get(cv2.CAP_PROP_FPS)

# Record script start time (This is the official timing start)
start_datetime = datetime.now() 

my_file = open('classes.txt', "r")
data = my_file.read()
class_list = data.split("\n") 

# Finish Line Area (Using a single, strict line for timing)
# Adjust these coordinates to your precise finish line location (e.g., a narrow vertical gate)
log_area = np.array([
    [500, 650],  # Top-Left of the narrow line
    [440, 750],  # Bottom-Left
    [1170, 750],  # Bottom-Right
    [1145, 650]   # Top-Right
], np.int32)

# Visualization Area (A wider polygon to show the general detection zone)
area_viz = np.array([(500, 652), (440, 746), (1170, 750), (1145, 678)], np.int32)

count = 0
processed_numbers = set()

# Open file for writing bib number data (Using 'w' to ensure a clean file start)
with open("bib_number_data.txt", "w") as file:
    # New header: Number, Name, Full Datetime, Time of Day, Elapsed Seconds
    file.write("Number\tName\tDatetime\tTimeOfDay\tElapsedSeconds\n") 

# -------------------------------------------------------------
# --- 3. MAIN VIDEO PROCESSING LOOP ---
# -------------------------------------------------------------
while True:
    ret, frame = cap.read()
    
    # Calculate current frame number
    current_frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
    
    count += 1
    if count % 1 != 0:
        continue
    if not ret:
        print("End of video stream or error reading frame.")
        break
    
    frame = cv2.resize(frame, (1680, 900))
    results = model.predict(frame, verbose=False) # Added verbose=False for cleaner output
    
    # Check if any boxes were detected
    if results and results[0].boxes:
        a = results[0].boxes.data
        
        # FIX: Ensure numpy() is called and data is converted for pandas
        px = pd.DataFrame(a.cpu().numpy()).astype("float")
    
        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            
            # cx and cy are center points of the detected bib
            cx = int(x1 + x2) // 2
            cy = int(y1 + y2) // 2
            
            # Check if the bib's center is inside the strict log_area
            result_area = cv2.pointPolygonTest(log_area, ((cx, cy)), False)
            
            # Only proceed if the center point is strictly inside the log_area
            if result_area >= 0:
                
                # Ensure coordinates are valid for cropping
                x1_crop = max(0, x1)
                y1_crop = max(0, y1)
                cropped_number_img = frame[y1_crop:y2, x1_crop:x2]
                
                # Check for valid crop size before proceeding
                if cropped_number_img.size == 0:
                    continue

                # --- OCR Preprocessing (Improved) ---
                scale_factor = 5
                h, w, _ = cropped_number_img.shape
                if w > 0 and h > 0:
                    cropped_number_img = cv2.resize(
                        cropped_number_img,
                        (w * scale_factor, h * scale_factor),
                        interpolation=cv2.INTER_CUBIC
                    )
                
                gray = cv2.cvtColor(cropped_number_img, cv2.COLOR_BGR2GRAY)
                blurred = cv2.medianBlur(gray, 3)

                _, ocr_image = cv2.threshold(
                    blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                )
                
                kernel = np.ones((2, 2), np.uint8)
                ocr_image = cv2.morphologyEx(ocr_image, cv2.MORPH_OPEN, kernel)
                ocr_image = np.ascontiguousarray(ocr_image, dtype=np.uint8)
                
                # --- Perform OCR ---
                bib_number_str = pytesseract.image_to_string(
                    ocr_image, 
                    config='--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789'
                )
                
                # Clean and lookup
                clean_number = ''.join(filter(str.isdigit, bib_number_str)).strip()
                name = BIB_ROSTER.get(clean_number, "UNKNOWN") # Look up name
                
                is_known = name != "UNKNOWN"
                is_new = clean_number and clean_number not in processed_numbers

                # --- Conditional Logging ---
                if is_new and is_known:
                    
                    # 1. Log to set to prevent duplicates
                    processed_numbers.add(clean_number) 

                    # 2. Calculate Timing
                    current_datetime = datetime.now()
                    
                    # Time elapsed from start of video/script in seconds
                    # Assuming constant FPS, time = frame_number / FPS
                    # This is better for video analysis than system time.
                    elapsed_seconds = current_frame_num / fps 
                    
                    # Format time data
                    current_date = current_datetime.strftime("%Y-%m-%d")
                    current_time_str = current_datetime.strftime("%H:%M:%S")
                    
                    # 3. Log to file (Required format)
                    try:
                        with open("bib_number_data.txt", "a") as file:
                            file.write(
                                f"{clean_number}\t"
                                f"{name}\t"
                                f"{current_date} {current_time_str}\t"
                                f"{current_time_str}\t"
                                f"{elapsed_seconds:.3f}\n"
                            )
                        print(f"**LOGGED: {clean_number} - {name} at {elapsed_seconds:.3f}s**")
                    except Exception as e:
                        print(f"ERROR writing to file: {e}")
                        
                # 4. Draw bounding box and text (using PIL for unicode names)
                color = (0, 255, 0) if is_known else (0, 0, 255)
                display_text = f"{clean_number} - {name}"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                try:
                    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_img)
                    font_path = "C:/Windows/Fonts/arial.ttf"
                    font = ImageFont.truetype(font_path, 30)
                    rgb_color = (color[2], color[1], color[0])
                    text_position = (x1, y1 - 75)
                    draw.text(text_position, display_text, font=font, fill=rgb_color)
                    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                except Exception:
                    cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
    # Draw the STICKT logging gate (Green) and the wider visualization area (Blue)
    cv2.polylines(frame, [log_area], True, (0, 255, 0), 3) 
    # cv2.polylines(frame, [area_viz], True, (255, 0, 0), 1)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()