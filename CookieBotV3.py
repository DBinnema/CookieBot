import csv

import pyautogui
import time
import datetime
import keyboard
from rapidfuzz import process

from tracking.adaptive_mean_filter import AdaptiveMeanFilter
from tracking.tracked_value import TrackedValue
import cv2
import numpy as np
import math
import os

import sys
import re

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as mticker


from PIL import ImageGrab
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


from matplotlib import ticker

#Since we will be reading values off the screen with suffixes it is helpful to fomalize the units here
SUFFIXES = {
    # Base units
    "k": 1_000,
    "thousand": 1_000,

    "m": 1_000_000,
    "mil": 1_000_000,
    "million": 1_000_000,

    "b": 1_000_000_000,
    "bil": 1_000_000_000,
    "billion": 1_000_000_000,

    "t": 1_000_000_000_000,
    "tril": 1_000_000_000_000,
    "trillion": 1_000_000_000_000,

    "qa": 1_000_000_000_000_000,
    "quad": 1_000_000_000_000_000,
    "quadrillion": 1_000_000_000_000_000,

    "qi": 1_000_000_000_000_000_000,
    "quin": 1_000_000_000_000_000_000,
    "quintillion": 1_000_000_000_000_000_000,

    "sx": 1_000_000_000_000_000_000_000,
    "sextillion": 1_000_000_000_000_000_000_000,

    "sp": 1_000_000_000_000_000_000_000_000,
    "septillion": 1_000_000_000_000_000_000_000_000,

    "oc": 1_000_000_000_000_000_000_000_000_000,
    "octillion": 1_000_000_000_000_000_000_000_000_000,

    "no": 1_000_000_000_000_000_000_000_000_000_000,
    "nonillion": 1_000_000_000_000_000_000_000_000_000_000,

    "dc": 1_000_000_000_000_000_000_000_000_000_000_000,
    "decillion": 1_000_000_000_000_000_000_000_000_000_000_000,
}

# these are the internally recognized "CANONICAL" sufixes
CANONICAL_SUFFIX = {
    "k": "thousand", "thousand": "thousand",
    "m": "million", "mil": "million", "million": "million",
    "b": "billion", "bil": "billion", "billion": "billion",
    "t": "trillion", "tril": "trillion", "trillion": "trillion",
    "qa": "quadrillion", "quad": "quadrillion", "quadrillion": "quadrillion",
    "qi": "quintillion", "quin": "quintillion", "quintillion": "quintillion",
    "sx": "sextillion", "sextillion": "sextillion",
    "sp": "septillion", "septillion": "septillion",
    "oc": "octillion", "octillion": "octillion",
    "no": "nonillion", "nonillion": "nonillion",
    "dc": "decillion", "decillion": "decillion",
}

def correct_suffix(ocr_suffix, known_suffixes, score_cutoff=80):
    match, score, _ = process.extractOne(ocr_suffix, known_suffixes, score_cutoff=score_cutoff)
    if match:
        return match
    return None





def get_number_from_region(region=(0, 0, 1400, 170), debug=False):
    try:
        # Capture the screen region
        img = np.array(ImageGrab.grab(bbox=region))

        # Convert to grayscale and apply thresholding
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if debug:
            cv2.imwrite("debug_number_read.png", thresh)

        # Run OCR with a strict whitelist for numbers and punctuation only
        text = pytesseract.image_to_string(
            thresh,
            config='--psm 6 -c tessedit_char_whitelist=0123456789.,'
        )

        print(f"[OCR Raw Number Text] {repr(text)}")

        # Extract the first numeric pattern found
        match = re.search(r'[\d.,]+', text)
        if match:
            num_str = match.group().replace(',', '')
            try:
                num = float(num_str)
                return num
            except ValueError:
                print(f"[OCR WARNING] Failed to convert number: '{num_str}'")
                return None

        print("[OCR WARNING] No numeric value found in OCR text.")
        return None

    except Exception as e:
        print(f"[OCR ERROR] {e}")
        return None



#this is the suffix to the cookies if it exists... million trillion....
def get_suffix_from_region(region, debug=False):
    try:
        img = np.array(ImageGrab.grab(bbox=region))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if debug:
            cv2.imwrite("debug_suffix_area.png", thresh)

        # OCR configured to recognize letters only (suffix)
        text = pytesseract.image_to_string(
            thresh,
            config='--psm 7 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ '
        ).lower().strip()

        print(f"[OCR Raw Suffix Text] {repr(text)}")

        # Clean OCR text: keep only letters
        cleaned = ''.join(c for c in text if c.isalpha())

        if not cleaned:
            print("[OCR WARNING] No suffix letters found.")
            return None

        canonical = CANONICAL_SUFFIX.get(cleaned)
        if not canonical:
            corrected = correct_suffix(cleaned, list(CANONICAL_SUFFIX.keys()))
            if corrected:
                canonical = CANONICAL_SUFFIX[corrected]
                print(f"[OCR FIX] Interpreted '{cleaned}' as '{corrected}' → '{canonical}'")

            else:
                print(f"[OCR WARNING] Unknown suffix: '{cleaned}'")
                return None

        return canonical

    except Exception as e:
        print(f"[OCR ERROR] {e}")
        return None



def compute_ocr_regions_cookie_count(
    cookies_box,
    suffix_width=80, suffix_pad=6,
    number_height=40, number_pad=5,
    cps_height=15, cps_pad=0,
    cps_suffix_width=100, cps_suffix_pad=50,
    cps_number_width=100  # ← NEW PARAM
):
    left, top, right, bottom = cookies_box

    # Suffix: to the left of the "cookies" word
    suffix_right = left - suffix_pad
    suffix_left = max(suffix_right - suffix_width, 0)
    suffix_top = top
    suffix_bottom = bottom

    # Number: directly above the "cookies" word
    number_left = suffix_left
    number_right = right
    number_bottom = top - number_pad
    number_top = max(number_bottom - number_height, 0)

    # CpS line: appears below the cookies word
    cps_top = bottom + cps_pad
    cps_bottom = cps_top + cps_height

    # CpS suffix is anchored to the right
    cps_suffix_right = right
    cps_suffix_left = max(cps_suffix_right - cps_suffix_width, 0)

    # CpS number is to the left of the suffix with a fixed width
    cps_number_right = cps_suffix_left - cps_suffix_pad
    cps_number_left = max(cps_number_right - cps_number_width, 0)

    cps_number_region = (cps_number_left, cps_top, cps_number_right, cps_bottom)
    cps_suffix_region = (cps_suffix_left, cps_top, cps_suffix_right, cps_bottom)

    suffix_region = (suffix_left, suffix_top, suffix_right, suffix_bottom)
    number_region = (number_left, number_top, number_right, number_bottom)

    return number_region, suffix_region, cps_number_region, cps_suffix_region


def read_filtered_value(number_region, suffix_region, value_filter, debug=False):
    try:
        raw_value = interpret_number_with_suffix(number_region, suffix_region, debug=debug)

        if raw_value is None:
            print("[OCR] Falling back to last known value")
            return value_filter.last()
        if debug:
            print(f"[OCR] Value before filter: {raw_value:.2f}")

        return value_filter.filter(raw_value)

    except Exception as e:
        print(f"[OCR ERROR] Exception in read value: {e}")
        return value_filter.last()


def interpret_number_with_suffix(number_region, suffix_region, debug=False):
    number = get_number_from_region(number_region, debug=debug)
    suffix = get_suffix_from_region(suffix_region, debug=debug)

    if number is None:
        print("[OCR] No valid number read")
        return None

    multiplier = SUFFIXES.get(suffix, 1) if suffix else 1
    return number * multiplier


def human_format(num):
    for unit in ['', 'K', 'M', 'B', 'T', 'Qa', 'Qi']:
        if abs(num) < 1000:
            return f"{num:.0f}{unit}"
        num /= 1000
    return f"{num:.1f}Qi+"





def record_cookie_count(timestamp, cookie_count, filename="cookie_log.csv"):
    with open(filename, "a") as f:
        f.write(f"{timestamp},{cookie_count}\n")  # CSV: time,cookie_count


def plot_cookie_log_with_events(cookie_log_path="cookie_log.csv", event_log_path="event_log.csv"):
    # Load cookie data
    df = pd.read_csv(cookie_log_path, parse_dates=["Timestamp"])
    df = df[df["Cookies"] > 0.0]  # Filter out invalid reads

    # Load event data (no header)
    event_df = pd.read_csv(event_log_path, names=["Timestamp", "Event"], parse_dates=["Timestamp"])

    # Define color mapping for known events
    event_colors = {
        "Bot Started": "blue",
        "Bot Stopped": "black",
        "Golden Cookie Clicked": "gold",
    }

    # Plot cookie growth
    plt.plot(df["Timestamp"], df["Cookies"], label="Cookies", color="green")

    # Overlay event markers
    for _, row in event_df.iterrows():
        color = event_colors.get(row["Event"], "red")  # Use red for unknown events
        plt.axvline(x=row["Timestamp"], color=color, linestyle="--", alpha=0.6)
        plt.text(row["Timestamp"], df["Cookies"].max() * 1.01, row["Event"],
                 rotation=90, fontsize=8, ha='right', va='bottom', color=color)

    # Final plot setup
    plt.xlabel("Time")
    plt.ylabel("Cookies")
    plt.title("Cookie Bank Over Time with Events")
    plt.legend()
    plt.grid(True)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()




def log_tick_formatter(val, pos=None):
    if val == 0:
        return "0"
    exponent = int(np.log10(val))
    return f"$10^{exponent}$"





def click(click_pos):
     pyautogui.moveTo(x=click_pos[0], y=click_pos[1])
     time.sleep(0.04)
     pyautogui.click(x=click_pos[0], y=click_pos[1])
     time.sleep(0.04)

def debugTemplateMatching(screenImg, template):
    result = cv2.matchTemplate(screenImg, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    print(f"Min: {min_val}, Max: {max_val}, Max Location: {max_loc}")

    result_display = result.copy()
    result_display = cv2.normalize(result_display, None, 0, 255, cv2.NORM_MINMAX)
    result_display = np.uint8(result_display)
    cv2.imwrite("debug_result_heatmap.png", result_display)



    #Better matching for finding things that may be different scale than the template images
def findBestScaledMatch_bbox(screenImg, template, **kwargs):
    """
    Wrapper for findBestScaledMatch that returns full bounding box (x, y, w, h).
    """
    screen_gray = cv2.cvtColor(screenImg, cv2.COLOR_BGR2GRAY)
    template_gray_orig = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    best_match = (0, (0, 0), None)  # (max_val, location, best_template)
    scale_range = kwargs.get('scale_range', (0.8, 1.2))
    scale_step = kwargs.get('scale_step', 0.05)
    threshold = kwargs.get('threshold', 0.8)

    for scale in np.arange(scale_range[0], scale_range[1] + scale_step, scale_step):
        resized_template = cv2.resize(template_gray_orig, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        if resized_template.shape[0] > screen_gray.shape[0] or resized_template.shape[1] > screen_gray.shape[1]:
            continue

        result = cv2.matchTemplate(screen_gray, resized_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val > best_match[0]:
            best_match = (max_val, max_loc, resized_template)
            if max_val > 0.95:
                break

    if best_match[0] < threshold:
        print("Did not find template at any scale.")
        return (0, 0, 0, 0)

    top_left = best_match[1]
    template_used = best_match[2]
    h, w = template_used.shape[:2]
    print(
        f"Best match confidence: {best_match[0]:.4f}, location: {best_match[1]}, template size: {resized_template.shape}")


    return (top_left[0], top_left[1], top_left[0] + w, top_left[1] + h)


def findBestScaledMatch_center(screenImg, template, **kwargs):
    """
    Returns center point from a (x1, y1, x2, y2) bounding box.
    """
    x1, y1, x2, y2 = findBestScaledMatch_bbox(screenImg, template, **kwargs)

    if x2 - x1 == 0 or y2 - y1 == 0:
        return (0, 0)  # No match found

    center_x = x1 + (x2 - x1) // 2
    center_y = y1 + (y2 - y1) // 2

    return (center_x, center_y)


#Blob detection for golden cookies
def find_golden_cookie_blobs(screenImg, min_area=1200, max_area=8000, min_circularity=0.1, debug=False):
    hsv = cv2.cvtColor(screenImg, cv2.COLOR_BGR2HSV)

    #Color matching
    lower_gold = np.array([15, 80, 120])
    upper_gold = np.array([40, 255, 255])

    mask = cv2.inRange(hsv, lower_gold, upper_gold)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    golden_cookies = []


    for cnt in contours:

        area = cv2.contourArea(cnt)

        if not (min_area <= area <= max_area):
            # Rejects blobs that are too small

            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * (area / (perimeter ** 2))
        if circularity < min_circularity:
            if debug:
                cv2.drawContours(screenImg, [cnt], -1, (255, 0, 0), 2)  # Blue = rejected for shape
                cv2.putText(screenImg, f"C:{circularity:.2f}", tuple(cnt[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        golden_cookies.append((cX, cY))

        if debug:
            cv2.drawContours(screenImg, [cnt], -1, (0, 255, 0), 2)  # Green = accepted
            cv2.circle(screenImg, (cX, cY), 5, (0, 255, 255), -1)
            cv2.putText(screenImg, f"C:{circularity:.2f}", (cX + 10, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 1)

    if debug:
        cv2.imwrite("golden_cookie_blobs_debug.png", screenImg)
        cv2.imwrite("golden_mask_debug.png", mask)


    return golden_cookies

#This locates the area for ocr based off the pattern mached name of the building
def compute_ocr_regions_shop(bbox,
                             number_width=80,        # ← new parameter
                             suffix_width=80,
                             suffix_pad=5,
                             number_height=20,
                             number_pad=2):
    left, top, right, bottom = bbox

    # Position the number region *below* the detected box
    number_top = bottom + number_pad
    number_bottom = number_top + number_height
    number_left = left
    number_right = number_left + number_width   # ← use custom width

    # Position the suffix region to the right of the number region
    suffix_left = number_right + suffix_pad
    suffix_right = suffix_left + suffix_width
    suffix_top = number_top
    suffix_bottom = number_bottom

    number_region = (number_left, number_top, number_right, number_bottom)
    suffix_region = (suffix_left, suffix_top, suffix_right, suffix_bottom)

    return number_region, suffix_region








class Building():
    def __init__(self, name, match_template, screen_image):

        self.name = name
        self.click_pos = None
        self.cost = float('inf')

        # OCR-related
        self.match_template = match_template
        self.screen_image = screen_image
        self.number_region = None
        self.suffix_region = None
        self.filter = AdaptiveMeanFilter()

        self.update_ocr_regions(screen_image)

    def update_ocr_regions(self, screen_img):
        # Find pointer location
        bbox = findBestScaledMatch_bbox(screen_img, self.match_template)
        if bbox == (0, 0, 0, 0):
            print(f"[WARN] Could not find template for {self.name}")
            return

        # Compute OCR regions
        number_region, suffix_region, *_ = compute_ocr_regions_shop(
            bbox,
            suffix_width=100,
            suffix_pad=1,
            number_height=18,
            number_pad=1
        )

        self.number_region = number_region
        self.suffix_region = suffix_region

    def read_current_cost(self, debug=True, screen_img=None, debug_output_dir="debug"):
        if self.number_region is None or self.suffix_region is None:
            print(f"[OCR] Regions not set for {self.name}, skipping read.")
            return self.filter.last()

        try:
            raw_value = interpret_number_with_suffix(self.number_region, self.suffix_region, debug=debug)
            print(f"[Raw building Cost] {raw_value}")

            if raw_value is None:
                print("[OCR] Falling back to last known value")
                return self.filter.last()

            print(f"[OCR] Cost before filter: {raw_value:.2f}")
            cost = self.filter.filter(raw_value)
            self.cost = cost

            # === DEBUG DRAWING TO FILE ===
            if debug and screen_img is not None:
                os.makedirs(debug_output_dir, exist_ok=True)
                debug_img = draw_debug_regions(
                    screen_img,
                    [self.number_region, self.suffix_region],
                    labels=["Number", "Suffix"]
                )
                debug_path = os.path.join(debug_output_dir, f"{self.name}_ocr_debug.png")
                cv2.imwrite(debug_path, debug_img)
                print(f"[DEBUG] Saved OCR region debug image to {debug_path}")

            return cost

        except Exception as e:
            print(f"[OCR ERROR] Exception in Cost read: {e}")
            return self.filter.last()







def update_Frame():
    # Take screenshot using PyAutoGUI
    screenshot = pyautogui.screenshot()
    # Convert screenshot to OpenCV format
    screenImg = np.array(screenshot)
    screenImg = cv2.cvtColor(screenImg, cv2.COLOR_RGB2BGR)
    return screenImg




def append_cookie_log(timestamp, cookie_count, filename):

    if cookie_count == 0.0:
        print(f"[LOG SKIP] Ignored 0.0 cookie count at {timestamp}")
        return  # Skip logging

    file_exists = os.path.isfile(filename) and os.path.getsize(filename) > 0
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            # File does not exist or is empty → write header first
            writer.writerow(["Timestamp", "Cookies"])
        writer.writerow([timestamp.isoformat(), cookie_count])

def log_event(message, timestamp=None, filename="event_log.csv"):
    if timestamp is None:
        timestamp = datetime.datetime.now()
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp.isoformat(), message])


def human_format(num, pos=None):
    """
    Formats a number like:
    1_000_000 -> 1M
    1_000_000_000 -> 1B
    1_000_000_000_000 -> 1T
    etc.
    """
    magnitude = 0
    original_num = num
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    suffixes = ['', 'K', 'M', 'B', 'T', 'P', 'E']
    # Format with one decimal if needed
    if num % 1 == 0:
        formatted = f'{int(num)}{suffixes[magnitude]}'
    else:
        formatted = f'{num:.1f}{suffixes[magnitude]}'
    return formatted


def draw_debug_regions(img, regions, labels=None, thickness=2):
    """
    Draws labeled rectangles on an image.
    `regions` is a list of (left, top, right, bottom) tuples.
    `labels` is an optional list of strings.
    """
    img_copy = img.copy()

    # Use distinct colors for each box if multiple
    base_colors = [
        (0, 255, 0),   # Green
        (255, 0, 0),   # Blue
        (0, 255, 255), # Yellow
        (255, 0, 255), # Magenta
        (0, 128, 255), # Orange
    ]

    for i, region in enumerate(regions):
        left, top, right, bottom = map(int, region)
        color = base_colors[i % len(base_colors)]
        cv2.rectangle(img_copy, (left, top), (right, bottom), color, thickness)

        if labels and i < len(labels):
            label = labels[i]

            # Draw background box for label
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_w, label_h = text_size
            label_bg_top_left = (left, top - label_h - 6)
            label_bg_bottom_right = (left + label_w + 4, top)

            cv2.rectangle(img_copy, label_bg_top_left, label_bg_bottom_right, color, -1)  # Filled label background
            cv2.putText(img_copy, label, (left + 2, top - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 2, cv2.LINE_AA)  # Black text for contrast

    return img_copy

def draw_debug_points(img, points, labels=None, color=(0, 255, 0), radius=10, thickness=3, font_scale=0.7, font_thickness=2):
    """
    Draw points with optional labels on an image.

    Args:
        img (numpy.ndarray): The image to draw on.
        points (list of tuples): List of (x, y) coordinates for points.
        labels (list of str, optional): Labels for each point. If None, no labels are drawn.
        color (tuple): BGR color of points and text.
        radius (int): Radius of the circles.
        thickness (int): Thickness of the circle outline.
        font_scale (float): Scale of the label font.
        font_thickness (int): Thickness of the label font.

    Returns:
        numpy.ndarray: Image with points (and labels) drawn.
    """
    img_out = img.copy()

    for i, point in enumerate(points):
        cv2.circle(img_out, point, radius, color, thickness)
        if labels and i < len(labels):
            text_pos = (point[0] + radius + 5, point[1] + radius // 2)  # offset text right/down
            cv2.putText(img_out, labels[i], text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)

    return img_out

##Progrram start
T = 3600



#Load the image(s) from file

#The main large cookie image
template = cv2.imread('Images/cropped_cookie.png')

#the "cookie" word which defines the OCR location for cookie count
word_template = cv2.imread('Images/cookies_word.png')


#These two are used to find the store spacing
cursor_building = cv2.imread('Images/Buildings/cursor.png')
grandma_building = cv2.imread('Images/Buildings/grandma.png')
#pointer_upgrade_Image = cv2.imread('Images/pointer_Upgrade.png')
#pointer_upgrade_1_Image = cv2.imread('Images/pointer_Upgrade_1.png')

print('Images loaded! \n')

output_dir = "Logs"
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "cookie_log.csv")

#Welcome Message
response = pyautogui.confirm(text='Welcome to CookieBotV2 \nTo start, ensure the big cookie is viable.', title='CookieBotV3', buttons=['Start', 'Close'])
if response == 'Close':
     quit()


#inital frame
screenImg = update_Frame()

#Finding the big cookie, basis for all game
large_cookie_center_pos = findBestScaledMatch_center(screenImg=screenImg, template=template)

if large_cookie_center_pos == (0, 0):
    print("Error: Big Cookie center not found. Closing Bot.")
    sys.exit()  # Exit the script safely



#this click focuses the window on the cookie clicker
click(large_cookie_center_pos)


#once focused take a frame
frame = update_Frame()


#Find the starting cookie count region based on the "cookies" word
cookie_word_region = findBestScaledMatch_bbox(screenImg, word_template)

#Locate the various read regions based off that reading
number_region, suffix_region, cps_number_region, cps_suffix_region = compute_ocr_regions_cookie_count(
    cookie_word_region,
    suffix_width=200,     # ← You can adjust this as needed
    suffix_pad=3,
    number_height=40,
    number_pad=3,
    cps_height=15,
    cps_number_width=45,
    cps_suffix_width=115,
    cps_suffix_pad=1


)







# Debugging image for startup regions and positions
regions = [cookie_word_region, number_region, suffix_region, cps_number_region, cps_suffix_region]
labels = ['Cookies Box', 'Number Region', 'Suffix Region', "CPS number", "CPS suffix", "Cost Number","Cost Suffix"]

debug_img = draw_debug_regions(frame, regions, labels=labels)
debug_img = draw_debug_points(debug_img, [large_cookie_center_pos], labels=["Big Cookie"])




# Save or display the debug image
cv2.imwrite('debug_startup.png', debug_img)
print("Saved debug_startup.png with OCR regions and points drawn")


#Now that regions are properly defined lets use them

#filter for properly tracking cookie bank
cookie_filter = AdaptiveMeanFilter()
#initial cookie reading
initial_cookie_count = read_filtered_value(number_region, suffix_region, cookie_filter)
if(initial_cookie_count):
    print(f"[DEBUG] Initial cookie count: {initial_cookie_count:.2f}")
else:
    print("[DEBUG] Initial cookie count could not be computed")


#filter for cps
cps_filter = AdaptiveMeanFilter()
#trying this with cps filter
initial_cps_amount = read_filtered_value(cps_number_region, cps_suffix_region, cps_filter, debug=False)
if(initial_cps_amount):
    print(f"[DEBUG] Initial CPS reading: {initial_cps_amount:.2f}")
else:
    print("[DEBUG] Initial CPS could not be computed")




#this is to account for luckmaxing which is 100mins of CPS**** (CPS * 60 *100)
if initial_cookie_count is not None and initial_cps_amount is not None:
    #We calculate the lucky max
    luckmax_threshold = initial_cps_amount * 60 * 100 # Stores the value above "lucky-maxing"
    spendable_cookie_count = initial_cookie_count - luckmax_threshold

    if(spendable_cookie_count > 0):{
        print("Above luckmax threshold")
        }
    print(f"[DEBUG] Spendable Cookies: {spendable_cookie_count:.2f}")
else:
    print("[DEBUG] Initial CPS count could not be computed")



# Setup (run once)
cursor_building = Building(name="Cursor",match_template=cv2.imread("Images/Buildings/cursor.png"), screen_image=frame)

cursor_building.read_current_cost(debug=True, screen_img=frame, debug_output_dir="debug")
cursor_cost = cursor_building.cost
print(f"Building Cost: {cursor_cost:.2f}")


# Read cost and update
#cursor_building.read_current_cost(debug=True)
#print(f"{cursor_building.name} cost: {cursor_building.current_cost:.2f}")




#farm = Building('farm',8, 1100, (previousStorePos[0], previousStorePos[1]+ offset ))
#previousStorePos = farm.clickPos

#mine = Building('mine',47, 12000, (previousStorePos[0], previousStorePos[1]+ offset ))
#previousStorePos = mine.clickPos

#factory = Building('factory',260, 130000, (previousStorePos[0], previousStorePos[1]+ offset ))
#previousStorePos = factory.clickPos

#bank = Building('bank',1400, 1400000, (previousStorePos[0], previousStorePos[1]+ offset ))
#previousStorePos = bank.clickPos

#temple = Building('temple', 7800, 12000000, (previousStorePos[0], previousStorePos[1]+ offset ))
#previousStorePos = temple.clickPos

#wizard_Tower = Building('wizard tower', 44000, 330000000, (previousStorePos[0], previousStorePos[1]+ offset ))
#previousStorePos = temple.clickPos



OCR_INTERVAL = 15  # Only do OCR every 15 frames (adjust as needed)
frames_since_last_ocr = OCR_INTERVAL  # Tracks how many frames since last OCR

COOKIE_SEARCH_INTERVAL = 3  # Only do cookie search (adjust as needed)
frames_since_cookie_search = COOKIE_SEARCH_INTERVAL  # Tracks how many frames since last OCR



#Uptime Counter
uptime_start = time.time()

#Cookie Logging
log_event("Bot Started")
cookie_log_data = []
last_logged_time = datetime.datetime.min # Track last valid timestamp



while keyboard.is_pressed('q') == False:

        #Per-frame variables
        frameStartTime = time.time()

        #updating the frame image
        frame = update_Frame()

        frames_since_last_ocr += 1  # Count up each frame
        frames_since_cookie_search += 1  # Count up each frame


        #Frame Logic Start

        #Search for golden cookie
        if frames_since_cookie_search >= COOKIE_SEARCH_INTERVAL:
            golden_cookies = find_golden_cookie_blobs(frame)
            if golden_cookies:
                print(f"Found {len(golden_cookies)} golden cookie(s): {golden_cookies}")
                for pos in golden_cookies:
                    pyautogui.click(pos)
                    log_event("Golden Cookie Clicked")
            else:
                print("No golden cookies found.")

            frames_since_cookie_search = 0  # Reset cookie search timer




        # Cookie Count Reading
        if frames_since_last_ocr >= OCR_INTERVAL:

            cookie_count = read_filtered_value(number_region, suffix_region, cookie_filter, debug=False)

            print(f"Cookie count: {cookie_count}")

            frames_since_last_ocr = 0  # Reset OCR frame timer



        #Cookie bank graphing

        timestamp = datetime.datetime.now()

        #Ensuring the time makes sense
        if timestamp > last_logged_time:
            append_cookie_log(timestamp, cookie_count, csv_path)
            last_logged_time = timestamp




        #For the rest of the frame we click big cookie
        frameCurrentTime = time.time()
        frameTime = (frameCurrentTime - frameStartTime)

        while(frameTime < 1):
             #For the rest of this frame, click the cookie

             pyautogui.doubleClick(large_cookie_center_pos)



             #check the frame time aiming for 1s frames
             frameCurrentTime = time.time()
             frameTime = (frameCurrentTime - frameStartTime)

        current_Time = time.time()
        current_Runtime = (current_Time - uptime_start)
        frameTime = (current_Time - frameStartTime)






quitTime = time.time()
delta_Uptime = round(quitTime - uptime_start)
log_event("Bot Stopped")
# Read cookie data
df = pd.read_csv(csv_path, parse_dates=["Timestamp"])
df = df[df["Cookies"] > 0.0]  # Optional: filter out zero cookie counts

# Read event data
event_df = pd.read_csv("event_log.csv", names=["Timestamp", "Event"], parse_dates=["Timestamp"])

# Define color mapping for known events
event_colors = {
    "Bot Started": "blue",
    "Bot Stopped": "red",
    "Golden Cookie Clicked": "gold",
}

# Find most recent "Bot Started" event
start_times = event_df[event_df["Event"] == "Bot Started"]["Timestamp"]
if start_times.empty:
    print("No 'Bot Started' event found.")
    exit()

last_start = start_times.max()

# Find the first "Bot Stopped" after that start (if any)
stop_times = event_df[(event_df["Event"] == "Bot Stopped") & (event_df["Timestamp"] > last_start)]["Timestamp"]
last_stop = stop_times.min() if not stop_times.empty else df["Timestamp"].max()

# Filter cookie data to only include the current bot run
df = df[(df["Timestamp"] >= last_start) & (df["Timestamp"] <= last_stop)]

# Create plot and get axis
fig, ax = plt.subplots()

# Plot cookie count
ax.plot(df["Timestamp"], df["Cookies"], label="Cookies")

# Add event markers
for _, row in event_df.iterrows():
    if row["Timestamp"] < last_start or row["Timestamp"] > last_stop:
        continue

    color = event_colors.get(row["Event"], "red")  # Default to red for unknown events
    ax.axvline(x=row["Timestamp"], color=color, linestyle="--", alpha=0.6)

    label = "G" if row["Event"] == "Bot Started" else ("S" if row["Event"] == "Bot Stopped" else row["Event"])
    ax.text(row["Timestamp"], df["Cookies"].max() * 1.01, label,
            rotation=90, fontsize=6, ha='right', va='bottom', color=color)

ax.set_xlabel("Time")
ax.set_ylabel("Cookies")
ax.set_title("Cookie Bank Over Time")
ax.legend()
ax.grid(True)
fig.autofmt_xdate()
plt.subplots_adjust(bottom=0.2, top=0.9)

# Set log scale on y-axis
ax.set_yscale('log')
ax.set_ylabel("Cookies (log scale)")

# Set custom human-readable formatter using your function
formatter = mticker.FuncFormatter(human_format)
ax.yaxis.set_major_formatter(formatter)


# Logarithmic scaling for Cookie Count
plt.yscale("log")
plt.ylabel("Cookies (log scale)")

# Dynamically compute min and max, add padding to max
min_cookies = max(1, df["Cookies"].min())
max_cookies = df["Cookies"].max() * 1.2  # 20% padding so line doesn't hug top

# Calculate nice log ticks covering full data range
log_min = int(np.floor(np.log10(min_cookies)))
log_max = int(np.ceil(np.log10(max_cookies)))
ticks = [10 ** i for i in range(log_min, log_max + 1)]

plt.yscale("log")
plt.ylabel("Cookies (log scale)")

# Dynamically calculate good log-scale tick marks
min_cookies = max(1, df["Cookies"].min())
max_cookies = df["Cookies"].max()
log_min = int(np.floor(np.log10(min_cookies)))
log_max = int(np.ceil(np.log10(max_cookies)))
ticks = [10 ** i for i in range(log_min, log_max + 1)]

plt.gca().set_yticks(ticks)
plt.gca().yaxis.set_major_formatter(formatter)

# Layout tweaks to avoid overlapping labels
plt.subplots_adjust(bottom=0.2, top=0.9)

# Finally, show the plot
plt.show()

print("Cookie Bot quitting...")











