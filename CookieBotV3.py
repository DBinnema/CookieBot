import pyautogui
import time
import datetime
import keyboard
import random
import win32api, win32con
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import re
import tkinter as tk
import threading
SUFFIXES = {
    "million": 1_000_000,
    "billion": 1_000_000_000,
    "trillion": 1_000_000_000_000,
    "quadrillion": 1_000_000_000_000_000,
    "quintillion": 1_000_000_000_000_000_000
}

from PIL import ImageGrab
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'





def get_cookie_count(region=(0, 0, 1400, 170)):

    img = np.array(ImageGrab.grab(bbox=region))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    cv2.imwrite("debug_cookie_area.png", thresh)


    text = pytesseract.image_to_string(thresh, config='--psm 6 -c tessedit_char_whitelist=0123456789.,')
    print(f"[OCR Raw] {repr(text)}")

    match = re.search(r'[\d,]+(?:\.\d+)?', text)
    return float(match.group().replace(',', '')) if match else 0.0
        
def init_cookie_count_read(cookie_pos):
    cx, cy = cookie_pos

    # These offsets depend on your screen/UI layout
    
    width = 500
    height = 80
    
    shift_x = 200
    shift_y = 10
    # Calculate top-left corner based on bottom-right point
    left = cx - width + shift_x
    top = cy - height + shift_y
    right = cx + shift_x
    bottom = cy + shift_y

    # Define region (left, top, right, bottom)
    return (left, top, right, bottom)
    
    



def click(cookie_center_pos):
     pyautogui.moveTo(x=cookie_center_pos[0], y=cookie_center_pos[1])
     time.sleep(0.07)
     pyautogui.click(x=cookie_center_pos[0], y=cookie_center_pos[1])
     time.sleep(0.07)
    
def debugTemplateMatching(screenImg, template):
    result = cv2.matchTemplate(screenImg, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    print(f"Min: {min_val}, Max: {max_val}, Max Location: {max_loc}")

    result_display = result.copy()
    result_display = cv2.normalize(result_display, None, 0, 255, cv2.NORM_MINMAX)
    result_display = np.uint8(result_display)
    cv2.imwrite("debug_result_heatmap.png", result_display)
    
def getCenterPosition(screenImg, template, ):

    #Greyscale both images for better matching
    screen_gray = cv2.cvtColor(screenImg, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    #The actual search function, returns an image where white pixels are th best match
    correlation = cv2.matchTemplate(screen_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    
    #this gets the whitest and darkest pixels on the result image
    min_value, max_value, min_location, max_location =  cv2.minMaxLoc(correlation)


    print('Confidence: %s' % max_value)

    confidence_Threshold = 0.8
    
    if max_value < confidence_Threshold:
        print('Did not find template.')
        return (0,0)
    else:
        print('Found template with %s confidence.' % max_value)
        #Setting the click point
        top_left = max_location
        half_w = template.shape[1] //2
        half_h = template.shape[0] //2 
        large_cookie_center_pos = top_left[0] + half_w, top_left[1] + half_h
    
    #Setting the click point
    top_left = max_location
    half_w = template.shape[1] //2
    half_h = template.shape[0] //2 
    large_cookie_center_pos = top_left[0] + half_w, top_left[1] + half_h
    return large_cookie_center_pos
    
    #Better matching for finding things that may be different scale than the template images
def findBestScaledMatch(screenImg, template, scale_range=(0.8, 1.2), scale_step=0.05, threshold=0.8):
    screen_gray = cv2.cvtColor(screenImg, cv2.COLOR_BGR2GRAY)
    template_gray_orig = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    best_match = (0, (0, 0), None)  # (max_val, location, best_template)
    for scale in np.arange(scale_range[0], scale_range[1] + scale_step, scale_step):
        resized_template = cv2.resize(template_gray_orig, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        if resized_template.shape[0] > screen_gray.shape[0] or resized_template.shape[1] > screen_gray.shape[1]:
            continue  # Skip if the template is larger than the screen

        result = cv2.matchTemplate(screen_gray, resized_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        #print(f"Scale: {scale:.2f}, Confidence: {max_val}")


        if max_val > best_match[0]:
            best_match = (max_val, max_loc, resized_template)
            if max_val > 0.95:
                print("Found good match, breaking search")

                break  # Very confident match, no need to check further

    print(f"Best scale match confidence: {best_match[0]}, location: {best_match[1]}")

    if best_match[0] < threshold:
        print("Did not find template at any scale.")
        return (0, 0)

    top_left = best_match[1]
    template_used = best_match[2]
    h, w = template_used.shape[:2]
    return (top_left[0] + w // 2, top_left[1] + h // 2)

#Different detection methods for golden cookies
def find_golden_cookie_blobs(screenImg, min_area=1200, max_area=6000, debug=False):
    hsv = cv2.cvtColor(screenImg, cv2.COLOR_BGR2HSV)

    # Gold-like color range — adjust to taste
    lower_gold = np.array([15, 80, 120])  # broader hue, lower saturation threshold
    upper_gold = np.array([40, 255, 255])

    # Create mask for gold
    mask = cv2.inRange(hsv, lower_gold, upper_gold)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    golden_cookies = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                golden_cookies.append((cX, cY))

                if debug:
                    cv2.drawContours(screenImg, [cnt], -1, (0, 255, 0), 2)
                    cv2.circle(screenImg, (cX, cY), 5, (255, 0, 0), -1)

    if debug:
        cv2.imwrite("golden_cookie_blobs_debug.png", screenImg)
        cv2.imwrite("golden_mask_debug.png", mask)

    return golden_cookies


#this is for creating a template, used in finding golden cookies quickly
def load_template_and_mask(template_path):
    template = cv2.imread(template_path)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Convert to HSV for mask creation
    hsv_template = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)

    # Define range and create binary mask
    lower_gold = np.array([15, 130, 130])
    upper_gold = np.array([45, 255, 255])
    mask_color = cv2.inRange(hsv_template, lower_gold, upper_gold)

    print(template_gray.shape, mask_color.shape)

    # Resize and convert to grayscale match inputs
    return template, mask_color


def findMaskedTemplateMatch(screenImg, template, mask, threshold=0.65, mean_gap=0.05):

    #greyscale the screen image
    screen_gray = cv2.cvtColor(screenImg, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    cv2.imwrite("screen_image_grey.png", screen_gray)

    result = cv2.matchTemplate(screen_gray, template_gray, cv2.TM_CCOEFF_NORMED, mask=mask)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    mean_val = np.mean(result)

    print(f"[DEBUG] Max: {max_val:.3f}, Mean: {mean_val:.3f}")

    # Save debug image with rectangle even if no match
    debug_img = screenImg.copy()
    h, w = template.shape[:2]
    cv2.rectangle(debug_img, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 0, 255), 2)
    cv2.imwrite("debug_frame_match_overlay.png", debug_img)

    if max_val >= threshold and (max_val - mean_val) > mean_gap:
        return (max_loc[0] + w // 2, max_loc[1] + h // 2), max_val

    return None, max_val








class Shop_Item():
     name = ''
     amount = 0
     base_cost = 0
     base_cps = 0.0

     current_cost = 0.0
     current_roi = 0

     def getTotalCpS(self):
          
          return 0
         

    
     def __init__(self, name, base_cost):
          self.name = name          
          self.base_cost = base_cost 

          self.current_roi = self.current_cost/self.base_cps

     def getClickPosition(self):
        return 

     
     

     def __str__(self):
          return f'{self.name}'
     
class Building(Shop_Item):
     def __init__(self, name, base_cps, base_cost, clickPos):
          self.name = name
          self.amount = 0
          self.base_cps = base_cps
          self.base_cost = base_cost
          self.clickPos = clickPos
          self.multiplier = 1
          
          
          self.current_cost = base_cost
          self.current_roi = self.current_cost/self.base_cps

     def addAmount(self):
        self.amount += 1        
        self.current_cost = int(self.base_cost * 1.15**(self.amount))
        self.current_roi = self.current_cost/self.base_cps

     def addMultiplier(self, multiplier):
          self.multiplier *= multiplier
     
    
     def getTotalCpS(self):
          CpS = (self.base_cps * self.amount) * (self.multiplier * 1)
          
          return CpS

     def getClickPosition(self):
        return self.clickPos
     
     def getROI(self):
          return self.current_roi
     
class Upgrade(Shop_Item):
    
     
     def __init__(self, name, base_cost, upgrade_Image, building, multiplier):
          self.name = name                
          self.base_cost = base_cost
          

          self.upgradeImage = upgrade_Image
          self.building = building
          self.multiplier = multiplier
     
          
          
          
          self.current_cost = base_cost
          
     def getTotalCpS(self):
          
          return 0
     

     def getROI(self):
          roi = self.base_cost/(self.building.amount * self.building.base_cps * self.building.multiplier)
          
          return roi
     
     def addAmount(self):
          #this means the upgrade has been bought
          self.building.addMultiplier(self.multiplier)

          #hacky Solution so upgrades never get bought twice
      
          self.base_cost = math.inf


          return


     def getClickPosition(self):
        shop_Frame = update_Frame()
       
        return getCenterPosition(shop_Frame, self.upgradeImage)
    
          

    
def update_Frame():        
    # Take screenshot using PyAutoGUI
    screenshot = pyautogui.screenshot()
    # Convert screenshot to OpenCV format
    screenImg = np.array(screenshot)
    screenImg = cv2.cvtColor(screenImg, cv2.COLOR_RGB2BGR)
    return screenImg

def getBestROI(building_list):
     bestBuilding = building_list[0]
     bestCpS = building_list[0].getROI()
     for element in building_list:
          print('---------------------\n%s RIO: %s s\n---------------------' %(element.name, element.getROI()))         
          if (element.getROI() < bestCpS):
               bestBuilding=element
               bestCpS = element.getROI()
     return bestBuilding

def getCurrentCpS(building_list):
     printToLog('cps Breakdown\n')
     CpS = 0
     for element in building_list:

          thisCpS = element.getTotalCpS()
          CpS += thisCpS
          printToLog('cps from %s: %s' %(element.name, thisCpS))
          
          
     return CpS
     

def recordCookieCount(cookiecount, time , cps):
     file1 = open("point_data.txt", "a")  # append mode
     file1.write('%s, %s, %s\n' % (time, cookiecount, cps))
     file1.close    

def printToLog(printstring):
    file1 = open("log.txt", "a")  # append mode
    file1.write('%s\n' % printstring)
    file1.close()


            
def upgradeFunc(value, multiplier):
     value *= multiplier

def inspect_hsv_values(image_path):
     import cv2
     import numpy as np

     img = cv2.imread(image_path)
     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

     def mouse_event(event, x, y, flags, param):
         if event == cv2.EVENT_MOUSEMOVE:
             pixel = hsv[y, x]
             print(f"HSV at ({x},{y}): {pixel}")

     cv2.imshow("Inspect HSV", img)
     cv2.setMouseCallback("Inspect HSV", mouse_event)
     cv2.waitKey(0)
     cv2.destroyAllWindows()


##Progrram start
T = 3600
#inspect_hsv_values("golden_cookie_blobs_debug.png")


#Load the image(s) from file

#The main large cookie image
template = cv2.imread('Images/cropped_cookie.png')
word_template = cv2.imread('Images/cookies_word.png')

# goldent cookie template loaded
golden_cookie_template, golden_mask = load_template_and_mask('Images/golden_Cookie.png')


#These two are used to find the store spacing
cursor_building = cv2.imread('Images/Buildings/cursor_blackout.png')
grandma_building = cv2.imread('Images/Buildings/grandma_blackout.png')
pointer_upgrade_Image = cv2.imread('Images/pointer_Upgrade.png')
pointer_upgrade_1_Image = cv2.imread('Images/pointer_Upgrade_1.png')

print('Images loaded! \n')

#Welcome Message
respone = pyautogui.confirm(text='Welcome to CookieBotV2 \nTo start, ensure the big cookie is viable.', title='CookieBotV3', buttons=['Start', 'Close'])
if respone == 'Close':
     quit()

#inital frame
screenImg = update_Frame()

debugTemplateMatching(screenImg, template)

#Clearing records for new run
clear_file = open("log.txt", 'w') 
clear_file.close()

clear_file = open("point_data.txt", 'w') 
clear_file.close()

#Finding the big cookie, basis for all game
large_cookie_center_pos = findBestScaledMatch(screenImg=screenImg, template=template)

#Find the starting cookie count region
print('Looking for cookie word')
cookie_word_region = findBestScaledMatch(screenImg, word_template)

print(template is None)  # Should print False
#cv2.imshow("Template", template)
cv2.waitKey(0)
cv2.destroyAllWindows()


if large_cookie_center_pos == (0, 0):
    print("Error: Big Cookie center not found. Closing Bot.")
    sys.exit()  # Exit the script safely


#this click focuses the window on the cookie clicker 
click(large_cookie_center_pos)


#once focused take a frame 
frame = update_Frame()






#Find the starting store positions
cursor_Pos = findBestScaledMatch(screenImg=frame, template=cursor_building)
grandma_Pos = findBestScaledMatch(screenImg=frame, template=grandma_building)
cursor = Building('cursor', 0.1, 15, cursor_Pos)
grandma = Building('grandma', 1, 100, grandma_Pos)

#Now that the first two store has been found hopefully we can calculate the next store positions


offset = grandma.clickPos[1] - cursor.clickPos[1]
previousStorePos = grandma.clickPos

#Defining Buildings
farm = Building('farm',8, 1100, (previousStorePos[0], previousStorePos[1]+ offset ))
previousStorePos = farm.clickPos

mine = Building('mine',47, 12000, (previousStorePos[0], previousStorePos[1]+ offset ))
previousStorePos = mine.clickPos

factory = Building('factory',260, 130000, (previousStorePos[0], previousStorePos[1]+ offset ))
previousStorePos = factory.clickPos

bank = Building('bank',1400, 1400000, (previousStorePos[0], previousStorePos[1]+ offset ))
previousStorePos = bank.clickPos

temple = Building('temple', 7800, 12000000, (previousStorePos[0], previousStorePos[1]+ offset ))
previousStorePos = temple.clickPos

wizard_Tower = Building('wizard tower', 44000, 330000000, (previousStorePos[0], previousStorePos[1]+ offset ))
previousStorePos = temple.clickPos


#Defining Upgrades

pointer_0 = Upgrade('pointer_0_upgrade', 100, pointer_upgrade_Image, cursor, 2.0)
pointer_1 = Upgrade('pointer_1_upgrade', 500, pointer_upgrade_Image, cursor, 2.0)






active_building_list = [cursor]

#Index at 1 because we assume that cursor is already unlocked
unlock_index = 1
complete_building_list = [cursor, grandma, farm, mine, factory, wizard_Tower]

active_upgrade_list = [pointer_0, pointer_1]


#reading the screen tests

read_region = init_cookie_count_read(cookie_word_region)

cookie_count = get_cookie_count(read_region)
print(f"[DEBUG] Read Cookie Count: {cookie_count}")



#Uptime Counter
uptime_start = time.time()

while keyboard.is_pressed('q') == False:
        
        #Per-frame variables     
        frameStartTime = time.time()        
        cookies_gained_this_frame = 0
        
       
     

        if keyboard.is_pressed('b') == True:
             frame = update_Frame()
             store_Pos0 = getCenterPosition(screenImg=frame, template=pointer_upgrade_Image)
             click(store_Pos0)

            # calculate the cookie count
        cookie_count = get_cookie_count(read_region)
        print(f"[DEBUG] Read Cookie Count: {cookie_count}")



        #Search for golden cookie

        frame = update_Frame()
        golden_cookies = find_golden_cookie_blobs(frame, debug=True)
        if golden_cookies:
            print(f"Found {len(golden_cookies)} golden cookie(s): {golden_cookies}")
            for pos in golden_cookies:
                pyautogui.click(pos)
        else:
            print("No golden cookies found.")




        

        #Unlock the next building if we can afford it
        if unlock_index < len(complete_building_list) and cookie_count > complete_building_list[unlock_index].base_cost//3:
             #print('Unlocking %s.' %(complete_building_list[unlock_index].name))
             #printToLog('Unlocking %s.' %(complete_building_list[unlock_index].name))

             active_building_list.append(complete_building_list[unlock_index])
             unlock_index +=1

             #No longer needed as using an unlock_index
             ##upgrade_building_list = upgrade_building_list[1:]

             #nextPurchase = getBestROI(active_building_list)

        



     #For the rest of the frame we click big cookie
        frameCurrentTime = time.time()
        frameTime = (frameCurrentTime - frameStartTime)

        while(frameTime < 1.8):
             #For the rest of this frame, click the cookie
             
             pyautogui.doubleClick(large_cookie_center_pos)



             #check the frame time aiming for 1s frames
             frameCurrentTime = time.time()
             frameTime = (frameCurrentTime - frameStartTime)
        
        current_Time = time.time()
        current_Runtime = (current_Time - uptime_start)        
        frameTime = (current_Time - frameStartTime)






        ##Print a frame message for debugging


        #frameMessage = 'Frame took %s seconds. \nEstimated Cookies %s CpS: %s ' %(format(frameTime, "0.2f"), cookie_count )
      
        #print(frameMessage)

    
quitTime = time.time()
delta_Uptime = round(quitTime - uptime_start)
total_cookie_amount = round(total_cookie_amount,1)
print("Cookie Bot quitting...\nTotalGenerated: %s \nRuntime: %s"  %(total_cookie_amount, delta_Uptime ))


#Debugging
mapImg = update_Frame()
#Draw a circle for the click point
mapImg = cv2.circle(frame, large_cookie_center_pos, radius =5, color=(255,0,0), thickness=-7)
mapImg = cv2.circle(frame, grandma.clickPos, radius =5, color=(255,255,0), thickness=-1)
mapImg = cv2.circle(frame, cursor.clickPos, radius =5, color=(255,0,255), thickness=-1)
mapImg = cv2.circle(frame, pointer_0.getClickPosition(), radius =5, color=(255,255,255), thickness=3)

cv2.imwrite('QuitImages/GameEnd-%s.jpeg' %round(total_cookie_amount), mapImg)







#Draw a locating rectangle

import cv2

def drawRect(image, top_left, template_shape, color=(255, 0, 0), thickness=2):
    image_w, image_h = template_shape[1], template_shape[0]
    bottom_right = (top_left[0] + image_w, top_left[1] + image_h)

    cv2.rectangle(image, top_left, bottom_right, color, thickness, lineType=cv2.LINE_4)
    return image



