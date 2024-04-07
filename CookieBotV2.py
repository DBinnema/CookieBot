from pyautogui import *
import pyautogui
import time
import keyboard
import random
import win32api, win32con
import cv2
import numpy as np

def click(x,y):
    win32api.SetCursorPos((x,y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0)
    time.sleep(0.01)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0)


#Progrram start
    
# Get the screen resolution
screen_width, screen_height = pyautogui.size()

# Take screenshot using PyAutoGUI
screenshot = pyautogui.screenshot()

# Convert screenshot to OpenCV format
screenImg = np.array(screenshot)
screenImg = cv2.cvtColor(screenImg, cv2.COLOR_RGB2BGR)


#Load the image(s) from file
template = cv2.imread('Images/cropped_cookie.png')

#The actual search function, returns an image where white pixels are th best match
correlation = cv2.matchTemplate(screenImg, template, cv2.TM_CCOEFF_NORMED)

#this gets the whitest and darkest pixels on the result image
min_value, max_value, min_location, max_location =  cv2.minMaxLoc(correlation)


print('Confidence: %s' % max_value)

confidence_Threshhold = 0.7
if max_value < confidence_Threshhold:
    print('Did not find large cookie.')
else:
    print('Found large cookie recording position.')
   
   #Setting the click point
    top_left = max_location
    half_w = template.shape[1] //2
    half_h = template.shape[0] //2 
    large_cookie_center_pos = top_left[0] + half_w, top_left[1] + half_h



#Bot Loop
while keyboard.is_pressed('q') == False:
        

        #Click the cookie
        click(large_cookie_center_pos[0], large_cookie_center_pos[1])
        print("Clicking Cookie")
        


        time.sleep(0.4)

print("Cookie Bot quitting...")





#Debugging

#Draw a circle for the click point
mapImg = cv2.circle(screenImg, large_cookie_center_pos, radius =5, color=(255,0,0), thickness=-1)

#Draw a locating rectangle
top_left = max_location

image_w = template.shape[1]
image_h =  template.shape[0]
bottom_right = top_left[0] + image_w, top_left[1] + image_h

mapImg = cv2.rectangle(mapImg, top_left, bottom_right,
               color=(255,0,0), thickness=2, lineType=cv2.LINE_4)


cv2.imshow('Cookie', mapImg)
cv2.waitKey(0)