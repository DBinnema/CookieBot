from pyautogui import *
import pyautogui
import time
import datetime
import keyboard
import random
import win32api, win32con
import cv2
import numpy as np
import math


def click(cookie_center_pos):
     pyautogui.moveTo(x=cookie_center_pos[0], y=cookie_center_pos[1])
     time.sleep(0.07)
     pyautogui.click(x=cookie_center_pos[0], y=cookie_center_pos[1])
     time.sleep(0.07)
    

    
    
def getclickposition(screenImg, template):

    #The actual search function, returns an image where white pixels are th best match
    correlation = cv2.matchTemplate(screenImg, template, cv2.TM_CCOEFF_NORMED)

    #this gets the whitest and darkest pixels on the result image
    min_value, max_value, min_location, max_location =  cv2.minMaxLoc(correlation)


    #print('Confidence: %s' % max_value)

    #confidence_Threshhold = 0.7
    #if max_value < confidence_Threshhold:
        #print('Did not find template.')
   # else:
       # print('Found template with %s confidence.' % max_value)
    
    #Setting the click point
    top_left = max_location
    half_w = template.shape[1] //2
    half_h = template.shape[0] //2 
    large_cookie_center_pos = top_left[0] + half_w, top_left[1] + half_h
    return large_cookie_center_pos


class Building:
     name = ''
     amount = 0
     base_cost = 0
     base_cps = 0.0

     current_cost = 0.0
     current_cps = 0
   
     

    
     def __init__(self, name, base_cps, base_cost, image):
          self.name = name
          self.amount = 0
          self.base_cps = base_cps
          self.base_cost = base_cost
          self.image = image
          self.current_cost = base_cost
          self.current_cps = self.current_cost/self.base_cps

     def addAmount(self):
        self.amount += 1        
        self.current_cost = int(self.base_cost * 1.15**(self.amount))
        self.current_cps = self.current_cost/self.base_cps

     def setClickPos(self, clickPos):
        self.clickPos = clickPos

     def __str__(self):
          return f'{self.name}'
   
          

    
def update_Frame():
        
    # Take screenshot using PyAutoGUI
    screenshot = pyautogui.screenshot()
    # Convert screenshot to OpenCV format
    screenImg = np.array(screenshot)
    screenImg = cv2.cvtColor(screenImg, cv2.COLOR_RGB2BGR)
    return screenImg

def getBestCpS(building_list):
     bestBuilding = building_list[0]
     bestCpS = building_list[0].current_cps
     for element in building_list:
          print('%s CpS: %s' %(element.name, element.current_cps))
          if (element.current_cps < bestCpS):
               bestBuilding=element
               bestCpS=element.current_cps
     return bestBuilding

def getCurrentCpS(building_list):
     CpS = 0
     for element in building_list:

          thisCpS = element.base_cps * element.amount
          CpS += thisCpS
          
     return CpS






        
    







##Progrram start

#Load the image(s) from file
template = cv2.imread('Images/cropped_cookie.png')

grandma_building = cv2.imread('Images/grandma_blackout.png')
cursor_building = cv2.imread('Images/cursor_blackout.png')

#Some info for starting logic
cursor = Building('cursor', 0.1, 15, cursor_building)
grandma = Building('grandma', 1, 100, grandma_building)
print('Images loaded! \n Starting Cookie Bot')


#inital frame
screenImg = update_Frame()

large_cookie_center_pos = getclickposition(screenImg=screenImg, template=template)

#this click focuses the window on the cookie clicker 
click(large_cookie_center_pos)

#once focused can find the buttons
frame = update_Frame()

cursor.setClickPos(getclickposition(screenImg=frame, template=cursor_building))
grandma.setClickPos(getclickposition(screenImg=frame, template=grandma_building))

active_building_list = [cursor]
upgrade_building_list = [grandma]




#assuming a fresh game
cookie_count = 0
nextPurchase = active_building_list[0]
currentCpS = 0

#Bot Loop
while keyboard.is_pressed('q') == False:
        #To time this loop (called frame)        
        frameStartTime = time.time()
        
        #Unlock the next building if we can afford it
        if upgrade_building_list and cookie_count > upgrade_building_list[0].base_cost//2:
             print('Unlocking %s.' %(upgrade_building_list[0].name))
             active_building_list.append(upgrade_building_list[0])
             upgrade_building_list = upgrade_building_list[1:]
             
             
              

        #If the Building we want can be afforded, purchase that
        if(cookie_count > nextPurchase.current_cost):                                
             click(nextPurchase.clickPos)

             #Trying to keep cookie count relitivly accurate
             cookie_count = cookie_count - nextPurchase.current_cost
             nextPurchase.addAmount()
             print('Buying %s. Now have %s %s.' %(nextPurchase.name, nextPurchase.amount,nextPurchase.name))
             nextPurchase = getBestCpS(active_building_list)
             currentCpS = getCurrentCpS(active_building_list)


        
        #
          #For the rest of the frame we click big cookie
        frameCurrentTime = time.time()
        frameTime = (frameCurrentTime - frameStartTime)

        while(frameTime < 0.9):
             #For the rest of this frame, click the cookie
             click(large_cookie_center_pos)
             cookie_count = cookie_count + 1
             #check the frame time aiming for 1s frames
             frameCurrentTime = time.time()
             frameTime = (frameCurrentTime - frameStartTime)
        
     
        frameCurrentTime = time.time()
        frameTime = (frameCurrentTime - frameStartTime)

        cookies_gained_this_frame = currentCpS * frameTime
        cookie_count += round(cookies_gained_this_frame)

        ##Print a frame message for debugging

        frameMessage = 'Frame took %s seconds. \nEstimated Cookies %s CpS: %s ' %(format(frameTime, "0.2f"), cookie_count, format(currentCpS, "0.2f"))

        print(frameMessage)

    

print("Cookie Bot quitting...")


#Debugging

#Draw a circle for the click point
mapImg = cv2.circle(frame, large_cookie_center_pos, radius =5, color=(255,0,0), thickness=-7)
mapImg = cv2.circle(frame, grandma.clickPos, radius =5, color=(255,255,0), thickness=-1)
mapImg = cv2.circle(frame, cursor.clickPos, radius =5, color=(255,0,255), thickness=-1)

cv2.imwrite('QuitImages/GameEnd-%s.jpeg' %currentCpS, mapImg)







#Draw a locating rectangle
"""
def drawRect():
    top_left = max_location

    image_w = template.shape[1]
    image_h =  template.shape[0]
    bottom_right = top_left[0] + image_w, top_left[1] + image_h

    mapImg = cv2.rectangle(mapImg, top_left, bottom_right, color=(255,0,0), thickness=2, lineType=cv2.LINE_4)
"""""

