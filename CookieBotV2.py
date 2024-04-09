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
    

    
    
def getCenterPosition(screenImg, template):

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
     current_roi = 0    

    
     def __init__(self, name, base_cps, base_cost, clickPos):
          self.name = name
          self.amount = 0
          self.base_cps = base_cps
          self.base_cost = base_cost
          self.clickPos = clickPos
          
          
          self.current_cost = base_cost
          self.current_roi = self.current_cost/self.base_cps

     def addAmount(self):
        self.amount += 1        
        self.current_cost = int(self.base_cost * 1.15**(self.amount))
        self.current_roi = self.current_cost/self.base_cps

     

     def __str__(self):
          return f'{self.name}'
   
          

    
def update_Frame():
        
    # Take screenshot using PyAutoGUI
    screenshot = pyautogui.screenshot()
    # Convert screenshot to OpenCV format
    screenImg = np.array(screenshot)
    screenImg = cv2.cvtColor(screenImg, cv2.COLOR_RGB2BGR)
    return screenImg

def getBestROI(building_list):
     bestBuilding = building_list[0]
     bestCpS = building_list[0].current_roi
     for element in building_list:
          print('%s RIO: %s s' %(element.name, element.current_roi))
          if (element.current_roi < bestCpS):
               bestBuilding=element
               bestCpS=element.current_roi
     return bestBuilding

def getCurrentCpS(building_list):
     CpS = 0
     for element in building_list:

          thisCpS = element.base_cps * element.amount
          CpS += thisCpS
          
     return CpS

def getBestBuilding(building_list, currentCpS):
     bestBuilding = building_list[0]
     bestValue = building_list[0].current_roi

     for element in building_list:
          value = element.current_roi = (element.current_cost/currentCpS)
          print('%s RIO + save time: %s s' %(element.name, element.current_roi))
          if (value < bestValue):
               bestBuilding=element
               bestValue=value
     return bestBuilding






        
    







##Progrram start

#Load the image(s) from file

#The main large cookie image
template = cv2.imread('Images/cropped_cookie.png')

#These two are used to find the store spacing
cursor_building = cv2.imread('Images/Buildings/cursor_blackout.png')
grandma_building = cv2.imread('Images/Buildings/grandma_blackout.png')

print('Images loaded! \n')

#Welcome Message
respone = pyautogui.confirm(text='Welcome to CookieBotV2 \nTo start, ensure the big cookie is viable.', title='CookieBotV2', buttons=['Start', 'Close'])
if respone == 'Close':
     quit()


#inital frame
screenImg = update_Frame()

#Finding the big cookie, basis for all game
large_cookie_center_pos = getCenterPosition(screenImg=screenImg, template=template)

#this click focuses the window on the cookie clicker 
click(large_cookie_center_pos)


#once focused take a frame 
frame = update_Frame()

#Find the starting store positions
cursor_Pos = getCenterPosition(screenImg=frame, template=cursor_building)
grandma_Pos = getCenterPosition(screenImg=frame, template=grandma_building)
cursor = Building('cursor', 0.1, 15, cursor_Pos)
grandma = Building('grandma', 1, 100, grandma_Pos)

#Now that the first two store has been found hopefully we can calculate the next store positions

offset = grandma.clickPos[1] - cursor.clickPos[1]
previousStorePos = grandma.clickPos


farm = Building('farm',8, 1100, (previousStorePos[0], previousStorePos[1]+ offset ))
previousStorePos = farm.clickPos

mine = Building('mine',47, 12000, (previousStorePos[0], previousStorePos[1]+ offset ))
previousStorePos = mine.clickPos

factory = Building('factory',260, 1300, (previousStorePos[0], previousStorePos[1]+ offset ))
previousStorePos = factory.clickPos


active_building_list = [cursor]
upgrade_building_list = [grandma, farm, mine, factory]




#assuming a fresh game
cookie_count = 0
nextPurchase = active_building_list[0]
currentCpS = 0

#Bot Loop
while keyboard.is_pressed('q') == False:
        #To time this loop (called frame)        
        frameStartTime = time.time()
        cookies_gained_this_frame = 0

        
        #Unlock the next building if we can afford it
        if upgrade_building_list and cookie_count > upgrade_building_list[0].base_cost//3:
             print('Unlocking %s.' %(upgrade_building_list[0].name))
             active_building_list.append(upgrade_building_list[0])
             upgrade_building_list = upgrade_building_list[1:]
             nextPurchase = getBestROI(active_building_list)
             
             
              

        #If the Building we want can be afforded, purchase that
        if(cookie_count - 3 > nextPurchase.current_cost):
             
             #Apply a negative amount to cookie count to offset the pre buy cps
             frameBuy = time.time()
             pre_Buy_Time = (frameBuy - frameStartTime)
             pre_Buy_CpS = currentCpS

             click(nextPurchase.clickPos)

             #Trying to keep cookie count relitivly accurate
             cookie_count -= nextPurchase.current_cost
             nextPurchase.addAmount()
             print('Buying %s. Now have %s %s.' %(nextPurchase.name, nextPurchase.amount,nextPurchase.name))             
             currentCpS = getCurrentCpS(active_building_list)
             nextPurchase = getBestROI(active_building_list)

             cookies_gained_this_frame -= currentCpS - pre_Buy_CpS


        
        #
          #For the rest of the frame we click big cookie
        frameCurrentTime = time.time()
        frameTime = (frameCurrentTime - frameStartTime)

        while(frameTime < 0.8):
             #For the rest of this frame, click the cookie
             click(large_cookie_center_pos)
             cookie_count = cookie_count + 1
             #check the frame time aiming for 1s frames
             frameCurrentTime = time.time()
             frameTime = (frameCurrentTime - frameStartTime)
        
     
        frameEndTime = time.time()
        frameTime = (frameEndTime - frameStartTime)

        cookies_gained_this_frame += round(currentCpS * frameTime, 1)
        cookie_count += cookies_gained_this_frame

        ##Print a frame message for debugging

        frameMessage = 'Frame took %s seconds. \nEstimated Cookies %s CpS: %s ' %(format(frameTime, "0.2f"), cookie_count, format(currentCpS, "0.2f"))

        print(frameMessage)

    

print("Cookie Bot quitting...")


#Debugging

#Draw a circle for the click point
mapImg = cv2.circle(frame, large_cookie_center_pos, radius =5, color=(255,0,0), thickness=-7)
mapImg = cv2.circle(frame, grandma.clickPos, radius =5, color=(255,255,0), thickness=-1)
mapImg = cv2.circle(frame, cursor.clickPos, radius =5, color=(255,0,255), thickness=-1)
mapImg = cv2.circle(frame, farm.clickPos, radius =5, color=(255,255,255), thickness=3)

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

