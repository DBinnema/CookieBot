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
import matplotlib.pyplot as plt
import sys


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

    confidence_Threshhold = 0.8
    if max_value < confidence_Threshhold:
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


class Shop_Item():
     name = ''
     amount = 0
     base_cost = 0
     base_cps = 0.0

     current_cost = 0.0
     current_roi = 0
         

    
     def __init__(self, name, base_cost):
          self.name = name          
          self.base_cost = base_cost 

          self.current_roi = self.current_cost/self.base_cps

     def getClickPosition(self):
        return 

     def addAmount(self):
        self.amount += 1        
        self.current_cost = int(self.base_cost * 1.15**(self.amount))
        self.current_roi = self.current_cost/self.base_cps
     

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

     def addMultiplier(self, multiplier):
          self.multiplier += multiplier
        

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
          
          

     def getROI(self):
          roi = self.base_cost/(self.building.amount * self.building.base_cps * self.building.multiplier)
          
          return roi
     
     def addAmount(self):
          #this means the upgrade has been bought
          self.building.addMultiplier(self.multiplier)
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
     CpS = 0
     for element in building_list:

          thisCpS = element.base_cps * element.amount * element.multiplier
          CpS += thisCpS
          
     return CpS

def printToLog(printstring):
     file1 = open("log.txt", "a")  # append mode
     file1.write('%s\n' % printstring)
     file1.close    



            



##Progrram start

#Load the image(s) from file

#The main large cookie image
template = cv2.imread('Images/cropped_cookie.png')

#These two are used to find the store spacing
cursor_building = cv2.imread('Images/Buildings/cursor_blackout.png')
grandma_building = cv2.imread('Images/Buildings/grandma_blackout.png')
pointer_upgrade_Image = cv2.imread('Images/pointer_Upgrade.png')

print('Images loaded! \n')

#Welcome Message
respone = pyautogui.confirm(text='Welcome to CookieBotV2 \nTo start, ensure the big cookie is viable.', title='CookieBotV2', buttons=['Start', 'Close'])
if respone == 'Close':
     quit()


#inital frame
screenImg = update_Frame()

#preparing new log
clear_file = open("log.txt", 'w') 
clear_file.close()

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

#Defining Buildings
farm = Building('farm',8, 1100, (previousStorePos[0], previousStorePos[1]+ offset ))
previousStorePos = farm.clickPos

mine = Building('mine',47, 12000, (previousStorePos[0], previousStorePos[1]+ offset ))
previousStorePos = mine.clickPos

factory = Building('factory',260, 130000, (previousStorePos[0], previousStorePos[1]+ offset ))
previousStorePos = factory.clickPos


#Defining Upgrades

pointer_0 = Upgrade('pointer_0_upgrade', 100, pointer_upgrade_Image, cursor, 2)






active_building_list = [cursor, pointer_0]

#Index at 1 because we assume that cursor is already unlocked
unlock_index = 1
complete_building_list = [cursor, grandma, farm, mine, factory]






#assuming a fresh game
cookie_count = 0
total_cookie_amount = 0
nextPurchase = active_building_list[0]
currentCpS = 0

#The cookies to add on one click
clickvalue = 1

uptime = time.time()
#Bot Loop
frameCount = 0


while keyboard.is_pressed('q') == False:
        
        #Per-frame variables     
        frameStartTime = time.time()        
        cookies_gained_this_frame = 0

        if keyboard.is_pressed('b') == True:
             frame = update_Frame()
             store_Pos0 = getCenterPosition(screenImg=frame, template=pointer_upgrade_Image)
             click(store_Pos0)
        

        #Unlock the next building if we can afford it
        if complete_building_list and cookie_count > complete_building_list[unlock_index].base_cost//3:
             print('Unlocking %s.' %(complete_building_list[unlock_index].name))
             printToLog('Unlocking %s.' %(complete_building_list[unlock_index].name))

             active_building_list.append(complete_building_list[unlock_index])
             unlock_index +=1

             #No longer needed as using an unlock_index
             ##upgrade_building_list = upgrade_building_list[1:]

             nextPurchase = getBestROI(active_building_list)          

        
        #If the Item we want can be afforded, purchase that
        if(cookie_count - 3 > nextPurchase.current_cost):
                          
             #Apply a negative amount to cookie count to offset the pre buy cps
             frameBuy = time.time()
             pre_Buy_Time = (frameBuy - frameStartTime)
             pre_Buy_CpS = currentCpS

             click(nextPurchase.getClickPosition())

             

             #Trying to keep cookie count relitivly accurate
             cookie_count -= round(nextPurchase.current_cost)
             nextPurchase.addAmount()
             
             print('Buying %s. Now have %s %s.' %(nextPurchase.name, nextPurchase.amount,nextPurchase.name))             
             currentCpS = getCurrentCpS(active_building_list)
             nextPurchase = getBestROI(active_building_list)

             printToLog('Bought %s. Now have %s %s.' %(nextPurchase.name, nextPurchase.amount,nextPurchase.name))

             #Setting a negative amount for the time before this frame
             cookies_gained_this_frame -= currentCpS - pre_Buy_CpS


     #For the rest of the frame we click big cookie
        frameCurrentTime = time.time()
        frameTime = (frameCurrentTime - frameStartTime)

        while(frameTime < 0.8):
             #For the rest of this frame, click the cookie
             
             pyautogui.doubleClick(large_cookie_center_pos)

             #adding value to the appropriate counters
             cookie_count += clickvalue
             total_cookie_amount += clickvalue             

             #check the frame time aiming for 1s frames
             frameCurrentTime = time.time()
             frameTime = (frameCurrentTime - frameStartTime)
        
     
        frameEndTime = time.time()
        frameTime = (frameEndTime - frameStartTime)

        cookies_gained_this_frame += round(currentCpS * frameTime, 1)

        #Incrementing counters
        cookie_count += cookies_gained_this_frame
        total_cookie_amount += cookies_gained_this_frame
       

        ##Print a frame message for debugging


        frameMessage = 'Frame took %s seconds. \nEstimated Cookies %s CpS: %s ' %(format(frameTime, "0.2f"), cookie_count, format(currentCpS, "0.2f"))

        print(frameMessage)

    
quitTime = time.time()
delta_Uptime = round(quitTime - uptime)
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
"""
def drawRect():
    top_left = max_location

    image_w = template.shape[1]
    image_h =  template.shape[0]
    bottom_right = top_left[0] + image_w, top_left[1] + image_h

    mapImg = cv2.rectangle(mapImg, top_left, bottom_right, color=(255,0,0), thickness=2, lineType=cv2.LINE_4)
"""""

