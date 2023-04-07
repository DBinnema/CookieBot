from pyautogui import *
import pyautogui
import time
import keyboard
import random
import win32api, win32con


# To get RGB and mouse pos run 
# import pyautogui
# pyautogui.distplayMousePosition()
# will do a running display on color and position

#Peforming a click
#pyautogui.click(x=100, y=100)
#clicks left mouse at coords
#
#win32api see "def click"

#
#windows shift s to capture a screen shape

def click(x,y):
    win32api.SetCursorPos((x,y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0)
    time.sleep(0.01)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0)

print("Cookie Bot is starting...")


#Locate the cookie
cookie_Click_Point = (265,486)

cursor_Can_Buy = False
cursor_Click_Point = (1636, 433)


#Bot Loop
while keyboard.is_pressed('q') == False:

    if pyautogui.locateOnScreen("cookie.PNG", confidence=0.55) != None:

        print("Cookie on screen")

        #Checking Game State...
        if pyautogui.locateOnScreen("cursor.PNG", confidence=0.8) != None:
            click(cursor_Click_Point[0], cursor_Click_Point[1])
            print("Buying Cursor")
            

        

        #Click the cookie
        click(cookie_Click_Point[0], cookie_Click_Point[1])
        


        time.sleep(0.1)

    else:
        print("Cannot Locate Cookie")

print("Cookie Bot quitting...")







