from pyautogui import *
import pyautogui
import time
import keyboard
import random
import win32api, win32con


while 1:
    if pyautogui.locateOnScreen("cookie.PNG") != None:
        print("Found Cookie")
        time.sleep(0.05)

    else:
        print("No Cookie found")
        time.sleep(0.05)
