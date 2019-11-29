import win32api, win32con
import pyautogui, sys
import time 

#def click(x,y):
#    #win32api.SetCursorPos((x,y))
#    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN,x,y,0,0)
#    #time.sleep(0.5)
#    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP,x,y,0,0)
#    #win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN,0,0)
#    
#    #win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP,0,0)
time.sleep(3)
#click(1800,300)
pyautogui.mouseDown(button='right')
time.sleep(0.5)
pyautogui.mouseUp(button='right')

pyautogui.click(button='right')