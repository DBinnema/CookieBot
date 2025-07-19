



# Cookie Bot
Bot for cookie clicker: https://orteil.dashnet.org/cookieclicker/

A Python bot that automates gameplay in [Cookie Clicker](https://orteil.dashnet.org/cookieclicker/) using screen capture, OCR, and input automation.

---
##Instructions
1. Open a new cookie clicker game.
2. run cookie clicker with the maincookie visible in web browser 
3. cookie clicker will run untill 'q' is held
---

## Features

- Automatically clicks the big cookie
- Detects and clicks golden cookies
- Reads cookie count and CPS using OCR
- Adaptive filters to smooth OCR data
- Buys upgrades and buildings based on cost-efficiency
- Implements Lucky Max strategy for optimal spending
- Debug mode with labeled visual overlays

---

## Tech Stack

- Python 3
- OpenCV (image processing)
- PyAutoGUI (mouse/keyboard control)
- pytesseract (OCR)
- NumPy (data manipulation)
