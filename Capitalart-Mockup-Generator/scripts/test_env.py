# test_env.py
import cv2
import pytz
from datetime import datetime

print("✅ OpenCV version:", cv2.__version__)
print("🕒 Adelaide time:", datetime.now(pytz.timezone("Australia/Adelaide")).strftime('%Y-%m-%d %I:%M %p %Z'))
