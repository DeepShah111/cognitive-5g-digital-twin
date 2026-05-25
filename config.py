# config.py
import os

# Target Area: We use a specific central point in BKC and a 1000-meter radius
TARGET_ADDRESS = "Bandra Kurla Complex, Mumbai, India"
RADIUS_METERS = 1000

# Project Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Automatically create a 'data' folder
os.makedirs(DATA_DIR, exist_ok=True)