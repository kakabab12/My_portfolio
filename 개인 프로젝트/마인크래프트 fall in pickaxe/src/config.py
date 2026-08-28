import json
import os
import shutil

# If config.json does not exist, copy from default.config.json
if not os.path.exists("config.json") and os.path.exists("default.config.json"):
    shutil.copy("default.config.json", "config.json")

if not os.path.exists("config.json"):
    raise FileNotFoundError("Configuration file 'config.json' not found. Please create it or copy from 'default.config.json' and make sure the name is correct.")

# Load configuration from config.json
with open("config.json", "r") as config_file:
    config = json.load(config_file)