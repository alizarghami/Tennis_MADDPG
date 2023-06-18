# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 12:07:16 2020

@author: Ali
"""

from tennis import Tennis


# please adjust the parameters below
env_path = "Tennis_Windows_x86_64/Tennis.exe"

# Create a navigation instance
t = Tennis(env_path, criteria=0.5)

# Load the pre-trained model
try:
#    t.load_model()
    print("Model loaded successfully")
except:
    print("No model to load...")

# Train the model
outcome = t.train()

# Save the trained model if the criteria is reached
if outcome:
    t.save_model()

if outcome:
    print("Criteria reached...")
    # Evaluate the model
    t.evaluate()

# Close the unity environment
t.close_env()