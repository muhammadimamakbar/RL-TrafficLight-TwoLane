import json
import time
#import paho.mqtt

while True:
    try:
        # Baca data dari file JSON
        with open('data.json', 'r') as f:
            data = json.load(f)

        print("Received data:", data)
        if data['simpang']['light'] == "rG":
            print("Right-to-Left")
            ## Right TL signal ON, Left TL signal OFF
            # Publish
        else :
            print("Left-to-Right")
            ## Left TL signal ON, Right TL signal OFF
            # Publishz

    except FileNotFoundError:
        print("File not found, waiting for data...")

    # Tunggu sebentar sebelum mencoba membaca lagi
    time.sleep(1)
