# src/data_loader.py
import sys
import os

# Force Python to print immediately to the terminal
sys.stdout.reconfigure(line_buffering=True)
print("Initializing libraries (this takes a few seconds)...")

import osmnx as ox
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def fetch_mumbai_buildings():
    print(f"Fetching 3D building geometry within {config.RADIUS_METERS}m of {config.TARGET_ADDRESS}...")
    
    try:
        # Fetching data using a radius is much faster and more reliable
        tags = {'building': True}
        buildings = ox.features_from_address(
            config.TARGET_ADDRESS, 
            tags=tags, 
            dist=config.RADIUS_METERS
        )
        
        print(f"Success! Downloaded {len(buildings)} real building footprints.")
        
        output_path = os.path.join(config.DATA_DIR, "mumbai_buildings.geojson")
        
        # Keep only the necessary geometry data
        cols_to_keep = buildings.columns.intersection(['geometry', 'building:levels', 'height'])
        clean_buildings = buildings[cols_to_keep]
        
        clean_buildings.to_file(output_path, driver='GeoJSON')
        print(f"Data saved perfectly to: {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    fetch_mumbai_buildings()