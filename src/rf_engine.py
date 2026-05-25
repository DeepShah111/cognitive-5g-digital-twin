# src/rf_engine.py
import numpy as np
import pandas as pd

def simulate_5g_tower_coverage(center_lat, center_lon, num_points=1500, optimized=False):
    """
    Simulates 5G mmWave coverage. 
    If optimized=True, it applies the AI's electrical downtilt fix to the dead zone.
    """
    lats = np.random.normal(center_lat, 0.004, num_points)
    lons = np.random.normal(center_lon, 0.004, num_points)
    
    distances = np.sqrt((lats - center_lat)**2 + (lons - center_lon)**2)
    max_dist = np.max(distances)
    
    normalized_dist = distances / max_dist
    signal_strength = 100 * (1 - (normalized_dist ** 1.5)) 
    
    df = pd.DataFrame({
        'lat': lats,
        'lon': lons,
        'signal_strength': np.clip(signal_strength, 0, 100)
    })
    
    # Inject Dead Zone
    dead_zone_mask = (df['lat'] > center_lat + 0.001) & (df['lon'] > center_lon + 0.001)
    
    if not optimized:
        # Pre-AI: Heavy mmWave blockage (drops to 15%)
        df.loc[dead_zone_mask, 'signal_strength'] = df.loc[dead_zone_mask, 'signal_strength'] * 0.15 
    else:
        # Post-AI: Agent applied +4.5 downtilt. Signal recovers to a usable 75% in the shadow!
        df.loc[dead_zone_mask, 'signal_strength'] = df.loc[dead_zone_mask, 'signal_strength'] * 0.75
        
    return df