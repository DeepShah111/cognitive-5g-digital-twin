# src/ml_engine.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class RF_UNet(nn.Module):
    def __init__(self):
        super(RF_UNet, self).__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True)
        )
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
        
    def forward(self, x):
        e1 = self.enc1(x)
        b = self.bottleneck(self.pool1(e1))
        merge1 = torch.cat([self.upconv1(b), e1], dim=1)
        return self.final_conv(self.dec1(merge1))

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "rf_vision_unet.pth")
device = torch.device("cpu") 

try:
    rf_model = RF_UNet()
    rf_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    rf_model.eval() 
    print("[SYSTEM] PyTorch Vision Model Loaded Successfully.")
except Exception as e:
    print(f"[WARNING] Could not load PyTorch model. Error: {e}")
    rf_model = None

def run_mesh_network_prediction(node_list, map_gdf, current_zone, is_optimized=False, grid_size=64):
    if rf_model is None:
        raise RuntimeError("PyTorch model not loaded.")
    
    all_network_data = []
    antenna_inventory = []
    
    for idx, (center_lat, center_lon) in enumerate(node_list):
        ai_input = np.zeros((grid_size, grid_size), dtype=np.float32)
        lat_step = 0.008 / grid_size
        lon_step = 0.008 / grid_size
        
        if not map_gdf.empty:
            for _, row in map_gdf.iterrows():
                b_lon = row.geometry.centroid.x
                b_lat = row.geometry.centroid.y
                x_idx = int((b_lon - (center_lon - 0.004)) / lon_step)
                y_idx = int((b_lat - (center_lat - 0.004)) / lat_step)
                if 0 <= x_idx < grid_size and 0 <= y_idx < grid_size:
                    ai_input[max(0, y_idx-1):min(grid_size, y_idx+2), max(0, x_idx-1):min(grid_size, x_idx+2)] = 1.0
        else:
            np.random.seed(int(center_lat * 10000))
            for _ in range(15):
                cx, cy = np.random.randint(15, 49), np.random.randint(15, 49)
                ai_input[cy-2:cy+3, cx-2:cx+3] = 1.0

        tensor_input = torch.tensor(ai_input).unsqueeze(0).unsqueeze(0).float()
        
        with torch.no_grad():
            predicted_tensor = rf_model(tensor_input)
        raw_heatmap = predicted_tensor.squeeze().numpy()

        # --- THE TOP 1% MLOPS FIX: 95th PERCENTILE NORMALIZATION ---
        # Guarantees both antennas have identically sized red cores, 
        # bypassing PyDeck's density skewing algorithm.
        p95 = np.percentile(raw_heatmap, 95)
        if p95 > 0.01:
            raw_heatmap = np.clip(raw_heatmap / p95, 0, 1.0)

        # High-Resolution Bilinear Interpolation
        target_scale = 128
        scale_factor = grid_size / target_scale
        upscaled_heatmap = np.zeros((target_scale, target_scale), dtype=np.float32)
        
        for y in range(target_scale):
            for x in range(target_scale):
                orig_x = x * scale_factor
                orig_y = y * scale_factor
                x_low, y_low = int(orig_x), int(orig_y)
                x_high, y_high = min(grid_size - 1, x_low + 1), min(grid_size - 1, y_low + 1)
                
                wa = (x_high - orig_x) * (y_high - orig_y)
                wb = (orig_x - x_low) * (y_high - orig_y)
                wc = (x_high - orig_x) * (orig_y - y_low)
                wd = (orig_x - x_low) * (orig_y - y_low)
                
                upscaled_heatmap[y, x] = (wa * raw_heatmap[y_low, x_low] + 
                                          wb * raw_heatmap[y_low, x_high] + 
                                          wc * raw_heatmap[y_high, x_low] + 
                                          wd * raw_heatmap[y_high, x_high])

        # Tighter Physical Fade Mask to prevent the signal from washing out the screen
        y_coords, x_coords = np.ogrid[:target_scale, :target_scale]
        center_val = target_scale / 2.0
        dist_from_center = np.sqrt((x_coords - center_val)**2 + (y_coords - center_val)**2)
        fade_mask = np.clip(1.0 - (dist_from_center / center_val)**1.8, 0, 1)
        
        upscaled_heatmap = upscaled_heatmap * fade_mask
        
        if is_optimized:
            upscaled_heatmap = np.where(upscaled_heatmap < 0.2, upscaled_heatmap * 4.2, upscaled_heatmap)
        
        fine_lat_step = 0.008 / target_scale
        fine_lon_step = 0.008 / target_scale
        
        for y in range(target_scale):
            for x in range(target_scale):
                real_lat = center_lat + ((y - (target_scale // 2)) * fine_lat_step)
                real_lon = center_lon + ((x - (target_scale // 2)) * fine_lon_step)
                signal = upscaled_heatmap[y, x] * 100 
                
                if signal > 8: 
                    all_network_data.append({'lat': real_lat, 'lon': real_lon, 'signal_strength': signal})
                    
        antenna_inventory.append({"lat": center_lat, "lon": center_lon, "name": f"Macro Cell 0{idx+1} ({current_zone})"})
                
    return pd.DataFrame(all_network_data), antenna_inventory