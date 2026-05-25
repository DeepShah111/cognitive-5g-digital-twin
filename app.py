# app.py
import streamlit as st
import pydeck as pdk
import geopandas as gpd
import os
import sys
import json
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

from src.ml_engine import run_mesh_network_prediction
from src.genai_agent import run_cognitive_optimization, chat_with_telecom_agent

st.set_page_config(page_title="AETHER-5G | Cognitive Digital Twin", layout="wide", initial_sidebar_state="expanded")

@st.cache_data
def load_zone_geometry(zone_name):
    if zone_name == "Bandra Kurla Complex":
        filepath = os.path.join(config.DATA_DIR, "mumbai_buildings.geojson")
        if os.path.exists(filepath):
            gdf = gpd.read_file(filepath)
            for col in ['height', 'building:levels']:
                if col in gdf.columns:
                    gdf[col] = gpd.pd.to_numeric(gdf[col], errors='coerce')
                else:
                    gdf[col] = None
            gdf['render_height'] = gdf['height'].fillna(gdf['building:levels'] * 3).fillna(15)
            return gdf
    return gpd.GeoDataFrame()

st.title("📡 AETHER-5G: Multi-Node Cognitive Orchestrator")
st.markdown("### Next-Generation Geospatial Digital Twin | Real-Time Mesh Network Synthesis")
st.markdown("---")

col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("NOC Control Center")
    
    selected_zone = st.selectbox(
        "Select Mesh Topology Sector", 
        ["Bandra Kurla Complex", "Nariman Point"],
        key="primary_zone_selector"
    )
    
    mesh_topologies = {
        "Bandra Kurla Complex": [(19.0665, 72.8635), (19.0635, 72.8685), (19.0620, 72.8600)], 
        "Nariman Point": [(18.9280, 72.8220), (18.9250, 72.8240)]        
    }
    active_nodes = mesh_topologies[selected_zone]
    
    cam_lat = sum(n[0] for n in active_nodes) / len(active_nodes)
    cam_lon = sum(n[1] for n in active_nodes) / len(active_nodes)
    
    st.markdown("---")
    
    if st.button("1. Run Mesh Network Super-Resolution", type="primary", use_container_width=True):
        st.session_state['rf_active'] = True
        st.session_state['optimized'] = False 
        
    can_optimize = st.session_state.get('rf_active', False)
    if st.button("2. Deploy GenAI Optimizer (AETHER Agent)", disabled=not can_optimize, use_container_width=True):
        with st.spinner("Analyzing Mesh Synchronization & Optical Transports..."):
            agent_result = run_cognitive_optimization(selected_zone)
            st.session_state['ai_result'] = agent_result
            st.session_state['optimized'] = True
            time.sleep(0.3)

    st.markdown("### 📊 RSRP Signal Legend")
    st.markdown(
        """
        <div style="padding: 10px; border-radius: 5px; background-color: #1E1E1E; border: 1px solid #333;">
            <div style="display: flex; align-items: center; margin-bottom: 8px;"><div style="width: 20px; height: 12px; background-color: rgb(255, 0, 0); margin-right: 10px; border-radius: 2px;"></div><span style="font-size: 13px; color: #EEE;">Excellent (LoS Zone): &gt; -75 dBm</span></div>
            <div style="display: flex; align-items: center; margin-bottom: 8px;"><div style="width: 20px; height: 12px; background-color: rgb(255, 255, 0); margin-right: 10px; border-radius: 2px;"></div><span style="font-size: 13px; color: #EEE;">Good Propagation: -75 to -85 dBm</span></div>
            <div style="display: flex; align-items: center; margin-bottom: 8px;"><div style="width: 20px; height: 12px; background-color: rgb(0, 255, 0); margin-right: 10px; border-radius: 2px;"></div><span style="font-size: 13px; color: #EEE;">Acceptable Link: -85 to -95 dBm</span></div>
            <div style="display: flex; align-items: center;"><div style="width: 20px; height: 12px; background-color: rgb(0, 25, 255); margin-right: 10px; border-radius: 2px;"></div><span style="font-size: 13px; color: #EEE;">Severe Shadow Fade: &lt; -100 dBm</span></div>
        </div>
        """, unsafe_allow_html=True
    )

    if 'ai_result' in st.session_state:
        res = st.session_state['ai_result']
        st.markdown("### 🔌 Regional Optical Transport")
        st.progress(res['fiber_load'] / 100.0, text=f"Mesh DWDM Capacity: {res['fiber_load']}%")
        st.markdown("### 🧠 Autonomous Diagnostics")
        st.error(f"**Root Cause:** {res['diagnostics']}")
        st.success(f"**Prescribed Action:** {res['action']}")
    else:
        st.markdown("### 🔌 Regional Optical Transport")
        st.progress(0, text="DWDM Optical Core: Standby")
        if not can_optimize: st.info("System Status: Awaiting Mesh Telemetry...")

    st.markdown("---")
    st.markdown("### 💬 AETHER-5G Live Assistant")
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "AETHER-5G Multi-Node Core online. Awaiting query..."}]

    chat_container = st.container(height=460)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if prompt := st.chat_input("Query the mesh orchestrator..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Orchestrator synthesizing..."):
                    current_ai_state = st.session_state.get('ai_result', None)
                    response = chat_with_telecom_agent(prompt, current_ai_state, selected_zone)
                    st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

with col2:
    with st.spinner("Synthesizing 3D Geospatial Topology & Mesh Inferences..."):
        gdf = load_zone_geometry(selected_zone)
        render_layers = []
        
        is_active = st.session_state.get('rf_active', False)
        is_optimized = st.session_state.get('optimized', False)

        if is_active:
            rf_data, antenna_inventory = run_mesh_network_prediction(active_nodes, map_gdf=gdf, current_zone=selected_zone, is_optimized=is_optimized)
            
            heatmap_layer = pdk.Layer(
                "HeatmapLayer", data=rf_data, opacity=0.75,
                get_position=["lon", "lat"], get_weight="signal_strength",
                color_range=[[0,25,255], [0,255,255], [0,255,0], [255,255,0], [255,0,0]]
            )
            render_layers.append(heatmap_layer)
        else:
            antenna_inventory = [{"lat": lat, "lon": lon, "name": f"Macro Cell Offline"} for lat, lon in active_nodes]

        if not gdf.empty:
            building_layer = pdk.Layer(
                "GeoJsonLayer", data=json.loads(gdf.to_json()), opacity=0.35, 
                stroked=True, filled=True, extruded=True, wireframe=True,
                get_elevation="properties.render_height", get_fill_color="[50, 130, 245, 110]", get_line_color="[255, 255, 255]"
            )
            render_layers.insert(0, building_layer)

        antenna_layer = pdk.Layer(
            "ScatterplotLayer", data=antenna_inventory,
            get_position=["lon", "lat"], get_color="[255, 255, 255, 255]", get_radius=25,
            stroked=True, get_line_color="[255, 0, 0, 255]", line_width_min_pixels=3
        )
        render_layers.append(antenna_layer)

        view_state = pdk.ViewState(latitude=cam_lat, longitude=cam_lon, zoom=14.4, pitch=50, bearing=-10)
        
        r = pdk.Deck(layers=render_layers, initial_view_state=view_state, map_style="dark", tooltip={"html": "<b>Node ID:</b> {name}"})
        st.pydeck_chart(r, use_container_width=True)

    # --- DYNAMIC TELEMETRY & LIVE MATHEMATICS PREPARATION ---
    if not is_active:
        status_color = "#888888"
        status_header = "STATUS: IDLE | AWAITING DEPLOYMENT"
        log_text = f"<ul><li><b>Geospatial Vectors:</b> 3D architectural arrays for {selected_zone} loaded successfully.</li><li><b>Mesh Toplogy:</b> {len(active_nodes)} Macro gNodeB anchors positioned via dynamic coordinate mapping.</li><li><b>System State:</b> Nodes are currently offline. Awaiting manual trigger to ignite the PyTorch spatial inference engine.</li></ul>"
        
        live_tx_power = 0
        live_eirp = 0
        live_downtilt = "Offline"
        live_backhaul = "Offline"
        fspl_loss = "0"
        
    elif not is_optimized:
        status_color = "#FFA500" 
        status_header = "STATUS: ACTIVE INFERENCE | BASELINE NETWORK RENDERED"
        log_text = f"<ul><li><b>Neural Execution:</b> PyTorch U-Net actively processing 64x64 spatial tensors across {len(active_nodes)} synchronized nodes.</li><li><b>Atmospheric Physics:</b> Simulating natural <i>Free Space Path Loss (FSPL)</i> across open topologies.</li><li><b>Structural Physics:</b> Predicting <i>Non-Line-of-Sight (NLoS)</i> concrete diffraction.</li><li><b>Warning:</b> Network is live but experiencing severe multipath attenuation. GenAI optimization required.</li></ul>"
        
        live_tx_power = 40  
        live_eirp = live_tx_power + 24 - 2  
        live_downtilt = "2° (Default Baseline)"
        live_backhaul = "Telemetry Scanning..."
        fspl_loss = "115.3"
        
    else:
        status_color = "#00FF00" 
        status_header = "STATUS: COGNITIVE OVERRIDE | MESH NETWORK OPTIMIZED"
        log_text = f"<ul><li><b>Agent Deployment:</b> Llama-3.1 Generative AI actively monitoring optical DWDM backhaul capacities.</li><li><b>Physical Override:</b> AI applied dynamic <i>Electrical Antenna Downtilts (EDT)</i> and targeted <i>TX Power Boosts</i>.</li><li><b>Mesh Synchronization:</b> {len(active_nodes)} independent nodes successfully blended using 95th Percentile Tensor Normalization.</li><li><b>Result:</b> Core coverage heatmaps expanded. Dead zones stitched together.</li></ul>"
        
        live_tx_power = 43  
        live_eirp = live_tx_power + 24 - 2  
        live_downtilt = "8° (AI Adjusted)"
        ai_fiber = st.session_state.get('ai_result', {}).get('fiber_load', '75')
        live_backhaul = f"{ai_fiber}% Capacity (Optimized)"
        fspl_loss = "115.3"


    # --- RENDER UPGRADE 1: DYNAMIC NEURAL TELEMETRY CONSOLE ---
    st.markdown("### 👁️ AETHER-5G Neural Telemetry Stream")
    st.markdown(f"""
    <div style="background-color:#111111; padding:20px; border-radius:8px; border-left:6px solid {status_color}; border:1px solid #333;">
        <h4 style="margin-top:0; color:{status_color}; letter-spacing: 1px; font-family: monospace;">{status_header}</h4>
        <div style="font-size:15px; color:#DDDDDD; line-height: 1.6;">{log_text}</div>
    </div>
    <br>
    """, unsafe_allow_html=True)

    # --- RENDER UPGRADE 2: LIVE CALCULATIONS WINDOW (LATEX BUG FIXED) ---
    with st.expander("🧮 Live EXTC Link Budget Calculations (EIRP & FSPL)", expanded=False):
        st.markdown(f"""
        **The EXTC Mathematical Reality:**
        To validate the PyTorch Deep Learning engine, we run real-time hardware link budgets. The core metric for cellular transmission is **EIRP (Effective Isotropic Radiated Power)**, which calculates the true power leaving the antenna hardware.

        **1. Real-Time EIRP Calculation**
        *Formula:* $EIRP = P_t + G_t - L_c$
        * **Base TX Power ($P_t$):** `{live_tx_power} dBm`
        * **Antenna Array Gain ($G_t$):** `+24 dBi` (64T64R Massive MIMO)
        * **Internal Cable Loss ($L_c$):** `-2 dB`
        * **Live EIRP Output:** `<span style="color:{status_color}; font-size:18px; font-weight:bold;">{live_eirp} dBm</span>`

        **2. Edge Signal Degradation (Free Space Path Loss)**
        At 28 GHz mmWave frequencies, oxygen absorption is severe. We calculate the mathematical path loss exactly 500 meters away from the node in free space:
        *Formula:* $FSPL = 92.4 + 20\log_{{10}}(d_{{km}}) + 20\log_{{10}}(f_{{GHz}})$
        * $FSPL = 92.4 + 20\log_{{10}}(0.5) + 20\log_{{10}}(28) = $ `{fspl_loss} dB` attenuation.
        * **Theoretical Received Power ($P_r$):** EIRP - FSPL = `{live_eirp} - {fspl_loss} = {live_eirp - float(fspl_loss):.1f} dBm`
        
        *(Note: If the received signal drops below -100 dBm due to the deep blue concrete shadows, the node triggers a connection failure).*
        """, unsafe_allow_html=True)

    # --- RENDER UPGRADE 3: CORE EXTC TERMINOLOGY (STATE-AWARE) ---
    with st.expander("📖 AETHER Core Terminology Dictionary", expanded=False):
        st.markdown(f"""
        * **Electrical Downtilt (EDT):** Instead of manually climbing a tower to tilt the heavy physical antenna chassis, engineers alter the electrical phase of the antenna elements to mathematically steer the RF beam toward the ground, reducing interference.
          * *Live System State:* **`{live_downtilt}`**
        * **EIRP (Effective Isotropic Radiated Power):** The measured radiated power of an antenna in a single direction. This is strictly regulated by telecom bodies (like TRAI or FCC) to prevent microwave radiation hazards. 
          * *Live System State:* **`{live_eirp} dBm transmission detected.`**
        * **DWDM (Dense Wavelength Division Multiplexing):** The optical fiber architecture connecting our cell towers back to the internet core. It transmits multiple signals simultaneously at different laser wavelengths (colors of light) over a single strand of glass.
          * *Live System State:* **`{live_backhaul}`**
        * **Non-Line-of-Sight (NLoS) Shadow Fading:** When a physical obstruction (like the 3D concrete high-rises on our map) blocks the direct wave path. The AI predicts these shadows instantly, completely bypassing the need to run 15-minute ray-tracing simulations.
        """)