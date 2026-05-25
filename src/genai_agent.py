# src/genai_agent.py
import os
import json
import random
import re
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Securely load the API key from the .env file
load_dotenv()

def run_cognitive_optimization(target_sector):
    print(f"\n[SYSTEM] Initializing AETHER-5G Mesh Engine for {target_sector}...")
    fiber_capacity = random.randint(58, 74) 

    try:
        llm = ChatGroq(temperature=0.15, model_name="llama-3.1-8b-instant")
    except Exception as e:
        return {"fiber_load": 0, "diagnostics": "API Connection Error", "action": "Verify Key", "citation": str(e)}

    prompt_template = """
    You are AETHER-5G: An advanced Cognitive Network Orchestration AI.
    A Multi-Node 5G mmWave Mesh Network is experiencing overlapping sector interference and NLoS fading in: {sector}.
    The regional DWDM trunk has {fiber_load}% optical capacity availability.
    
    Prescribe a synchronized dual-cell optimization plan. Return ONLY a valid JSON object with flat string keys:
    "diagnostics": "Explain the multi-path reflections and co-channel interference across the dual-node topology."
    "action": "Prescribe electrical antenna downtilt tweaks for BOTH cells and explicit optical transport load adjustments."
    "citation": "Reference 3GPP Release 17 NR-U or ITU-T G.652 standards."
    """
    
    prompt = PromptTemplate(input_variables=["sector", "fiber_load"], template=prompt_template)
    try:
        raw_response = (prompt | llm).invoke({"sector": target_sector, "fiber_load": fiber_capacity})
        json_match = re.search(r'\{.*\}', raw_response.content, re.DOTALL)
        ai_data = json.loads(json_match.group(0)) if json_match else json.loads(raw_response.content)
        
        def flatten(val):
            return " ".join([str(v) for v in val.values()]) if isinstance(val, dict) else str(val)

        return {
            "status": "success",
            "fiber_load": fiber_capacity,
            "diagnostics": flatten(ai_data.get("diagnostics", "Multi-cell interference detected.")),
            "action": flatten(ai_data.get("action", "Adjust tilt parameters across mesh.")),
            "citation": flatten(ai_data.get("citation", "3GPP compliant."))
        }
    except Exception:
        return {
            "status": "error",
            "fiber_load": fiber_capacity,
            "diagnostics": f"Structural multipath fade confirmed in dual-node {target_sector} topology.",
            "action": "Execute +3.5° electrical downtilt on Cell 01 and Cell 02.",
            "citation": "ITU-T G.652 Single-Mode Protocol."
        }

def chat_with_telecom_agent(user_message, current_state, active_zone):
    try:
        llm = ChatGroq(temperature=0.35, model_name="llama-3.1-8b-instant")
        context_str = f"Monitoring {active_zone} multi-node mesh. No current overrides applied."
        if current_state:
            context_str = f"Active Sector: {active_zone} Multi-Node Mesh. Load: {current_state.get('fiber_load')}%. " \
                          f"Diagnostics: {current_state.get('diagnostics')}. " \
                          f"Action: {current_state.get('action')}."
        
        template = """
        You are AETHER-5G: A Cognitive Network Orchestration Brain managing a multi-cell mesh topology.
        Current Mesh Context: {context}
        Engineer Inquiry: {question}
        Provide an engineering response. Use core EXTC terms (e.g., optical return loss, 3GPP beamforming, multi-user MIMO, handover interference). Keep the response highly technical, clean, and concise.
        """
        prompt = PromptTemplate(input_variables=["context", "question"], template=template)
        return (prompt | llm).invoke({"context": context_str, "question": user_message}).content
    except Exception as e:
        return f"AETHER-5G Core Connection Interrupted: {str(e)}"