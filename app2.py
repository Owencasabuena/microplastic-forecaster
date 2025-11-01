import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydeck as pdk
import re

# ===============================
# FUNCTIONS AND DATA
# ===============================

all_TSS = [50, 200]
all_Phos = [0.03, 2.5]
all_S = [0.5, 33]
weights = {'TSS': 0.5, 'Phosphates': 0.4, 'Salinity': 0.1}

SITE_LOCATIONS = {
    'Kawit (Site A - Polluted)': {'lat': 14.4617, 'lon': 120.9200, 'pis': 90, 'tss': 181, 'phosphates': 2.13, 'salinity': 0.92},
    'Noveleta (Site B - Less-Impacted)': {'lat': 14.4503, 'lon': 120.8769, 'pis': 10, 'tss': 93, 'phosphates': 0.05, 'salinity': 30.57},
    'Bacoor City (Coastal)': {'lat': 14.4697, 'lon': 120.9419, 'pis': 71, 'tss': 30, 'phosphates': 1.8, 'salinity': 15.0},
    'Imus (Coastal Area)': {'lat': 14.4363, 'lon': 120.9264, 'pis': 55, 'tss': 120, 'phosphates': 0.8, 'salinity': 10.0},
    'Cavite City (Coastal)': {'lat': 14.4842, 'lon': 120.8925, 'pis': 85, 'tss': 160, 'phosphates': 2.0, 'salinity': 5.0},
    'Rosario (Coastal)': {'lat': 14.43, 'lon': 120.86, 'pis': 60, 'tss': 130, 'phosphates': 1.0, 'salinity': 12.0},
    'Tanza (Coastal)': {'lat': 14.3917, 'lon': 120.8250, 'pis': 45, 'tss': 45, 'phosphates': 0.6, 'salinity': 25.0},
    'Naic (Coastal)': {'lat': 14.31, 'lon': 120.76, 'pis': 30, 'tss': 80, 'phosphates': 0.4, 'salinity': 30.0}
}

def normalize(val, vmin, vmax):
    return (val - vmin) / (vmax - vmin) if vmax != vmin else 0.0

def calculate_pollution_impact(site_data):
    tss = normalize(site_data['TSS'], min(all_TSS), max(all_TSS))
    phos = normalize(site_data['Phosphates'], min(all_Phos), max(all_Phos))
    sal = normalize(site_data['S'], min(all_S), max(all_S))
    return (weights['TSS']*tss + weights['Phosphates']*phos + weights['Salinity']*sal) * 100

def simulate_microplastic(site, initial_mass, days, A, Ea, DF_base):
    R = 8.314; g = 9.8
    rho_polymer = 920; V_particle = 1e-9
    rho_0, alpha, beta, T_ref = 1000, 0.7, 0.2, 25

    T_C, S, modifier = site['T_C'], site['S'], site['pollution_modifier']
    T_K = T_C + 273.15; DF = DF_base * modifier
    k = A * np.exp(-Ea/(R*T_K)) * DF
    mass = initial_mass * np.exp(-k*days*24*3600)
    rho_water = rho_0 + alpha*S - beta*(T_C-T_ref)
    F_net = (rho_water - rho_polymer)*V_particle*g
    half_life_days = np.log(2)/k/86400 if k>0 else np.inf
    return mass, k, half_life_days, F_net, rho_water, DF

def dms_to_decimal(dms_str):
    dms_pattern = r"(\d+)°(\d+)'([\d.]+)\"?([NSEW])"
    match = re.match(dms_pattern, dms_str.strip())
    if not match: return float(dms_str)
    deg, min_, sec, hemi = match.groups()
    decimal = float(deg)+float(min_)/60+float(sec)/3600
    if hemi in ['S','W']: decimal *= -1
    return decimal

def time_to_sink(F_net, attachment_rate):
    if F_net <= 0: return 0
    if attachment_rate <= 0: return np.inf
    return 1.0 / attachment_rate  # days until sinking

# ===============================
# STREAMLIT DASHBOARD
# ===============================

st.set_page_config(page_title="Microplastic Forecaster", layout="wide")
st.title("Microplastic Pollution Forecaster")
st.markdown("Hybrid computational framework for coastal water assessment.")

st.sidebar.header("Input Parameters")
input_tss = st.sidebar.number_input("TSS (mg/L)", 0.0, 1000.0, 93.0)
input_phosphates = st.sidebar.number_input("Phosphates (mg/L)", 0.0, 10.0, 0.05)
input_salinity = st.sidebar.number_input("Salinity (ppt)", 0.0, 40.0, 30.57)
input_temp = st.sidebar.number_input("Temperature (°C)", 0.0, 50.0, 26.23)
input_abundance = st.sidebar.number_input("Observed Abundance (optional)", 0.0, 10000.0, 0.0)
attachment_rate = st.sidebar.number_input("Attachment Rate (1/days)", 0.0, 10.0, 0.1)

input_location = st.sidebar.text_input("Location Name", "My Research Site")
input_lat_str = st.sidebar.text_input("Latitude", "14.45")
input_lon_str = st.sidebar.text_input("Longitude", "120.90")

if st.sidebar.button("Calculate Forecast", use_container_width=True):
    st.balloons()
    try:
        input_lat = dms_to_decimal(input_lat_str)
        input_lon = dms_to_decimal(input_lon_str)
    except:
        st.error("Invalid coordinate format. Use decimal or DMS.")
        st.stop()

    user_data = {'TSS': input_tss,'Phosphates': input_phosphates,'S': input_salinity,'T_C': input_temp,'pollution_modifier':1.0}
    pis_score = calculate_pollution_impact(user_data)

    if input_tss>150: user_data['pollution_modifier']=0.7
    mass,k,hl,F,rho_w,DF = simulate_microplastic(user_data,1e-9,365,1e3,120000,0.8)
    sink_days = time_to_sink(F, attachment_rate)

    # Pollution Hotspot Map
    st.subheader("1. Pollution Hotspot Map")
    map_df = pd.DataFrame(SITE_LOCATIONS).T.reset_index()
    map_df = map_df.rename(columns={'index':'Site'})
    map_df['lat']=map_df['lat'].astype(float); map_df['lon']=map_df['lon'].astype(float)
    map_df.loc[len(map_df)] = [input_location,input_lat,input_lon,pis_score,input_tss,input_phosphates,input_salinity]
    map_df['color'] = [[0,104,201]]*(len(map_df)-1)+[[255,43,43]]
    layer=pdk.Layer("ScatterplotLayer",data=map_df,get_position='[lon,lat]',get_fill_color='color',get_radius=300)
    view=pdk.ViewState(latitude=map_df["lat"].mean(),longitude=map_df["lon"].mean(),zoom=11)
    st.pydeck_chart(pdk.Deck(layers=[layer],initial_view_state=view))

    # PIS Score
    st.subheader("2. Pollution Impact Score")
    st.metric("Your Site's PIS", f"{pis_score:.2f}")
    st.bar_chart(pd.DataFrame({'Site':list(SITE_LOCATIONS.keys())+[input_location],'PIS':[loc['pis'] for loc in SITE_LOCATIONS.values()]+[pis_score]}).set_index('Site'))

    # Degradation
    st.subheader("3. Degradation Prediction")
    st.info(f"k={k:.2e} s⁻¹, Half-life={hl/365:.2e} years → extremely persistent microplastics.")

    # Buoyancy
    st.subheader("4. Buoyancy Prediction")
    fate="Floating" if F>0 else "Sinking"
    st.info(f"Water density={rho_w:.2f} kg/m³ vs Plastic density=920 kg/m³ → {fate} predicted.")

    # Time to Sink Visualization
    st.subheader("5. Time to Sink Prediction")
    if sink_days==np.inf: st.warning("Particles will stay afloat indefinitely unless attachment rate > 0.")
    else: st.success(f"Estimated sinking time = {sink_days:.1f} days")
    fig,ax=plt.subplots()
    times=np.linspace(0,max(10,sink_days),100)
    status=np.where(times>=sink_days,0,1)
    ax.plot(times,status,label="Floating(1)/Sinking(0)")
    ax.set_xlabel("Time (days)"); ax.set_ylabel("Status"); ax.legend(); ax.set_title("Floating vs Sinking Over Time")
    st.pyplot(fig)

    # ===============================
    # 6. Water Column Visualization (Particle Path)
    # ===============================
    st.subheader("6. Floating vs Sinking in the Water Column")

    max_depth = 10  # meters
    total_time = max(10, sink_days if sink_days != np.inf else 10)  # days
    times = np.linspace(0, total_time, 200)
    depths = np.zeros_like(times)

    if sink_days == np.inf:
        depths[:] = 0  # stays at surface
    else:
        sinking_speed = max_depth / (total_time - sink_days if total_time > sink_days else 1)
        for i, t in enumerate(times):
            if t < sink_days:
                depths[i] = 0  # at surface
            else:
                d = (t - sink_days) * sinking_speed
                depths[i] = min(d, max_depth)

    # Create cross-section diagram
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.fill_between(times, 0, -max_depth, color='#87CEEB', alpha=0.3, label="Water")
    ax2.fill_between(times, -max_depth, -max_depth - 2, color='#8B4513', alpha=0.5, label="Sediment")
    ax2.plot(times, -depths, 'ro-', markersize=5, label="Microplastic Particle")
    ax2.axhline(0, color='blue', linestyle='--', label='Water Surface')
    ax2.axhline(-max_depth, color='brown', linestyle='--', label='Sediment Boundary')

    ax2.set_ylim(-max_depth - 2, 2)
    ax2.set_xlim(0, total_time)
    ax2.set_xlabel("Time (days)")
    ax2.set_ylabel("Depth (m)")
    ax2.set_title("Microplastic Floating vs. Sinking Path")
    ax2.legend(loc="upper right")
    st.pyplot(fig2)
