"""
OpenEnv-Orbital-Command | ui/orbit_3d.py

Generates an interactive 3D Plotly globe of the orbital ring with:
- Earth via Scattergeo / orthographic projection
- Sun hemisphere (day/night boundary approximations)
- Satellite dots colored by battery level
- Ground station markers with FoV implied
- ISL connection lines between neighboring satellites
- Imaging requests
"""
from __future__ import annotations
from typing import List, Dict
import plotly.graph_objects as go

def _deg_to_lon(deg: float) -> float:
    """Map 0-359 orbit degrees to -180 to 180 Earth longitude.
    Assume the satellites orbit the equator (Latitude 0)."""
    return deg if deg <= 180 else deg - 360

def _battery_color(level: float) -> str:
    if level >= 50: return "#00e676"
    if level >= 20: return "#ffab40"
    return "#ff1744"

def generate_orbit_3d(
    satellites: List[Dict],
    ground_stations: List[Dict],
    isl_topology: Dict[str, List[str]],
    imaging_requests: List[Dict],
    step: int = 0,
    score: float = 0.0,
) -> go.Figure:
    fig = go.Figure()

    # 1. Eclipse Zone (Night side of the Earth)
    # The eclipse in the env is from 170 to 350.
    # We can draw it as a shaded polygon or a thick line.
    # Using a line on the equator to signify the eclipse bounding.
    eclipse_lons = [_deg_to_lon(d) for d in range(170, 351, 5)]
    fig.add_trace(go.Scattergeo(
        lon=eclipse_lons,
        lat=[0]*len(eclipse_lons),
        mode="lines",
        line=dict(width=16, color="rgba(5, 5, 10, 0.7)"),
        name="Eclipse Zone",
        hoverinfo="skip"
    ))

    # 2. ISL Topology Lines
    drawn_isl = set()
    for src_id, neighbors in isl_topology.items():
        src_sat = next((s for s in satellites if s.get("sat_id") == src_id), None)
        if not src_sat: continue
        for dst_id in neighbors:
            link_key = tuple(sorted([src_id, dst_id]))
            if link_key in drawn_isl: continue
            drawn_isl.add(link_key)
            dst_sat = next((s for s in satellites if s.get("sat_id") == dst_id), None)
            if not dst_sat: continue
            
            fig.add_trace(go.Scattergeo(
                lon=[_deg_to_lon(src_sat.get("orbital_position", 0)), _deg_to_lon(dst_sat.get("orbital_position", 0))],
                lat=[0, 0],
                mode="lines",
                line=dict(width=2, color="#00aaff", dash="dot"),
                name=f"ISL: {src_id}-{dst_id}",
                hoverinfo="none",
                showlegend=False
            ))

    # 3. Ground Stations
    gs_lons = [_deg_to_lon(s.get("position_deg", 0)) for s in ground_stations]
    gs_colors = ["#00dd88" if s.get("status") == "online" else ("#555555" if s.get("status") == "offline" else "#ffaa00") for s in ground_stations]
    gs_text = [f"Station: {s.get('station_id')}<br>Status: {s.get('status')}" for s in ground_stations]
    
    if ground_stations:
        fig.add_trace(go.Scattergeo(
            lon=gs_lons, lat=[0]*len(gs_lons),
            mode="markers",
            marker=dict(symbol="triangle-up", size=14, color=gs_colors, line=dict(width=1, color="white")),
            text=gs_text,
            name="Ground Stations",
            hoverinfo="text"
        ))

    # 4. Imaging Requests
    req_lons = [_deg_to_lon(r.get("target_deg", 0)) for r in imaging_requests]
    req_colors = [{"EMERGENCY": "#ff3333", "URGENT": "#ff9900", "ROUTINE": "#55aa55"}.get(r.get("priority", "ROUTINE"), "#555555") for r in imaging_requests]
    req_names = [f"Target {r.get('id')} ({r.get('priority')})" for r in imaging_requests]
    
    if imaging_requests:
        fig.add_trace(go.Scattergeo(
            lon=req_lons, lat=[0]*len(req_lons),
            mode="markers",
            marker=dict(symbol="x", size=12, color=req_colors, line=dict(width=2)),
            text=req_names,
            name="Imaging Targets",
            hoverinfo="text"
        ))

    # 5. Satellites
    sat_lons = [_deg_to_lon(s.get("orbital_position", 0)) for s in satellites]
    sat_colors = [_battery_color(s.get("battery_level", 100)) if s.get("mode") != "dead" else "#333333" for s in satellites]
    sat_texts = [f"<b>{s.get('sat_id')}</b><br>Battery: {s.get('battery_level'):.1f}%<br>Storage: {s.get('storage_used'):.1f}%<br>Thermal: {s.get('thermal_level'):.1f}°" for s in satellites]
    
    if satellites:
        fig.add_trace(go.Scattergeo(
            lon=sat_lons, lat=[0]*len(sat_lons),
            mode="markers+text",
            marker=dict(size=10, color=sat_colors, line=dict(width=2, color="white")),
            text=[s.get('sat_id').split("-")[-1][:5] for s in satellites],
            textfont=dict(color="white", size=10),
            textposition="top center",
            hovertext=sat_texts,
            name="Satellites",
            hoverinfo="text"
        ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        geo=dict(
            projection_type="orthographic",
            showland=True, landcolor="rgba(17, 30, 51, 0.3)",
            showocean=True, oceancolor="rgba(7, 13, 26, 0.05)",
            showcountries=True, countrycolor="rgba(42, 114, 214, 0.4)",
            showcoastlines=True, coastlinecolor="rgba(42, 114, 214, 0.7)",
            showlakes=True, lakecolor="rgba(7, 13, 26, 0.05)",
            showframe=True, framecolor="rgba(42, 114, 214, 0.3)", framewidth=2,
            bgcolor="rgba(0,0,0,0)",
            center=dict(lon_0=sat_lons[0] if sat_lons else 0, lat_0=0) # Track the first satellite
        ),
        showlegend=False,
        annotations=[
            dict(
                text=f"ORBITAL COMMAND | Step: {step} | Score: {score:.1f}",
                x=0.02, y=0.98, showarrow=False,
                font=dict(color="#2196f3", size=14, family="Orbitron")
            )
        ]
    )
    
    return fig
