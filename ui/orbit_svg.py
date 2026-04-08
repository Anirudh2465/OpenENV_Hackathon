"""
OpenEnv-Orbital-Command | ui/orbit_svg.py

Generates an animated SVG of the orbital ring with:
- Earth at the centre with a subtle glow
- Sun hemisphere overlay (yellow arc)
- Eclipse zone (dark arc, 170°–350°)
- Satellite dots coloured by battery level (green→yellow→red)
- Ground station markers with FoV arcs
- ISL connection lines between neighbouring satellites
- Orbital ring with tick marks every 30°
"""
from __future__ import annotations
import math
from typing import List, Dict, Optional

# Canvas constants
CANVAS_W = 520
CANVAS_H = 520
CENTRE_X = 260
CENTRE_Y = 260
ORBIT_RADIUS = 200
EARTH_RADIUS = 35
SAT_RADIUS = 9
STATION_RADIUS = 7

# Colour palette (dark space theme)
BG_COLOUR       = "#0a0e1a"
ORBIT_COLOUR    = "#1e2d4a"
SUN_COLOUR      = "#ffdd57"
ECLIPSE_COLOUR  = "#0d1321"
EARTH_COLOUR    = "#1a6bb5"
EARTH_GLOW      = "#2196f3"
STAR_COLOUR     = "#ffffff"


def _deg_to_xy(deg: int, radius: float = ORBIT_RADIUS) -> tuple:
    """Convert orbital degree (0=right, clockwise) to SVG (x, y)."""
    rad = math.radians(90 - deg)   # 0° → top, clockwise
    x = CENTRE_X + radius * math.cos(rad)
    y = CENTRE_Y - radius * math.sin(rad)
    return x, y


def _arc_path(start_deg: int, end_deg: int, radius: float, large_arc: bool = False) -> str:
    """SVG arc path string from start_deg to end_deg on a circle."""
    sx, sy = _deg_to_xy(start_deg, radius)
    ex, ey = _deg_to_xy(end_deg, radius)
    la = 1 if large_arc else 0
    sweep = 1  # clockwise
    return f"M {sx:.2f} {sy:.2f} A {radius} {radius} 0 {la} {sweep} {ex:.2f} {ey:.2f}"


def _battery_colour(level: float) -> str:
    """Interpolate colour based on battery level."""
    if level >= 60:
        g = int(180 + (level - 60) * 1.25)
        return f"rgb(50,{min(220, g)},80)"
    if level >= 30:
        t = (level - 30) / 30
        r = int(220 - t * 100)
        g = int(100 + t * 120)
        return f"rgb({r},{g},40)"
    r = min(230, int(230 * (1 - level / 30)))
    return f"rgb({r},{max(40, int(40 * level/30))},40)"


def _mode_icon(mode: str) -> str:
    """Unicode icon for satellite mode (embedded in SVG text)."""
    return {
        "active":       "●",
        "sleep":        "◐",
        "transmitting": "↗",
        "isl_relay":    "⟳",
        "thermal_safe": "🌡",
        "dead":         "✕",
    }.get(mode, "●")


def generate_orbit_svg(
    satellites: List[Dict],        # list of dicts with orbital_position, battery_level, mode, sat_id, in_sunlight
    ground_stations: List[Dict],   # list of dicts with position_deg, status, station_id, fov_deg
    isl_topology: Dict[str, List[str]],
    imaging_requests: List[Dict],  # list of dicts with target_deg, priority
    step: int = 0,
    score: float = 0.0,
) -> str:
    """Render the full SVG string for the orbital dashboard."""

    # Pre-generate star positions (deterministic from seed)
    import random
    rng = random.Random(42)
    stars = [(rng.randint(5, CANVAS_W - 5), rng.randint(5, CANVAS_H - 5), rng.uniform(0.3, 1.0))
             for _ in range(120)]

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_W}" height="{CANVAS_H}" '
        f'viewBox="0 0 {CANVAS_W} {CANVAS_H}" style="background:{BG_COLOUR}; font-family: monospace;">',
        "<defs>",
        # Earth glow filter
        f'<filter id="glow"><feGaussianBlur stdDeviation="6" result="blur"/>'
        f'<feComposite in="SourceGraphic" in2="blur" operator="over"/></filter>',
        f'<filter id="glow-small"><feGaussianBlur stdDeviation="3" result="blur"/>'
        f'<feComposite in="SourceGraphic" in2="blur" operator="over"/></filter>',
        # Gradient for Earth
        f'<radialGradient id="earth-grad" cx="50%" cy="40%" r="60%">'
        f'<stop offset="0%" stop-color="#3a90e0"/>'
        f'<stop offset="100%" stop-color="#0d3b6e"/>'
        f'</radialGradient>',
        # Sun gradient
        f'<radialGradient id="sun-grad" cx="50%" cy="50%" r="50%">'
        f'<stop offset="0%" stop-color="#fff8d0"/>'
        f'<stop offset="70%" stop-color="#ffdd57"/>'
        f'<stop offset="100%" stop-color="#ffaa00" stop-opacity="0"/>'
        f'</radialGradient>',
        "</defs>",
    ]

    # Stars
    for sx, sy, opacity in stars:
        lines.append(f'<circle cx="{sx}" cy="{sy}" r="1" fill="{STAR_COLOUR}" opacity="{opacity:.2f}"/>')

    # Eclipse zone arc (dark fill on orbit)
    # Eclipse is 170°→350° (the "night" side)
    # Draw as a thick arc
    sx1, sy1 = _deg_to_xy(170, ORBIT_RADIUS + 18)
    ex1, ey1 = _deg_to_xy(350, ORBIT_RADIUS + 18)
    sx2, sy2 = _deg_to_xy(350, ORBIT_RADIUS - 18)
    ex2, ey2 = _deg_to_xy(170, ORBIT_RADIUS - 18)
    lines.append(
        f'<path d="M {sx1:.1f} {sy1:.1f} A {ORBIT_RADIUS+18} {ORBIT_RADIUS+18} 0 1 1 {ex1:.1f} {ey1:.1f} '
        f'L {sx2:.1f} {sy2:.1f} A {ORBIT_RADIUS-18} {ORBIT_RADIUS-18} 0 1 0 {ex2:.1f} {ey2:.1f} Z" '
        f'fill="#070d1a" opacity="0.85"/>'
    )

    # Eclipse label
    elx, ely = _deg_to_xy(260, ORBIT_RADIUS + 50)
    lines.append(f'<text x="{elx:.0f}" y="{ely:.0f}" text-anchor="middle" font-size="11" fill="#4a6080">🌑 ECLIPSE ZONE</text>')

    # Sun glow (on the sunlit side — °0 = right, sun at 0°)
    sun_x, sun_y = _deg_to_xy(0, ORBIT_RADIUS + 65)
    lines.append(f'<circle cx="{sun_x:.0f}" cy="{sun_y:.0f}" r="28" fill="url(#sun-grad)" opacity="0.9"/>')
    lines.append(f'<text x="{sun_x:.0f}" y="{sun_y+5:.0f}" text-anchor="middle" font-size="16">☀️</text>')

    # Orbit ring with tick marks
    lines.append(f'<circle cx="{CENTRE_X}" cy="{CENTRE_Y}" r="{ORBIT_RADIUS}" '
                 f'fill="none" stroke="{ORBIT_COLOUR}" stroke-width="2" stroke-dasharray="4 6" opacity="0.6"/>')

    # Tick marks every 30°
    for tick_deg in range(0, 360, 30):
        tx_out_x, tx_out_y = _deg_to_xy(tick_deg, ORBIT_RADIUS + 10)
        tx_in_x, tx_in_y = _deg_to_xy(tick_deg, ORBIT_RADIUS - 10)
        lines.append(f'<line x1="{tx_out_x:.1f}" y1="{tx_out_y:.1f}" x2="{tx_in_x:.1f}" y2="{tx_in_y:.1f}" '
                     f'stroke="#2a3a55" stroke-width="1"/>')
        lx, ly = _deg_to_xy(tick_deg, ORBIT_RADIUS + 24)
        lines.append(f'<text x="{lx:.0f}" y="{ly:.0f}" text-anchor="middle" font-size="9" fill="#3a5070">{tick_deg}°</text>')

    # Earth
    lines.append(f'<circle cx="{CENTRE_X}" cy="{CENTRE_Y}" r="{EARTH_RADIUS + 8}" '
                 f'fill="{EARTH_GLOW}" opacity="0.2" filter="url(#glow)"/>')
    lines.append(f'<circle cx="{CENTRE_X}" cy="{CENTRE_Y}" r="{EARTH_RADIUS}" '
                 f'fill="url(#earth-grad)"/>')
    lines.append(f'<text x="{CENTRE_X}" y="{CENTRE_Y + 5}" text-anchor="middle" font-size="20">🌍</text>')

    # ISL connections (draw before satellites for layering)
    drawn_isl = set()
    for src_id, neighbors in isl_topology.items():
        src_sat = next((s for s in satellites if s.get("sat_id") == src_id), None)
        if not src_sat:
            continue
        for dst_id in neighbors:
            link_key = tuple(sorted([src_id, dst_id]))
            if link_key in drawn_isl:
                continue
            drawn_isl.add(link_key)
            dst_sat = next((s for s in satellites if s.get("sat_id") == dst_id), None)
            if not dst_sat:
                continue
            sx, sy = _deg_to_xy(src_sat.get("orbital_position", 0))
            dx, dy = _deg_to_xy(dst_sat.get("orbital_position", 0))
            lines.append(f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{dx:.1f}" y2="{dy:.1f}" '
                         f'stroke="#00aaff" stroke-width="1.5" stroke-dasharray="5 3" opacity="0.5"/>')

    # Imaging request targets
    for req in imaging_requests:
        pos = req.get("target_deg", 0)
        pri = req.get("priority", "ROUTINE")
        tx, ty = _deg_to_xy(pos, ORBIT_RADIUS - 28)
        colour = {"EMERGENCY": "#ff3333", "URGENT": "#ff9900", "ROUTINE": "#55aa55"}.get(pri, "#555555")
        lines.append(f'<circle cx="{tx:.1f}" cy="{ty:.1f}" r="5" fill="{colour}" opacity="0.8"/>')
        lines.append(f'<circle cx="{tx:.1f}" cy="{ty:.1f}" r="9" fill="none" stroke="{colour}" stroke-width="1" opacity="0.4"/>')
        lines.append(f'<text x="{tx:.0f}" y="{ty-12:.0f}" text-anchor="middle" font-size="8" fill="{colour}">📷</text>')

    # Ground stations
    for stn in ground_stations:
        pos = stn.get("position_deg", 0)
        status = stn.get("status", "online")
        fov = stn.get("fov_deg", 15)
        stn_id = stn.get("station_id", "?")
        gx, gy = _deg_to_xy(pos, ORBIT_RADIUS + 32)
        colour = "#00dd88" if status == "online" else ("#555555" if status == "offline" else "#ffaa00")

        # FoV arc (inner orbit)
        arc_r = ORBIT_RADIUS - 22
        fov_sx, fov_sy = _deg_to_xy(pos - fov, arc_r)
        fov_ex, fov_ey = _deg_to_xy(pos + fov, arc_r)
        lines.append(f'<path d="M {fov_sx:.1f} {fov_sy:.1f} A {arc_r} {arc_r} 0 0 1 {fov_ex:.1f} {fov_ey:.1f}" '
                     f'fill="none" stroke="{colour}" stroke-width="1.5" opacity="0.4" stroke-dasharray="3 3"/>')

        lines.append(f'<circle cx="{gx:.1f}" cy="{gy:.1f}" r="{STATION_RADIUS}" fill="{colour}" opacity="0.9"/>')
        lx, ly = _deg_to_xy(pos, ORBIT_RADIUS + 50)
        short_name = stn_id.replace("Station_", "")
        lines.append(f'<text x="{lx:.0f}" y="{ly:.0f}" text-anchor="middle" font-size="9" fill="{colour}">{short_name}</text>')

    # Satellites
    for sat in satellites:
        pos = sat.get("orbital_position", 0)
        bat = sat.get("battery_level", 100.0)
        mode = sat.get("mode", "active")
        sat_id = sat.get("sat_id", "?")
        thermal = sat.get("thermal_level", 50.0)

        sx, sy = _deg_to_xy(pos)
        col = _battery_colour(bat) if mode != "dead" else "#333333"

        # Glow for active/transmitting
        if mode in ("active", "transmitting", "isl_relay"):
            lines.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="{SAT_RADIUS + 5}" '
                         f'fill="{col}" opacity="0.25" filter="url(#glow-small)"/>')

        # Satellite body
        lines.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="{SAT_RADIUS}" '
                     f'fill="{col}" stroke="#ffffff" stroke-width="1.2" opacity="0.95"/>')

        # Thermal ring
        if thermal > 70:
            t_col = "#ff6600" if thermal > 85 else "#ffaa00"
            lines.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="{SAT_RADIUS + 3}" '
                         f'fill="none" stroke="{t_col}" stroke-width="1.5" opacity="0.7"/>')

        # Label
        short = sat_id.split("-")[-1][:5]
        lines.append(f'<text x="{sx:.0f}" y="{sy - SAT_RADIUS - 5:.0f}" text-anchor="middle" '
                     f'font-size="9" fill="#ccddff" font-weight="bold">{short}</text>')

        # Battery mini-bar below satellite
        bar_w = 16
        bar_h = 3
        bx = sx - bar_w / 2
        by = sy + SAT_RADIUS + 4
        fill_w = (bat / 100.0) * bar_w
        lines.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w}" height="{bar_h}" '
                     f'fill="#1a2a3a" rx="1"/>')
        lines.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{fill_w:.1f}" height="{bar_h}" '
                     f'fill="{col}" rx="1"/>')

    # HUD overlay — top-left
    lines.append(f'<text x="12" y="22" font-size="13" fill="#aaccff" font-weight="bold">ORBITAL COMMAND</text>')
    lines.append(f'<text x="12" y="38" font-size="10" fill="#5577aa">Step: {step}  |  Score: {score:.1f}</text>')

    # Legend — bottom-right
    legend_items = [
        ("● Active", "#50dc80"),
        ("◐ Sleep", "#5599cc"),
        ("↗ TX", "#aaccff"),
        ("✕ Dead", "#555555"),
        ("📷 Target", "#ff9900"),
        ("── ISL", "#00aaff"),
    ]
    for i, (label, colour) in enumerate(legend_items):
        lx = CANVAS_W - 90
        ly = CANVAS_H - 100 + i * 16
        lines.append(f'<text x="{lx}" y="{ly}" font-size="9" fill="{colour}">{label}</text>')

    lines.append("</svg>")
    return "\n".join(lines)
