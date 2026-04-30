"""Mission planner — route calculations and ArduPilot waypoint export.

Handles:
- Route distance calculation (haversine formula)
- Flight time estimation
- Photo interval and count based on altitude, overlap, and camera FOV
- Coverage area estimation
- Battery feasibility check
- ArduPilot .waypoints file generation (QGC WPL 110 format)
"""

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EARTH_RADIUS_M = 6_371_000  # meters

# Default camera: typical action camera (e.g. GoPro-style)
DEFAULT_HFOV_DEG = 70.0  # horizontal field of view in degrees
DEFAULT_VFOV_DEG = 55.0  # vertical field of view in degrees

# Battery
DEFAULT_MAX_FLIGHT_MIN = 18  # conservative for a ~3000-5000 mAh 3S/4S on F450

# ArduPilot MAVLink command IDs
MAV_CMD_NAV_WAYPOINT = 16
MAV_CMD_NAV_TAKEOFF = 22
MAV_CMD_NAV_LAND = 21

# Coordinate frame: global, altitude relative to home
MAV_FRAME_GLOBAL_RELATIVE_ALT = 3


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MissionSummary:
    """Summary of a planned mission."""

    waypoints: list[tuple[float, float]]  # [(lat, lon), ...]
    altitude_m: float
    speed_ms: float
    overlap_pct: float
    num_passes: int

    total_distance_m: float
    flight_time_s: float
    num_photos: int
    coverage_area_m2: float
    swath_width_m: float
    photo_interval_m: float
    battery_ok: bool
    battery_margin_pct: float  # positive = headroom, negative = over budget


# ---------------------------------------------------------------------------
# Core calculations
# ---------------------------------------------------------------------------

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in meters between two GPS points."""
    lat1, lon1, lat2, lon2 = (math.radians(v) for v in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def route_distance(waypoints: list[tuple[float, float]]) -> float:
    """Total route distance in meters along a list of (lat, lon) waypoints."""
    if len(waypoints) < 2:
        return 0.0
    return sum(
        haversine(waypoints[i][0], waypoints[i][1], waypoints[i + 1][0], waypoints[i + 1][1])
        for i in range(len(waypoints) - 1)
    )


def flight_time(distance_m: float, speed_ms: float, num_passes: int = 1) -> float:
    """Estimated flight time in seconds (distance * passes / speed)."""
    if speed_ms <= 0:
        return float("inf")
    return (distance_m * num_passes) / speed_ms


def ground_coverage(altitude_m: float, hfov_deg: float = DEFAULT_HFOV_DEG,
                    vfov_deg: float = DEFAULT_VFOV_DEG) -> tuple[float, float]:
    """Ground footprint of a single photo at the given altitude.

    Returns:
        (width_m, height_m) — width is across-track, height is along-track.
    """
    width = 2 * altitude_m * math.tan(math.radians(hfov_deg / 2))
    height = 2 * altitude_m * math.tan(math.radians(vfov_deg / 2))
    return width, height


def photo_interval(altitude_m: float, overlap_pct: float,
                   vfov_deg: float = DEFAULT_VFOV_DEG) -> float:
    """Distance between consecutive photos along the route (meters).

    Based on the along-track ground coverage and desired overlap.
    """
    _, height = ground_coverage(altitude_m, vfov_deg=vfov_deg)
    return height * (1 - overlap_pct / 100.0)


def num_photos(distance_m: float, interval_m: float) -> int:
    """Number of photos needed to cover the route distance at the given interval."""
    if interval_m <= 0:
        return 0
    return max(1, math.ceil(distance_m / interval_m) + 1)


def coverage_area(distance_m: float, altitude_m: float,
                  hfov_deg: float = DEFAULT_HFOV_DEG) -> float:
    """Approximate survey coverage area in square meters.

    Simplified as: route_distance * swath_width.
    """
    width, _ = ground_coverage(altitude_m, hfov_deg=hfov_deg)
    return distance_m * width


def check_battery(flight_time_s: float, max_flight_min: float = DEFAULT_MAX_FLIGHT_MIN) -> tuple[bool, float]:
    """Check if the mission fits within battery life.

    Returns:
        (is_ok, margin_pct) — margin_pct is positive if within budget.
    """
    max_s = max_flight_min * 60
    if max_s <= 0:
        return False, -100.0
    margin = (max_s - flight_time_s) / max_s * 100
    return flight_time_s <= max_s, margin


# ---------------------------------------------------------------------------
# High-level planner
# ---------------------------------------------------------------------------

def plan_mission(
    waypoints: list[tuple[float, float]],
    altitude_m: float = 25.0,
    speed_ms: float = 4.0,
    overlap_pct: float = 75.0,
    num_passes: int = 1,
    max_flight_min: float = DEFAULT_MAX_FLIGHT_MIN,
) -> MissionSummary:
    """Compute a full mission summary from waypoints and parameters."""
    dist = route_distance(waypoints)
    ft = flight_time(dist, speed_ms, num_passes)
    interval = photo_interval(altitude_m, overlap_pct)
    photos = num_photos(dist * num_passes, interval)
    area = coverage_area(dist, altitude_m)
    swath, _ = ground_coverage(altitude_m)
    ok, margin = check_battery(ft, max_flight_min)

    return MissionSummary(
        waypoints=waypoints,
        altitude_m=altitude_m,
        speed_ms=speed_ms,
        overlap_pct=overlap_pct,
        num_passes=num_passes,
        total_distance_m=dist,
        flight_time_s=ft,
        num_photos=photos,
        coverage_area_m2=area,
        swath_width_m=swath,
        photo_interval_m=interval,
        battery_ok=ok,
        battery_margin_pct=margin,
    )


# ---------------------------------------------------------------------------
# ArduPilot .waypoints export (QGC WPL 110)
# ---------------------------------------------------------------------------

def generate_waypoints_file(
    waypoints: list[tuple[float, float]],
    altitude_m: float = 25.0,
    speed_ms: float = 4.0,
) -> str:
    """Generate an ArduPilot-compatible .waypoints file string.

    Format: QGC WPL 110
    Line format: INDEX  CURRENT  FRAME  COMMAND  P1  P2  P3  P4  LAT  LON  ALT  AUTOCONTINUE

    The mission structure:
      0 — Home (first waypoint, ground level)
      1 — Takeoff to altitude
      2..N-1 — Navigate waypoints at altitude
      N — Land at last waypoint
    """
    if not waypoints:
        return ""

    lines = ["QGC WPL 110"]
    home_lat, home_lon = waypoints[0]

    # Line 0: Home position
    lines.append(
        f"0\t1\t0\t{MAV_CMD_NAV_WAYPOINT}\t0\t0\t0\t0\t"
        f"{home_lat:.8f}\t{home_lon:.8f}\t0.000000\t1"
    )

    # Line 1: Takeoff
    lines.append(
        f"1\t0\t{MAV_FRAME_GLOBAL_RELATIVE_ALT}\t{MAV_CMD_NAV_TAKEOFF}\t"
        f"0\t0\t0\t0\t{home_lat:.8f}\t{home_lon:.8f}\t{altitude_m:.6f}\t1"
    )

    # Lines 2..N-1: Waypoints
    for i, (lat, lon) in enumerate(waypoints):
        idx = i + 2
        lines.append(
            f"{idx}\t0\t{MAV_FRAME_GLOBAL_RELATIVE_ALT}\t{MAV_CMD_NAV_WAYPOINT}\t"
            f"0\t0\t0\t0\t{lat:.8f}\t{lon:.8f}\t{altitude_m:.6f}\t1"
        )

    # Last line: Land at final waypoint
    last_lat, last_lon = waypoints[-1]
    land_idx = len(waypoints) + 2
    lines.append(
        f"{land_idx}\t0\t{MAV_FRAME_GLOBAL_RELATIVE_ALT}\t{MAV_CMD_NAV_LAND}\t"
        f"0\t0\t0\t0\t{last_lat:.8f}\t{last_lon:.8f}\t0.000000\t1"
    )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Formatting helpers (for UI display)
# ---------------------------------------------------------------------------

def format_distance(meters: float) -> str:
    """Human-readable distance string."""
    if meters >= 1000:
        return f"{meters / 1000:.2f} km"
    return f"{meters:.0f} m"


def format_time(seconds: float) -> str:
    """Human-readable time string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes >= 60:
        hours = minutes // 60
        minutes = minutes % 60
        return f"{hours}h {minutes}m"
    return f"{minutes}m {secs}s"


def format_area(sq_meters: float) -> str:
    """Human-readable area string."""
    if sq_meters >= 10_000:
        return f"{sq_meters / 10_000:.2f} ha"
    return f"{sq_meters:,.0f} m²"
