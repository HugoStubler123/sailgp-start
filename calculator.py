"""
Strategy calculations for the SailGP start strategist.
Geometry helpers, race data fetching and all tactical computations.
"""
import json
import numpy as np
import requests
import xml.etree.ElementTree as ET

EARTH_R = 6_371_000.0
XML_URL = "http://xml.sailgp.tech/xml"


# ── Local XML parsing ───────────────────────────────────────────────────────

def parse_local_xml(xml_content: str) -> dict:
    """
    Parse a single race XML file (drag-and-drop).
    Uses the Boundary from the XML as the start box polygon.
    """
    root = ET.fromstring(xml_content)
    marks = _parse_marks(root)
    boundary = _parse_boundary(root)

    for m in ("SL1", "SL2", "M1"):
        if m not in marks:
            raise ValueError(f"Missing mark {m} in XML")

    SL1, SL2, M1 = marks["SL1"], marks["SL2"], marks["M1"]
    clat = 0.5 * (SL1[0] + SL2[0])
    clon = 0.5 * (SL1[1] + SL2[1])

    sl1x, sl1y = latlon_to_xy(SL1[0], SL1[1], clat, clon)
    sl2x, sl2y = latlon_to_xy(SL2[0], SL2[1], clat, clon)
    m1x, m1y = latlon_to_xy(M1[0], M1[1], clat, clon)

    # Boundary = the start box (race committee may not close the box)
    # Use boundary vertices as-is — do NOT prepend SL1/SL2
    box_polygon = []
    for v in boundary:
        bx, by = latlon_to_xy(v["lat"], v["lon"], clat, clon)
        box_polygon.append({"name": f"B{v['seq']}", "x": bx, "y": by})

    # TWD reference: bearing from LG1 to WG1
    ref_twd = None
    if "LG1" in marks and "WG1" in marks:
        ref_twd = bearing_ll(*marks["LG1"], *marks["WG1"])

    return {
        "sl1": (sl1x, sl1y), "sl2": (sl2x, sl2y), "m1": (m1x, m1y),
        "sl1_ll": SL1, "sl2_ll": SL2, "m1_ll": M1,
        "box_polygon": box_polygon, "center": (clat, clon),
        "ref_twd": ref_twd, "all_marks": marks,
    }


def bearing_ll(lat1, lon1, lat2, lon2):
    """Compass bearing (degrees) from (lat1,lon1) to (lat2,lon2)."""
    dlon = np.radians(lon2 - lon1)
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    x = np.sin(dlon) * np.cos(lat2r)
    y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    return float((np.degrees(np.arctan2(x, y)) + 360) % 360)


# ── Geometry helpers ─────────────────────────────────────────────────────────

def latlon_to_xy(lat, lon, lat0, lon0):
    x = EARTH_R * np.radians(lon - lon0) * np.cos(np.radians(lat0))
    y = EARTH_R * np.radians(lat - lat0)
    return float(x), float(y)


def rotate_xy(x, y, angle_deg):
    a = np.radians(angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    return float(x * ca - y * sa), float(x * sa + y * ca)


def bearing_xy(x1, y1, x2, y2):
    return float((np.degrees(np.arctan2(x2 - x1, y2 - y1)) + 360) % 360)


def distance_xy(x1, y1, x2, y2):
    return float(np.hypot(x2 - x1, y2 - y1))


def compute_twa(heading_deg, twd_deg):
    diff = (heading_deg - twd_deg + 360) % 360
    return float(diff if diff <= 180 else 360 - diff)


def normalize_angle(a):
    return float(((a + 180) % 360) - 180)


# ── Race data fetching ───────────────────────────────────────────────────────

def fetch_race_geometry(race_id: str) -> dict:
    resp = requests.get(f"{XML_URL}/{race_id}", timeout=15)
    resp.raise_for_status()
    all_xmls = json.loads(resp.content.decode("utf-8", errors="ignore")).get("xmls", [])
    if not all_xmls:
        raise ValueError(f"No XML data for race {race_id}")
    post_xml = all_xmls[-1]["xml"]
    post_bnd = _parse_boundary(ET.fromstring(post_xml))
    pre_xml = post_xml
    for entry in reversed(all_xmls):
        if len(_parse_boundary(ET.fromstring(entry["xml"]))) > len(post_bnd):
            pre_xml = entry["xml"]
            break
    marks = _parse_marks(ET.fromstring(pre_xml))
    pre_bnd = _parse_boundary(ET.fromstring(pre_xml))
    post_bnd2 = _parse_boundary(ET.fromstring(post_xml))
    for m in ("SL1", "SL2", "M1"):
        if m not in marks:
            raise ValueError(f"Missing mark {m} in race {race_id}")
    SL1, SL2, M1 = marks["SL1"], marks["SL2"], marks["M1"]
    clat = 0.5 * (SL1[0] + SL2[0])
    clon = 0.5 * (SL1[1] + SL2[1])
    sl1x, sl1y = latlon_to_xy(SL1[0], SL1[1], clat, clon)
    sl2x, sl2y = latlon_to_xy(SL2[0], SL2[1], clat, clon)
    m1x, m1y = latlon_to_xy(M1[0], M1[1], clat, clon)
    for v in pre_bnd:
        v["x"], v["y"] = latlon_to_xy(v["lat"], v["lon"], clat, clon)
    box_polygon = _build_start_box(pre_bnd, post_bnd2, sl1x, sl1y, sl2x, sl2y)
    return {
        "sl1": (sl1x, sl1y), "sl2": (sl2x, sl2y), "m1": (m1x, m1y),
        "sl1_ll": SL1, "sl2_ll": SL2, "m1_ll": M1,
        "box_polygon": box_polygon, "center": (clat, clon),
    }


def _parse_marks(root):
    marks = {}
    course = root.find("Course")
    if course is not None:
        for cm in course.findall("CompoundMark"):
            for m in cm.findall("Mark"):
                lat, lon = m.attrib.get("TargetLat"), m.attrib.get("TargetLng")
                if lat and lon:
                    marks[m.attrib.get("Name")] = (float(lat), float(lon))
    return marks


def _parse_boundary(root):
    boundary = []
    for cl in root.findall("CourseLimit"):
        if cl.attrib.get("name") == "Boundary":
            for lim in cl.findall("Limit"):
                boundary.append({
                    "seq": int(lim.attrib.get("SeqID", 0)),
                    "lat": float(lim.attrib["Lat"]),
                    "lon": float(lim.attrib["Lon"]),
                })
    return boundary


def _build_start_box(pre_bnd, post_bnd, sl1x, sl1y, sl2x, sl2y):
    post_latlons = {(round(v["lat"], 6), round(v["lon"], 6)) for v in post_bnd}
    removed_seqs = {
        v["seq"] for v in pre_bnd
        if (round(v["lat"], 6), round(v["lon"], 6)) not in post_latlons
    }
    if not removed_seqs:
        return []
    n = len(pre_bnd)
    removed_idx = [i for i, v in enumerate(pre_bnd) if v["seq"] in removed_seqs]
    anchor_before = (min(removed_idx) - 1) % n
    anchor_after = (max(removed_idx) + 1) % n
    path = []
    idx = anchor_before
    while True:
        path.append(pre_bnd[idx])
        if idx == anchor_after:
            break
        idx = (idx + 1) % n
    d1 = np.hypot(path[0]["x"] - sl1x, path[0]["y"] - sl1y)
    d2 = np.hypot(path[0]["x"] - sl2x, path[0]["y"] - sl2y)
    ordered = list(reversed(path)) if d1 < d2 else path
    from_marks = [
        {"name": "SL1", "x": sl1x, "y": sl1y},
        {"name": "SL2", "x": sl2x, "y": sl2y},
    ]
    for v in ordered:
        from_marks.append({"name": f"B{v['seq']}", "x": v["x"], "y": v["y"]})
    return from_marks


# ── Strategy calculations ────────────────────────────────────────────────────

def compute_line_bias(sl1, sl2, twd):
    sl_bearing = bearing_xy(sl2[0], sl2[1], sl1[0], sl1[1])
    bias_deg = normalize_angle(twd - sl_bearing)
    sl_length = distance_xy(*sl1, *sl2)
    bias_m = sl_length * np.sin(np.radians(abs(bias_deg)))
    if bias_deg > 0:
        favored = "SL1 (starboard)"
    elif bias_deg < 0:
        favored = "SL2 (port / pin)"
    else:
        favored = "Square"
    return {
        "bias_deg": round(bias_deg, 1), "bias_m": round(bias_m, 1),
        "favored": favored, "sl_bearing": round(sl_bearing, 1),
        "sl_length": round(sl_length, 1),
    }


def segment_info(origin, target, twd, tws, polar):
    brg = bearing_xy(*origin, *target)
    twa = compute_twa(brg, twd)
    dist = distance_xy(*origin, *target)
    opt = polar.optimal_upwind_twa(tws)
    if twa >= opt:
        speed = polar.boat_speed_ms(tws, twa)
        time_s = dist / speed if speed > 0 else float("inf")
        direct = True
    else:
        uw_x = np.sin(np.radians(twd))
        uw_y = np.cos(np.radians(twd))
        upwind_dist = (target[0] - origin[0]) * uw_x + (target[1] - origin[1]) * uw_y
        vmg = polar.vmg_upwind_ms(tws)
        time_s = max(upwind_dist, 0.1) / vmg if vmg > 0 else float("inf")
        direct = False
    return {
        "twa": round(twa, 1), "time_s": round(time_s, 1),
        "distance_m": round(dist, 1), "bearing": round(brg, 1),
        "is_direct": direct,
    }


def twa_time_to_m1(mark, m1, twd, tws, polar):
    return segment_info(mark, m1, twd, tws, polar)


def fastest_line_point(sl1, sl2, m1, twd, tws, polar, n=300):
    best_t, best_time = 0.0, float("inf")
    best_pt = sl1
    all_times = []
    for i in range(n + 1):
        frac = i / n
        px = sl1[0] + frac * (sl2[0] - sl1[0])
        py = sl1[1] + frac * (sl2[1] - sl1[1])
        info = segment_info((px, py), m1, twd, tws, polar)
        all_times.append(info["time_s"])
        if info["time_s"] < best_time:
            best_time = info["time_s"]
            best_t = frac
            best_pt = (px, py)
    return {
        "t": round(best_t, 3), "point": best_pt,
        "time_s": round(best_time, 1), "all_times": all_times,
    }


# ── Entry point = d metres below pin on the downwind side ───────────────────

def entry_point(sl2, d_below_pin, twd):
    """Entry position: d metres directly downwind of the pin (SL2)."""
    dw_x = -np.sin(np.radians(twd))
    dw_y = -np.cos(np.radians(twd))
    return (sl2[0] + d_below_pin * dw_x, sl2[1] + d_below_pin * dw_y)


# ── X point on the RIGHT (deep) boundary of the box ────────────────────────

def _right_boundary_segs(box_polygon, twd):
    """Return boundary segments sorted by mid_xr descending (rightmost first)."""
    n = len(box_polygon)
    segs = []
    for i in range(n):
        j = (i + 1) % n
        if {box_polygon[i]["name"], box_polygon[j]["name"]} == {"SL1", "SL2"}:
            continue
        p1r = rotate_xy(box_polygon[i]["x"], box_polygon[i]["y"], twd)
        p2r = rotate_xy(box_polygon[j]["x"], box_polygon[j]["y"], twd)
        mid_xr = 0.5 * (p1r[0] + p2r[0])
        segs.append({"i": i, "j": j, "p1r": p1r, "p2r": p2r, "mid_xr": mid_xr})
    segs.sort(key=lambda s: s["mid_xr"], reverse=True)
    return segs


def x_on_boundary(box_polygon, sl2, d_below_pin, twd):
    """
    Place X on the RIGHT (deep) boundary of the box.
    d_below_pin sets X's height: yr = SL2_yr - d.
    Intersects with rightmost boundary segments.
    Falls back to the rightmost vertex if d exceeds the boundary.
    """
    if not box_polygon:
        return entry_point(sl2, d_below_pin, twd)

    sl2r = rotate_xy(*sl2, twd)
    target_yr = sl2r[1] - d_below_pin

    segs = _right_boundary_segs(box_polygon, twd)
    if not segs:
        return entry_point(sl2, d_below_pin, twd)

    # Use the top 3 rightmost segments as the "right boundary"
    right_segs = segs[:3]

    candidates = []
    for s in right_segs:
        p1r, p2r = s["p1r"], s["p2r"]
        y1, y2 = p1r[1], p2r[1]
        if min(y1, y2) <= target_yr <= max(y1, y2):
            if abs(y2 - y1) < 1e-6:
                t = 0.5
            else:
                t = (target_yr - y1) / (y2 - y1)
            i, j = s["i"], s["j"]
            ox = box_polygon[i]["x"] + t * (box_polygon[j]["x"] - box_polygon[i]["x"])
            oy = box_polygon[i]["y"] + t * (box_polygon[j]["y"] - box_polygon[i]["y"])
            ixr = p1r[0] + t * (p2r[0] - p1r[0])
            candidates.append((ox, oy, ixr))

    if candidates:
        # pick rightmost intersection
        best = max(candidates, key=lambda c: c[2])
        return (best[0], best[1])

    # d out of range — snap to deepest right vertex
    all_pts = []
    for s in right_segs:
        for k in (s["i"], s["j"]):
            all_pts.append((box_polygon[k]["x"], box_polygon[k]["y"],
                            rotate_xy(box_polygon[k]["x"], box_polygon[k]["y"], twd)[1]))
    # find vertex closest in y to target_yr
    nearest = min(all_pts, key=lambda p: abs(p[2] - target_yr))
    return (nearest[0], nearest[1])


def x_boundary_depth_range(box_polygon, sl2, twd):
    """
    Min and max d_below_pin that keeps X on the right boundary.
    Returns (d_min, d_max).
    """
    if not box_polygon:
        return (0.0, 300.0)
    sl2r = rotate_xy(*sl2, twd)
    segs = _right_boundary_segs(box_polygon, twd)
    right_segs = segs[:3]
    all_yr = []
    for s in right_segs:
        all_yr.append(s["p1r"][1])
        all_yr.append(s["p2r"][1])
    if not all_yr:
        return (0.0, 300.0)
    max_yr = max(all_yr)
    min_yr = min(all_yr)
    d_min = round(sl2r[1] - max_yr, 0)  # can be negative (X above SL2)
    d_max = round(sl2r[1] - min_yr, 0)
    return (d_min, d_max)


def ttk_from_x(x_pt, target, twd, tws, polar, time_to_start):
    info = segment_info(x_pt, target, twd, tws, polar)
    t = info["time_s"]
    return {
        "time_to_line": t,
        "ttk": round(time_to_start - t, 1),
        "ratio": round(time_to_start / t, 2) if t > 0 else float("inf"),
        "twa": info["twa"], "distance_m": info["distance_m"],
    }


# ── Laylines ────────────────────────────────────────────────────────────────

def laylines(sl1, sl2, twd, tws, polar, length=500):
    opt = polar.optimal_upwind_twa(tws)
    approach_from = (twd - opt + 180 + 360) % 360
    lines = []
    for mark, name in [(sl1, "SL1"), (sl2, "SL2")]:
        ex = mark[0] + length * np.sin(np.radians(approach_from))
        ey = mark[1] + length * np.cos(np.radians(approach_from))
        lines.append({"start": mark, "end": (ex, ey), "label": f"{name} layline"})
    return lines


# ── Depth segments ──────────────────────────────────────────────────────────

def depth_segments(box_polygon, sl1, sl2, twd, tws, polar):
    if not box_polygon:
        return []
    n = len(box_polygon)
    results = []
    for mark, name in [(sl1, "SL1"), (sl2, "SL2")]:
        mr = rotate_xy(*mark, twd)
        best_xr, best_orig = None, None
        for i in range(n):
            j = (i + 1) % n
            if {box_polygon[i]["name"], box_polygon[j]["name"]} == {"SL1", "SL2"}:
                continue
            p1r = rotate_xy(box_polygon[i]["x"], box_polygon[i]["y"], twd)
            p2r = rotate_xy(box_polygon[j]["x"], box_polygon[j]["y"], twd)
            y1, y2 = p1r[1], p2r[1]
            if abs(y2 - y1) < 1e-6 or not (min(y1, y2) <= mr[1] <= max(y1, y2)):
                continue
            t = (mr[1] - y1) / (y2 - y1)
            ix = p1r[0] + t * (p2r[0] - p1r[0])
            if ix > mr[0] and (best_xr is None or ix > best_xr):
                best_xr = ix
                ox = box_polygon[i]["x"] + t * (box_polygon[j]["x"] - box_polygon[i]["x"])
                oy = box_polygon[i]["y"] + t * (box_polygon[j]["y"] - box_polygon[i]["y"])
                best_orig = (ox, oy)
        if best_orig:
            info = segment_info(mark, best_orig, twd, tws, polar)
            results.append({
                "mark": name, "start": mark, "end": best_orig,
                "twa": info["twa"], "time_s": info["time_s"],
                "distance_m": info["distance_m"],
            })
    return results


def box_entry_depth(box_polygon, sl1, sl2, twd):
    if not box_polygon:
        return 0.0
    sl1r = rotate_xy(*sl1, twd)
    sl2r = rotate_xy(*sl2, twd)
    line_yr = min(sl1r[1], sl2r[1])
    min_yr = min(rotate_xy(p["x"], p["y"], twd)[1] for p in box_polygon)
    return round(line_yr - min_yr, 1)
