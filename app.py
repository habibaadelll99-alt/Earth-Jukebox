import os
import numpy as np
from PIL import Image
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "static", "images")

AW, AH = 1200, 600
SILENCE_THRESHOLD = 0.08

SCENES = {
    "lights": "nightlights.png",
    "lightning": "lightning.png",
    "japan": "jaban.png",
    "mediterranean": "BlackMarble_mediterranean.png",
    "emissions": "emissionsGlobe_12_2021.png",
}

scene_rgb = {}
LIT_MIN = 45  # pixels darker than this (max channel, 0-255) are treated as dark and ignored


def load_scene(scene, path):
    if not os.path.exists(path):
        print(f"[WARN] Missing image: {path}")
        return np.zeros((AH, AW, 3), dtype=np.uint8)
    arr = np.array(Image.open(path).convert("RGB").resize((AW, AH)), dtype=np.uint8)
    if scene == "japan":
        arr[int(AH * 0.92):, int(AW * 0.88):] = 0  # ignore corner logo
    if scene == "emissions":
        grey = np.abs(arr[..., 0].astype(int) - arr[..., 2].astype(int)) < 25
        arr[grey] = 0  # grey coastlines are not data
    return arr


for name, filename in SCENES.items():
    scene_rgb[name] = load_scene(name, os.path.join(IMAGES_DIR, filename))


def sample(scene, px, py, vw=1.0, vh=1.0):
    """Returns (level, tone, heat) for the lit pixels around a point.
    level: brightness x coverage (0..1), 0 when the area is dark.
    tone:  blue-vs-red balance (-1 red/orange .. +1 blue/purple).
    heat:  green-vs-red ratio (0 deep red .. 1 yellow/white)."""
    arr = scene_rgb[scene]
    hw = max(3, int(vw * AW * 0.012))
    hh = max(3, int(vh * AH * 0.012))
    cx, cy = int(px * (AW - 1)), int(py * (AH - 1))
    win = arr[max(0, cy - hh):cy + hh + 1, max(0, cx - hw):cx + hw + 1].reshape(-1, 3).astype(np.float32)
    if win.size == 0:
        return 0.0, 0.0, 0.0
    lit = win[win.max(axis=1) >= LIT_MIN]
    if len(lit) == 0:
        return 0.0, 0.0, 0.0
    coverage = len(lit) / len(win)
    brightness = float(lit.max(axis=1).mean() / 255.0)
    level = min(1.0, brightness * (0.4 + 0.6 * coverage ** 0.5))
    r, g, b = (float(v) for v in lit.mean(axis=0))
    tone = (b - r) / (b + r + 1e-6)
    heat = min(1.0, g / max(r, 1.0))
    return level, tone, heat


def num(d, key, default):
    try:
        return float(d.get(key, default))
    except (TypeError, ValueError):
        return default


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/intensity", methods=["POST"])
def intensity():
    d = request.get_json(silent=True) or {}
    scene = d.get("scene", "lights")
    if scene not in scene_rgb:
        return jsonify({"level": 0.0, "tone": 0.0, "heat": 0.0})
    px = min(1.0, max(0.0, num(d, "px", 0.5)))
    py = min(1.0, max(0.0, num(d, "py", 0.5)))
    vw = min(1.0, max(0.01, num(d, "vw", 1.0)))
    vh = min(1.0, max(0.01, num(d, "vh", 1.0)))
    level, tone, heat = sample(scene, px, py, vw, vh)
    return jsonify({"level": level, "tone": tone, "heat": heat})


# (west, east, north, south) in degrees. Adjust japan / mediterranean / lightning
# to match the real extent of your images so place names are accurate.
EXTENTS = {
    "lights": (-180, 180, 90, -90),
    "lightning": (-180, 180, 90, -90),
    "emissions": (-180, 180, 90, -90),
    "japan": (122, 148, 47, 30),
    "mediterranean": (-10, 40, 47, 28),
}

PLACES = [
    ("Cairo", "Egypt", 30.04, 31.24), ("Alexandria", "Egypt", 31.20, 29.92),
    ("Lagos", "Nigeria", 6.52, 3.38), ("Johannesburg", "South Africa", -26.2, 28.05),
    ("Nairobi", "Kenya", -1.29, 36.82), ("Casablanca", "Morocco", 33.57, -7.59),
    ("Addis Ababa", "Ethiopia", 9.03, 38.74), ("Kinshasa", "DR Congo", -4.44, 15.27),
    ("Algiers", "Algeria", 36.75, 3.06), ("Tunis", "Tunisia", 36.81, 10.18),
    ("Riyadh", "Saudi Arabia", 24.71, 46.68), ("Dubai", "UAE", 25.2, 55.27),
    ("Tehran", "Iran", 35.69, 51.39), ("Baghdad", "Iraq", 33.31, 44.36),
    ("Tel Aviv", "Israel", 32.08, 34.78), ("Istanbul", "Turkey", 41.01, 28.98),
    ("Athens", "Greece", 37.98, 23.73), ("Rome", "Italy", 41.9, 12.5),
    ("Madrid", "Spain", 40.42, -3.7), ("Paris", "France", 48.86, 2.35),
    ("London", "UK", 51.51, -0.13), ("Berlin", "Germany", 52.52, 13.4),
    ("Kyiv", "Ukraine", 50.45, 30.52), ("Moscow", "Russia", 55.76, 37.62),
    ("Novosibirsk", "Russia", 55.03, 82.92), ("Stockholm", "Sweden", 59.33, 18.07),
    ("Karachi", "Pakistan", 24.86, 67.0), ("Delhi", "India", 28.61, 77.21),
    ("Mumbai", "India", 19.08, 72.88), ("Kolkata", "India", 22.57, 88.36),
    ("Dhaka", "Bangladesh", 23.81, 90.41), ("Bangkok", "Thailand", 13.76, 100.5),
    ("Jakarta", "Indonesia", -6.21, 106.85), ("Singapore", "Singapore", 1.35, 103.82),
    ("Manila", "Philippines", 14.6, 120.98), ("Beijing", "China", 39.9, 116.4),
    ("Shanghai", "China", 31.23, 121.47), ("Guangzhou", "China", 23.13, 113.26),
    ("Hong Kong", "China", 22.32, 114.17), ("Seoul", "South Korea", 37.57, 126.98),
    ("Tokyo", "Japan", 35.68, 139.69), ("Osaka", "Japan", 34.69, 135.5),
    ("Sydney", "Australia", -33.87, 151.21), ("Melbourne", "Australia", -37.81, 144.96),
    ("Auckland", "New Zealand", -36.85, 174.76), ("Vancouver", "Canada", 49.28, -123.12),
    ("Toronto", "Canada", 43.65, -79.38), ("Los Angeles", "USA", 34.05, -118.24),
    ("Chicago", "USA", 41.88, -87.63), ("Houston", "USA", 29.76, -95.37),
    ("New York", "USA", 40.71, -74.01), ("Mexico City", "Mexico", 19.43, -99.13),
    ("Bogota", "Colombia", 4.71, -74.07), ("Lima", "Peru", -12.05, -77.04),
    ("Sao Paulo", "Brazil", -23.55, -46.63), ("Rio de Janeiro", "Brazil", -22.91, -43.17),
    ("Buenos Aires", "Argentina", -34.6, -58.38), ("Santiago", "Chile", -33.45, -70.67),
]

TEXT = {
    "lights": ["Almost no artificial light here.", "A scatter of small settlements.",
               "A busy urban belt glowing in the dark.", "A dense metropolitan core, among the brightest spots at night."],
    "lightning": ["Calm skies, no flashes detected.", "Occasional distant flashes.",
                  "An active storm cell with frequent strikes.", "A violent thunderstorm, strikes nearly continuous."],
    "japan": ["Dark sea or mountains, very little light.", "Small towns and fishing villages.",
              "A busy coastal conurbation.", "A megacity core, packed with light."],
    "mediterranean": ["Open water, only faint ship lights.", "Quiet coastline and island villages.",
                      "Busy coastal towns and ports.", "A major port city blazing on the shore."],
    "emissions": ["Negligible fossil CO2 emissions.", "Light emissions from scattered settlements.",
                  "Significant emissions from industry and traffic.", "A major emissions hotspot of power plants, industry and dense cities."],
}


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


@app.route("/api/point", methods=["POST"])
def point():
    d = request.get_json(silent=True) or {}
    scene = d.get("scene", "lights")
    if scene not in scene_rgb:
        return jsonify({"title": "Unknown", "coords": "", "level": 0, "text": ""})
    px = min(1.0, max(0.0, num(d, "px", 0.5)))
    py = min(1.0, max(0.0, num(d, "py", 0.5)))

    level, tone, heat = sample(scene, px, py, 1.0, 1.0)

    w, e, n, s = EXTENTS.get(scene, EXTENTS["lights"])
    lon, lat = w + px * (e - w), n - py * (n - s)

    name, country, km = min(
        ((c[0], c[1], float(haversine(lat, lon, c[2], c[3]))) for c in PLACES), key=lambda t: t[2]
    )
    if km <= 300:
        title = f"{name}, {country}"
    elif km <= 1500:
        title = f"About {round(km, -1):.0f} km from {name}"
    else:
        title = "Remote region or open ocean"

    band = 0 if level < 0.08 else 1 if level < 0.3 else 2 if level < 0.7 else 3
    coords = f"{abs(lat):.1f}°{'N' if lat >= 0 else 'S'}, {abs(lon):.1f}°{'E' if lon >= 0 else 'W'}"
    return jsonify({"title": title, "coords": coords, "level": round(level * 100), "text": TEXT[scene][band], "tone": tone, "heat": heat})


if __name__ == "__main__":
    app.run(debug=True, port=5000)