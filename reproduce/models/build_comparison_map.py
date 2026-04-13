#!/usr/bin/env python3
"""
Build an interactive deck.gl / MapLibre comparison viewer for Nuremberg.

Uses the same rendering approach as the TerraPulse dashboard:
  - MapLibre GL for ESRI satellite tiles (zoomable, high-res)
  - deck.gl BitmapLayer with nearest-neighbor filtering for pixel-perfect overlays
  - 4-corner georeferencing for accurate projection (no stretching/shaking)
"""

import gc
import os
import sys
import time
import base64
from io import BytesIO

import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from reproduce.models.shared.config import N_CLASSES, CLASS_NAMES

PRED_DIR = os.path.join(PROJECT_ROOT, "reproduce", "models", "prediction_maps")

# 4-corner georeferencing computed from the actual S2 raster (EPSG:32632 → WGS84)
# Image orientation: row 0 = north (top), row H = south (bottom)
# deck.gl BitmapLayer bounds order: [bottomLeft, topLeft, topRight, bottomRight]
# which is [SW, NW, NE, SE]
WGS84_CORNERS = [
    [10.944405, 49.379838],  # SW (bottom-left of image)
    [10.950141, 49.524573],  # NW (top-left of image)
    [11.206967, 49.519957],  # NE (top-right of image)
    [11.200478, 49.375246],  # SE (bottom-right of image)
]

# Class colors RGBA
CLASS_COLORS_RGBA = [
    ( 20, 200,  20, 160),   # tree_cover
    (210, 180,  50, 180),   # shrubland
    (140, 255, 100, 160),   # grassland
    (255, 240,  50, 160),   # cropland
    (255,  40,  40, 180),   # built_up
    (200, 200, 200, 160),   # bare_sparse
    ( 40, 100, 255, 180),   # water
]

ORIG_COLORS = np.array([
    [  0, 128,   0], [170, 160,  60], [152, 230, 100], [255, 225,  80],
    [220,  40,  40], [180, 180, 180], [ 30,  80, 220],
], dtype=np.uint8)


def ts():
    return time.strftime("%H:%M:%S")


def reconstruct_class_map(png_path):
    img = np.array(Image.open(png_path))
    H = img.shape[0] - 40
    rgb = img[:H]
    pred = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for ci in range(7):
        mask = np.all(rgb == ORIG_COLORS[ci], axis=2)
        pred[mask] = ci
    return pred


def class_map_to_rgba_png_b64(pred):
    """Convert class map to RGBA PNG as base64 data URI. No upscaling — deck.gl
    handles nearest-neighbor magnification via texture params."""
    H, W = pred.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    for ci in range(N_CLASSES):
        mask = pred == ci
        rgba[mask] = CLASS_COLORS_RGBA[ci]
    img = Image.fromarray(rgba)
    buf = BytesIO()
    img.save(buf, format='PNG', optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return f"data:image/png;base64,{b64}"


def diff_map_to_rgba_png_b64(v3_pred, v5_pred):
    """Diff overlay: only disagreement pixels visible, colored by V5."""
    H, W = v3_pred.shape
    disagree = (v3_pred != v5_pred) & (v3_pred != 255) & (v5_pred != 255)
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    for ci in range(N_CLASSES):
        mask = disagree & (v5_pred == ci)
        rgba[mask] = (*CLASS_COLORS_RGBA[ci][:3], 220)
    img = Image.fromarray(rgba)
    buf = BytesIO()
    img.save(buf, format='PNG', optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return f"data:image/png;base64,{b64}"


def build_html(layer_uris):
    corners_json = str(WGS84_CORNERS)

    # Build layer entries for JS
    layer_entries = []
    for key, name, uri in layer_uris:
        layer_entries.append(f'{{ key: "{key}", name: "{name}", uri: "{uri}" }}')
    layers_js = ",\n            ".join(layer_entries)

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Nuremberg Model Comparison</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://unpkg.com/maplibre-gl@4.1.2/dist/maplibre-gl.js"></script>
    <link href="https://unpkg.com/maplibre-gl@4.1.2/dist/maplibre-gl.css" rel="stylesheet" />
    <script src="https://unpkg.com/deck.gl@9.0.16/dist.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #0a0a0f; color: #e0e0e0; }}
        #map {{ width: 100vw; height: 100vh; }}

        .panel {{
            position: absolute;
            background: rgba(12, 12, 20, 0.94);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.5);
            padding: 16px 20px;
            z-index: 10;
        }}
        .panel h4 {{
            margin: 0 0 12px;
            font-size: 14px;
            font-weight: 600;
            color: #fff;
            letter-spacing: 0.3px;
        }}

        /* Controls */
        #controls {{ top: 16px; right: 16px; min-width: 270px; }}
        .layer-row {{
            display: flex; align-items: center; gap: 10px;
            padding: 5px 0; cursor: pointer; font-size: 13px;
            transition: color 0.15s;
        }}
        .layer-row:hover {{ color: #fff; }}
        .layer-row input[type="radio"] {{
            accent-color: #4a9eff; width: 16px; height: 16px; cursor: pointer;
            margin: 0;
        }}
        .layer-row label {{ cursor: pointer; }}

        .slider-group {{
            margin-top: 14px; padding-top: 12px;
            border-top: 1px solid rgba(255,255,255,0.08);
        }}
        .slider-group label {{
            display: block; font-size: 12px; color: #888; margin-bottom: 6px;
        }}
        .slider-group .val {{ color: #6eaaff; float: right; font-weight: 500; }}
        .slider-group input[type="range"] {{
            width: 100%; accent-color: #4a9eff; height: 4px;
        }}

        /* Legend */
        #legend {{ bottom: 16px; right: 16px; }}
        .legend-item {{ display: flex; align-items: center; gap: 8px; font-size: 13px; line-height: 1.8; }}
        .swatch {{
            width: 20px; height: 14px; border-radius: 3px; flex-shrink: 0;
            border: 1px solid rgba(255,255,255,0.12);
        }}

        /* Pixel info tooltip */
        #info {{ bottom: 16px; left: 16px; font-size: 12px; color: #999; display: none; }}
        #info strong {{ color: #fff; }}
    </style>
</head>
<body>
<div id="map"></div>

<div id="controls" class="panel">
    <h4>Prediction Layers</h4>
    <div id="layerList"></div>
    <div class="slider-group">
        <label>Overlay Opacity <span class="val" id="opVal">60%</span></label>
        <input type="range" id="opSlider" min="0" max="100" value="60">
    </div>
</div>

<div id="legend" class="panel">
    <h4>Land Cover</h4>
    <div class="legend-item"><div class="swatch" style="background:rgb(20,200,20)"></div>Tree Cover</div>
    <div class="legend-item"><div class="swatch" style="background:rgb(210,180,50)"></div>Shrubland</div>
    <div class="legend-item"><div class="swatch" style="background:rgb(140,255,100)"></div>Grassland</div>
    <div class="legend-item"><div class="swatch" style="background:rgb(255,240,50)"></div>Cropland</div>
    <div class="legend-item"><div class="swatch" style="background:rgb(255,40,40)"></div>Built-up</div>
    <div class="legend-item"><div class="swatch" style="background:rgb(200,200,200)"></div>Bare/Sparse</div>
    <div class="legend-item"><div class="swatch" style="background:rgb(40,100,255)"></div>Water</div>
</div>

<div id="info" class="panel"></div>

<script>
// ── Layer data ──
const LAYERS = [
    {layers_js}
];
const CORNERS = {corners_json};
// deck.gl BitmapLayer bounds: [SW, NW, NE, SE]

// ── Map ──
const ESRI_STYLE = {{
    version: 8,
    sources: {{
        'esri': {{
            type: 'raster',
            tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}'],
            tileSize: 256,
            attribution: '&copy; Esri, Maxar, Earthstar Geographics',
        }},
    }},
    layers: [{{ id: 'esri-sat', type: 'raster', source: 'esri', minzoom: 0, maxzoom: 19 }}],
}};

let activeKey = null;
let opacity = 0.6;
let deckgl = null;

function createDeckLayer(uri) {{
    return new deck.BitmapLayer({{
        id: 'prediction-overlay',
        image: uri,
        bounds: CORNERS,
        opacity: opacity,
        textureParameters: {{
            [globalThis.luma?.GL?.TEXTURE_MIN_FILTER || 0x2600]: globalThis.luma?.GL?.NEAREST || 0x2600,
            [globalThis.luma?.GL?.TEXTURE_MAG_FILTER || 0x2800]: globalThis.luma?.GL?.NEAREST || 0x2600,
        }},
    }});
}}

function updateDeck() {{
    const layerInfo = LAYERS.find(l => l.key === activeKey);
    if (!layerInfo) {{
        deckgl.setProps({{ layers: [] }});
        return;
    }}
    deckgl.setProps({{ layers: [createDeckLayer(layerInfo.uri)] }});
}}

// ── Initialize ──
window.addEventListener('DOMContentLoaded', () => {{
    deckgl = new deck.DeckGL({{
        container: 'map',
        mapStyle: ESRI_STYLE,
        mapLib: maplibregl,
        initialViewState: {{
            longitude: 11.13,
            latitude: 49.44,
            zoom: 13,
            pitch: 0,
            bearing: 0,
        }},
        controller: true,
        layers: [],
    }});

    // Build radio buttons
    const list = document.getElementById('layerList');
    LAYERS.forEach(l => {{
        const row = document.createElement('div');
        row.className = 'layer-row';
        row.innerHTML = `<input type="radio" name="layer" id="r_${{l.key}}" value="${{l.key}}"><label for="r_${{l.key}}">${{l.name}}</label>`;
        list.appendChild(row);
    }});
    // "None" option
    const noneRow = document.createElement('div');
    noneRow.className = 'layer-row';
    noneRow.innerHTML = '<input type="radio" name="layer" id="r_none" value="none" checked><label for="r_none">None (satellite only)</label>';
    list.appendChild(noneRow);

    // Events
    list.addEventListener('change', e => {{
        activeKey = e.target.value === 'none' ? null : e.target.value;
        updateDeck();
    }});

    document.getElementById('opSlider').addEventListener('input', e => {{
        opacity = parseInt(e.target.value) / 100;
        document.getElementById('opVal').textContent = e.target.value + '%';
        updateDeck();
    }});
}});
</script>
</body>
</html>"""


def main():
    os.makedirs(PRED_DIR, exist_ok=True)

    print(f"[{ts()}] Building deck.gl comparison viewer...")

    print(f"[{ts()}] Loading prediction maps...")
    v3_pred = reconstruct_class_map(os.path.join(PRED_DIR, "nuremberg_v3.png"))
    v5_pred = reconstruct_class_map(os.path.join(PRED_DIR, "nuremberg_v5.png"))
    cb_pred = reconstruct_class_map(os.path.join(PRED_DIR, "nuremberg_catboost_v5.png"))

    # Load ESA WorldCover labels
    from reproduce.models.shared.config import get_train_cities, get_val_cities, get_test_cities
    all_cities = get_train_cities() + get_val_cities() + get_test_cities()
    nuremberg = [c for c in all_cities if c.name == "nuremberg"][0]
    H, W = v3_pred.shape
    try:
        from reproduce.models.shared.data import load_pixel_labels
        wc_labels = load_pixel_labels(nuremberg, year=2021)
        if wc_labels is not None:
            print(f"  ESA labels: {wc_labels.shape}")
        else:
            wc_labels = np.full((H, W), 255, dtype=np.uint8)
    except Exception:
        wc_labels = np.full((H, W), 255, dtype=np.uint8)

    print(f"[{ts()}] Encoding layers as PNG data URIs (no upscale - deck.gl does nearest)...")
    uris = []

    uri = class_map_to_rgba_png_b64(v3_pred)
    print(f"  V3: {len(uri)//1024:,} KB")
    uris.append(("v3", "SSNet V3 (86.4% acc, 2.8M params)", uri))

    uri = class_map_to_rgba_png_b64(v5_pred)
    print(f"  V5: {len(uri)//1024:,} KB")
    uris.append(("v5", "SSNet V5 (86.5% acc, 1.2M params)", uri))

    uri = class_map_to_rgba_png_b64(cb_pred)
    print(f"  CB: {len(uri)//1024:,} KB")
    uris.append(("cb", "CatBoost pixel_v5 (81.7% acc)", uri))

    uri = diff_map_to_rgba_png_b64(v3_pred, v5_pred)
    print(f"  Diff: {len(uri)//1024:,} KB")
    uris.append(("diff", "V3 vs V5 Disagreement", uri))

    uri = class_map_to_rgba_png_b64(wc_labels)
    print(f"  Labels: {len(uri)//1024:,} KB")
    uris.append(("labels", "ESA WorldCover Ground Truth", uri))

    print(f"[{ts()}] Generating HTML...")
    html = build_html(uris)

    out_path = os.path.join(PRED_DIR, "nuremberg_comparison.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\n[{ts()}] Done! {out_path} ({size_mb:.1f} MB)")
    print("  Open in browser for pixel-perfect comparison.")


if __name__ == "__main__":
    main()
