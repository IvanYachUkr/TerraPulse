#!/usr/bin/env python3
"""
Build a multi-city deck.gl map for SpectralSpatialNet v8 predictions.

For each of the 6 test cities, renders:
  - SSNet v8 predictions
  - ESA WorldCover ground truth labels

All overlaid on ESRI satellite tiles in a single interactive HTML.
"""

import gc
import os
import pickle
import sys
import time
import base64
from io import BytesIO

import numpy as np
import torch
from PIL import Image
import rasterio
from rasterio.warp import transform

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from reproduce.models.shared.config import (
    SEED, N_CLASSES, CLASS_NAMES,
    get_test_cities, city_has_raw_tifs,
)
from reproduce.models.shared.data import (
    load_raw_feature_cube, load_pixel_labels, compute_center_indices,
)

CKPT_DIR = os.path.join(PROJECT_ROOT, "reproduce", "models", "checkpoints")
OUT_DIR = os.path.join(PROJECT_ROOT, "reproduce", "models", "prediction_maps")

CLASS_COLORS_RGBA = [
    ( 20, 200,  20, 160),   # tree_cover
    (210, 180,  50, 180),   # shrubland
    (140, 255, 100, 160),   # grassland
    (255, 240,  50, 160),   # cropland
    (255,  40,  40, 180),   # built_up
    (200, 200, 200, 160),   # bare_sparse
    ( 40, 100, 255, 180),   # water
]


def ts():
    return time.strftime("%H:%M:%S")


def get_raster_corners_wgs84(city):
    """Get WGS84 4-corner coordinates from the S2 raster."""
    from reproduce.models.shared.data import raw_dir, YEARS, SEASONS
    rd = raw_dir(city)
    anchor = None
    for y in YEARS:
        for s in SEASONS:
            p = os.path.join(rd, f"sentinel2_{city.name}_{y}_{s}.tif")
            if os.path.exists(p):
                anchor = p
                break
        if anchor:
            break
    if not anchor:
        return None, 0, 0

    with rasterio.open(anchor) as src:
        crs = src.crs
        b = src.bounds
        H, W = src.height, src.width

    corners_x = [b.left, b.right, b.right, b.left]
    corners_y = [b.top, b.top, b.bottom, b.bottom]
    lons, lats = transform(crs, 'EPSG:4326', corners_x, corners_y)
    corners = [
        [lons[3], lats[3]],  # SW
        [lons[0], lats[0]],  # NW
        [lons[1], lats[1]],  # NE
        [lons[2], lats[2]],  # SE
    ]
    center_lon = sum(lons) / 4
    center_lat = sum(lats) / 4
    return corners, center_lon, center_lat, H, W


def class_map_to_png_b64(pred):
    H, W = pred.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    for ci in range(N_CLASSES):
        mask = pred == ci
        rgba[mask] = CLASS_COLORS_RGBA[ci]
    img = Image.fromarray(rgba)
    buf = BytesIO()
    img.save(buf, format='PNG', optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('ascii')


# ── Model loading ────────────────────────────────────────────────────────────

def load_ssnet_v8(device):
    from reproduce.models.architectures.spectral_spatial_v8 import SpectralSpatialNetV8
    model = SpectralSpatialNetV8(
        n_bands=12, n_timesteps=6, n_indices=145,
        spatial_dims=(32, 64, 128), expand_ratio=4,
        temporal_dim=128, n_attn_layers=2, n_heads=4,
        n_classes=7, dropout=0.12,
        prior_hidden=96,
    ).to(device)
    state = torch.load(os.path.join(CKPT_DIR, "ssnet_v8.pt"),
                       map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    with open(os.path.join(CKPT_DIR, "ssnet_v8_fixed_scaler.pkl"), "rb") as f:
        sc = pickle.load(f)
    return model, sc["patches"], sc["indices"]


# ── v8 prediction (needs center_raw) ────────────────────────────────────────

def predict_v8_full(model, ps, idx_s, cube, H, W, device):
    """Predict all pixels using SSNet v8, row by row with 3×3 patches."""
    padded = np.pad(cube, ((1, 1), (1, 1), (0, 0)),
                    mode='constant', constant_values=0.0)
    pred = np.full((H, W), 255, dtype=np.uint8)
    BATCH = 4096

    for r in range(H):
        row_patches = np.empty((W, 9 * 72), dtype=np.float32)
        for c in range(W):
            patch = padded[r:r+3, c:c+3, :]
            row_patches[c] = patch.reshape(-1)

        # Center pixel is patch position 4 (0-indexed), each pixel has 72 bands
        centers = row_patches[:, 4*72:5*72].copy()
        valid = np.isfinite(centers).any(axis=1)
        n_valid = valid.sum()
        if n_valid == 0:
            continue

        valid_patches = row_patches[valid]
        valid_centers = centers[valid]
        np.nan_to_num(valid_patches, nan=0.0, copy=False)
        np.nan_to_num(valid_centers, nan=0.0, copy=False)

        indices = compute_center_indices(valid_centers)
        patches_s = ps.transform(valid_patches).astype(np.float32)
        indices_s = idx_s.transform(indices).astype(np.float32)
        # center_raw is the unscaled center pixel for the prior head
        center_raw = valid_centers.astype(np.float32)

        preds_valid = np.empty(n_valid, dtype=np.uint8)
        with torch.no_grad():
            for s in range(0, n_valid, BATCH):
                e = min(s + BATCH, n_valid)
                xp = torch.from_numpy(patches_s[s:e]).to(device)
                xi = torch.from_numpy(indices_s[s:e]).to(device)
                xc = torch.from_numpy(center_raw[s:e]).to(device)
                out = model(xp, xi, xc)
                logits = out["logits"]
                preds_valid[s:e] = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
                del xp, xi, xc, out, logits

        pred[r, valid] = preds_valid

    return pred


# ── HTML generation ──────────────────────────────────────────────────────────

def build_html(cities_data):
    cities_js_parts = []
    for cd in cities_data:
        layers_js = []
        for key, name, uri in cd["layers"]:
            layers_js.append(f'{{ key: "{key}", name: "{name}", uri: "{uri}" }}')

        cities_js_parts.append(f"""{{
            name: "{cd['name']}",
            center: [{cd['center_lon']}, {cd['center_lat']}],
            corners: {cd['corners']},
            accuracy: "{cd.get('accuracy', '')}",
            layers: [{', '.join(layers_js)}],
        }}""")

    all_cities_js = ",\n        ".join(cities_js_parts)

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>SSNet v8 - All Test Cities</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://unpkg.com/maplibre-gl@4.1.2/dist/maplibre-gl.js"></script>
    <link href="https://unpkg.com/maplibre-gl@4.1.2/dist/maplibre-gl.css" rel="stylesheet" />
    <script src="https://unpkg.com/deck.gl@9.0.16/dist.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0a0a0f; color: #e0e0e0; }}
        #map {{ width: 100vw; height: 100vh; }}

        .panel {{
            position: absolute;
            background: rgba(12, 12, 20, 0.94);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.5);
            padding: 14px 18px;
            z-index: 10;
        }}
        .panel h4 {{
            margin: 0 0 10px;
            font-size: 13px;
            font-weight: 600;
            color: #fff;
            letter-spacing: 0.3px;
        }}

        #controls {{ top: 16px; right: 16px; min-width: 270px; }}
        #cityPanel {{ top: 16px; left: 16px; min-width: 220px; }}

        .city-btn {{
            display: block; width: 100%; padding: 7px 12px;
            margin-bottom: 4px; border: 1px solid rgba(255,255,255,0.1);
            border-radius: 6px; background: transparent; color: #ccc;
            cursor: pointer; font-size: 12px; text-align: left;
            transition: all 0.15s;
        }}
        .city-btn:hover {{ background: rgba(74,158,255,0.15); color: #fff; }}
        .city-btn.active {{
            background: rgba(74,158,255,0.25); color: #fff;
            border-color: rgba(74,158,255,0.5);
        }}
        .city-btn .acc {{ float: right; color: #6eaaff; font-size: 11px; font-weight: 500; }}

        .layer-row {{
            display: flex; align-items: center; gap: 8px;
            padding: 4px 0; cursor: pointer; font-size: 12px;
        }}
        .layer-row:hover {{ color: #fff; }}
        .layer-row input {{ accent-color: #4a9eff; width: 14px; height: 14px; cursor: pointer; margin: 0; }}
        .layer-row label {{ cursor: pointer; }}

        .slider-group {{
            margin-top: 12px; padding-top: 10px;
            border-top: 1px solid rgba(255,255,255,0.08);
        }}
        .slider-group label {{ display: block; font-size: 11px; color: #888; margin-bottom: 4px; }}
        .slider-group .val {{ color: #6eaaff; float: right; font-weight: 500; }}
        .slider-group input[type="range"] {{ width: 100%; accent-color: #4a9eff; }}

        #legend {{ bottom: 16px; right: 16px; }}
        .legend-item {{ display: flex; align-items: center; gap: 8px; font-size: 12px; line-height: 1.7; }}
        .swatch {{
            width: 18px; height: 12px; border-radius: 2px; flex-shrink: 0;
            border: 1px solid rgba(255,255,255,0.12);
        }}

        #info {{ bottom: 16px; left: 16px; font-size: 11px; color: #999; max-width: 240px; }}
        #info strong {{ color: #fff; }}
        #info .model-tag {{
            display: inline-block; padding: 2px 8px; border-radius: 4px;
            background: rgba(74,158,255,0.2); color: #6eaaff;
            font-weight: 600; font-size: 12px; margin-bottom: 6px;
        }}
    </style>
</head>
<body>
<div id="map"></div>

<div id="cityPanel" class="panel">
    <h4>Test Cities</h4>
    <div id="cityList"></div>
</div>

<div id="controls" class="panel">
    <h4>Prediction Layers</h4>
    <div id="layerList"></div>
    <div class="slider-group">
        <label>Overlay Opacity <span class="val" id="opVal">60%</span></label>
        <input type="range" id="opSlider" min="0" max="100" value="60">
    </div>
</div>

<div id="legend" class="panel">
    <h4>Land Cover Classes</h4>
    <div class="legend-item"><div class="swatch" style="background:rgb(20,200,20)"></div>Tree Cover</div>
    <div class="legend-item"><div class="swatch" style="background:rgb(210,180,50)"></div>Shrubland</div>
    <div class="legend-item"><div class="swatch" style="background:rgb(140,255,100)"></div>Grassland</div>
    <div class="legend-item"><div class="swatch" style="background:rgb(255,240,50)"></div>Cropland</div>
    <div class="legend-item"><div class="swatch" style="background:rgb(255,40,40)"></div>Built-up</div>
    <div class="legend-item"><div class="swatch" style="background:rgb(200,200,200)"></div>Bare/Sparse</div>
    <div class="legend-item"><div class="swatch" style="background:rgb(40,100,255)"></div>Water</div>
</div>

<div id="info" class="panel">
    <div class="model-tag">SSNet V8.0</div><br>
    <strong>Prior-guided + center-expert</strong><br>
    Test accuracy: <strong>86.62%</strong><br>
    Val balanced: <strong>74.96%</strong><br>
    1.66M params · 67 epochs
</div>

<script>
const CITIES = [
    {all_cities_js}
];

const ESRI_STYLE = {{
    version: 8,
    sources: {{
        'esri': {{
            type: 'raster',
            tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}'],
            tileSize: 256,
            attribution: '&copy; Esri',
        }},
    }},
    layers: [{{ id: 'esri-sat', type: 'raster', source: 'esri', minzoom: 0, maxzoom: 19 }}],
}};

let currentCity = 0;
let activeLayerKey = 'v8';
let opacity = 0.6;
let deckgl = null;

function updateDeck() {{
    const city = CITIES[currentCity];
    const layerInfo = city.layers.find(l => l.key === activeLayerKey);
    if (!layerInfo) {{
        deckgl.setProps({{ layers: [] }});
        return;
    }}
    deckgl.setProps({{
        layers: [new deck.BitmapLayer({{
            id: 'overlay',
            image: layerInfo.uri,
            bounds: city.corners,
            opacity: opacity,
        }})]
    }});
}}

function selectCity(idx) {{
    currentCity = idx;
    const city = CITIES[idx];

    deckgl.setProps({{
        initialViewState: {{
            longitude: city.center[0],
            latitude: city.center[1],
            zoom: 12,
            pitch: 0, bearing: 0,
            transitionDuration: 800,
        }}
    }});

    document.querySelectorAll('.city-btn').forEach((b, i) => {{
        b.classList.toggle('active', i === idx);
    }});

    const list = document.getElementById('layerList');
    list.innerHTML = '';
    city.layers.forEach(l => {{
        const row = document.createElement('div');
        row.className = 'layer-row';
        const checked = l.key === activeLayerKey ? 'checked' : '';
        row.innerHTML = `<input type="radio" name="layer" id="r_${{l.key}}" value="${{l.key}}" ${{checked}}><label for="r_${{l.key}}">${{l.name}}</label>`;
        list.appendChild(row);
    }});
    const noneRow = document.createElement('div');
    noneRow.className = 'layer-row';
    const noneChecked = activeLayerKey === null ? 'checked' : '';
    noneRow.innerHTML = `<input type="radio" name="layer" id="r_none" value="none" ${{noneChecked}}><label for="r_none">Satellite only</label>`;
    list.appendChild(noneRow);

    updateDeck();
}}

window.addEventListener('DOMContentLoaded', () => {{
    deckgl = new deck.DeckGL({{
        container: 'map',
        mapStyle: ESRI_STYLE,
        mapLib: maplibregl,
        initialViewState: {{
            longitude: CITIES[0].center[0],
            latitude: CITIES[0].center[1],
            zoom: 12, pitch: 0, bearing: 0,
        }},
        controller: true,
        layers: [],
    }});

    const cityList = document.getElementById('cityList');
    CITIES.forEach((c, i) => {{
        const btn = document.createElement('button');
        btn.className = 'city-btn' + (i === 0 ? ' active' : '');
        btn.innerHTML = c.name + (c.accuracy ? `<span class="acc">${{c.accuracy}}</span>` : '');
        btn.onclick = () => selectCity(i);
        cityList.appendChild(btn);
    }});

    document.getElementById('controls').addEventListener('change', e => {{
        if (e.target.name === 'layer') {{
            activeLayerKey = e.target.value === 'none' ? null : e.target.value;
            updateDeck();
        }}
    }});

    document.getElementById('opSlider').addEventListener('input', e => {{
        opacity = parseInt(e.target.value) / 100;
        document.getElementById('opVal').textContent = e.target.value + '%';
        updateDeck();
    }});

    selectCity(0);
}});
</script>
</body>
</html>"""


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[{ts()}] SSNet v8 prediction map builder")
    print(f"  Device: {device}")

    test_cities = [c for c in get_test_cities() if city_has_raw_tifs(c)]
    print(f"  Cities: {[c.name for c in test_cities]}")

    # Per-city test accuracy from metrics
    import json
    metrics_path = os.path.join(CKPT_DIR, "ssnet_v8_metrics.json")
    per_city_acc = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            m = json.load(f)
        for cname, cdata in m.get("test_per_city", {}).items():
            per_city_acc[cname] = cdata.get("accuracy", 0)

    # Load v8 model
    print(f"\n[{ts()}] Loading SSNet v8...")
    v8_model, v8_ps, v8_is = load_ssnet_v8(device)
    print(f"  v8: {v8_model.n_params():,} params")

    cities_data = []

    for city in test_cities:
        print(f"\n[{ts()}] ====== {city.name} ======")

        corners, clon, clat, H, W = get_raster_corners_wgs84(city)
        if corners is None:
            print(f"  SKIP: no raster")
            continue
        print(f"  Raster: {H}x{W}, center: ({clat:.3f}, {clon:.3f})")

        # Load raw cube
        print(f"  [{ts()}] Loading raw cube...")
        cube, H, W = load_raw_feature_cube(city)
        np.nan_to_num(cube, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

        # SSNet v8
        print(f"  [{ts()}] SSNet v8 predicting {H*W:,} pixels...")
        v8_pred = predict_v8_full(v8_model, v8_ps, v8_is, cube, H, W, device)
        v8_uri = class_map_to_png_b64(v8_pred)
        print(f"    v8 done: {len(v8_uri)//1024:,} KB")
        del v8_pred
        del cube
        gc.collect()

        # ESA labels
        print(f"  [{ts()}] Loading ESA labels...")
        labels = load_pixel_labels(city, year=2021)
        if labels is not None:
            labels_uri = class_map_to_png_b64(labels)
            print(f"    Labels done: {len(labels_uri)//1024:,} KB")
        else:
            labels_uri = class_map_to_png_b64(np.full((H, W), 255, np.uint8))
            print(f"    No labels available")
        del labels
        gc.collect()

        city_acc = per_city_acc.get(city.name, 0)
        display_name = city.name.replace("_", " ").title()
        acc_str = f"{city_acc*100:.1f}%" if city_acc else ""

        cities_data.append({
            "name": display_name,
            "corners": corners,
            "center_lon": clon,
            "center_lat": clat,
            "accuracy": acc_str,
            "layers": [
                ("v8", f"SSNet v8 ({acc_str})", v8_uri),
                ("labels", "ESA WorldCover Labels", labels_uri),
            ],
        })
        del v8_uri, labels_uri
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Generate HTML
    print(f"\n[{ts()}] Generating HTML ({len(cities_data)} cities)...")
    html = build_html(cities_data)
    out_path = os.path.join(OUT_DIR, "v8_predictions.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"[{ts()}] Done! {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
