#!/usr/bin/env python3
"""
Build a multi-city deck.gl comparison map for all test cities.

For each of the 6 test cities, renders:
  - SSNet V3 predictions
  - SSNet V5 predictions
  - CatBoost pixel_v5 predictions
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

sys.path.insert(0, os.path.join(PROJECT_ROOT, "reproduce", "mlp"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "reproduce", "pixel"))
from importlib import import_module

CKPT_DIR = os.path.join(PROJECT_ROOT, "reproduce", "models", "checkpoints")
CB_DIR = os.path.join(PROJECT_ROOT, "data", "cities", "models_pixel_v5")
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
    # deck.gl BitmapLayer: [SW, NW, NE, SE]
    # Our order: NW=0, NE=1, SE=2, SW=3
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

def load_ssnet_v3(device):
    from reproduce.models.architectures.spectral_spatial import SpectralSpatialNetV2
    model = SpectralSpatialNetV2(
        spatial_dims=(48, 96, 192), expand_ratio=4,
        temporal_dim=192, n_attn_layers=3,
    ).to(device)
    state = torch.load(os.path.join(CKPT_DIR, "ssnet_v3_ep3_backup.pt"),
                       map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    with open(os.path.join(CKPT_DIR, "ssnet_scaler_v3_backup.pkl"), "rb") as f:
        sc = pickle.load(f)
    return model, sc["patches"], sc["indices"]


def load_ssnet_v5(device):
    from reproduce.models.architectures.spectral_spatial_v5 import SpectralSpatialNetV5
    model = SpectralSpatialNetV5(
        n_bands=12, n_timesteps=6, n_indices=145,
        spatial_dims=(32, 64, 128), expand_ratio=4,
        temporal_dim=128, n_attn_layers=2, n_heads=4,
        n_classes=7, dropout=0.15,
        spatial_branch_drop=0.10, index_branch_drop=0.25,
    ).to(device)
    state = torch.load(os.path.join(CKPT_DIR, "ssnet_v5.pt"),
                       map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    with open(os.path.join(CKPT_DIR, "ssnet_v5_fixed_scaler.pkl"), "rb") as f:
        sc = pickle.load(f)
    return model, sc["patches"], sc["indices"]


def load_catboost():
    from catboost import CatBoostClassifier
    model = CatBoostClassifier()
    model.load_model(os.path.join(CB_DIR, "catboost_pixel_v5_deep_unweighted.cbm"))
    return model


# ── SSNet prediction (all pixels, row-by-row) ───────────────────────────────

def predict_ssnet_full(model, ps, idx_s, cube, H, W, device):
    """Predict all pixels using SSNet, row by row with 3×3 patches."""
    padded = np.pad(cube, ((1, 1), (1, 1), (0, 0)),
                    mode='constant', constant_values=0.0)
    pred = np.full((H, W), 255, dtype=np.uint8)
    BATCH = 4096

    for r in range(H):
        row_patches = np.empty((W, 9 * 72), dtype=np.float32)
        for c in range(W):
            patch = padded[r:r+3, c:c+3, :]
            row_patches[c] = patch.reshape(-1)

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

        preds_valid = np.empty(n_valid, dtype=np.uint8)
        with torch.no_grad():
            for s in range(0, n_valid, BATCH):
                e = min(s + BATCH, n_valid)
                xp = torch.from_numpy(patches_s[s:e]).to(device)
                xi = torch.from_numpy(indices_s[s:e]).to(device)
                out = model(xp, xi)
                if isinstance(out, dict):
                    logits = out["logits"]
                else:
                    logits = out
                preds_valid[s:e] = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)

        pred[r, valid] = preds_valid

    return pred


# ── CatBoost prediction (all pixels) ────────────────────────────────────────

def predict_catboost_full(cb_model, city, H, W):
    """Predict all pixels using CatBoost with full feature engineering."""
    step2 = import_module("02_train_catboost")
    raw_d = step2._raw_dir(city)

    YEARS = [2020, 2021]
    SEASONS = ["spring", "summer", "autumn"]
    INDEX_NAMES = step2.INDEX_NAMES

    all_bands = []
    indices_by_tag, sar_by_tag = {}, {}

    for year in YEARS:
        for season in SEASONS:
            tag = f"{year}_{season}"
            s2_path = os.path.join(raw_d, f"sentinel2_{city.name}_{year}_{season}.tif")
            s1_path = os.path.join(raw_d, f"sentinel1_{city.name}_{year}_{season}.tif")

            s2 = step2._load_tif(s2_path, step2.SENTINEL_NODATA) if os.path.exists(s2_path) else None
            if s2 is not None and s2.shape[0] >= 10:
                for bi in range(len(step2.SENTINEL_BANDS)):
                    all_bands.append(s2[bi])
                idx = step2._compute_indices(s2[:10])
                indices_by_tag[tag] = idx
                for n in INDEX_NAMES:
                    all_bands.append(idx[n])
            else:
                for _ in step2.SENTINEL_BANDS:
                    all_bands.append(np.full((H, W), np.nan, np.float32))
                for _ in INDEX_NAMES:
                    all_bands.append(np.full((H, W), np.nan, np.float32))

            s1 = step2._load_tif(s1_path, step2.SAR_NODATA) if os.path.exists(s1_path) else None
            if s1 is not None:
                all_bands.append(s1[0])
                all_bands.append(s1[1])
                vvvh = np.where(np.abs(s1[1]) > 1e-10, s1[0] / s1[1], np.nan).astype(np.float32)
                all_bands.append(vvvh)
                sar_by_tag[tag] = {"vv": s1[0], "vh": s1[1]}
            else:
                for _ in range(3):
                    all_bands.append(np.full((H, W), np.nan, np.float32))

    # Temporal diffs
    for year in YEARS:
        for sf, st in [("spring", "summer"), ("summer", "autumn")]:
            tf, tt = f"{year}_{sf}", f"{year}_{st}"
            if tf in indices_by_tag and tt in indices_by_tag:
                for n in INDEX_NAMES:
                    all_bands.append((indices_by_tag[tt][n] - indices_by_tag[tf][n]).astype(np.float32))

    for season in SEASONS:
        t0, t1 = f"{YEARS[0]}_{season}", f"{YEARS[1]}_{season}"
        if t0 in indices_by_tag and t1 in indices_by_tag:
            for n in INDEX_NAMES:
                all_bands.append((indices_by_tag[t1][n] - indices_by_tag[t0][n]).astype(np.float32))

    for year in YEARS:
        ts_s, ts_a = f"{year}_spring", f"{year}_autumn"
        if ts_s in indices_by_tag and ts_a in indices_by_tag:
            for n in ["NDVI", "NDWI", "EVI2", "BSI"]:
                all_bands.append((indices_by_tag[ts_a][n] - indices_by_tag[ts_s][n]).astype(np.float32))

    for year in YEARS:
        for sf, st in [("spring", "summer"), ("summer", "autumn")]:
            tf, tt = f"{year}_{sf}", f"{year}_{st}"
            if tf in sar_by_tag and tt in sar_by_tag:
                for b in ["vv", "vh"]:
                    all_bands.append((sar_by_tag[tt][b] - sar_by_tag[tf][b]).astype(np.float32))

    for season in SEASONS:
        t0, t1 = f"{YEARS[0]}_{season}", f"{YEARS[1]}_{season}"
        if t0 in sar_by_tag and t1 in sar_by_tag:
            for b in ["vv", "vh"]:
                all_bands.append((sar_by_tag[t1][b] - sar_by_tag[t0][b]).astype(np.float32))

    del indices_by_tag, sar_by_tag
    gc.collect()

    n_feat = len(all_bands)
    print(f"    CatBoost features: {n_feat}")

    # Predict in chunks of rows
    pred = np.full((H, W), 255, dtype=np.uint8)
    CHUNK_ROWS = 200

    for r0 in range(0, H, CHUNK_ROWS):
        r1 = min(r0 + CHUNK_ROWS, H)
        n_pix = (r1 - r0) * W
        chunk = np.empty((n_pix, n_feat), dtype=np.float32)
        for fi, band in enumerate(all_bands):
            chunk[:, fi] = band[r0:r1].reshape(-1)

        np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        valid = np.abs(chunk).sum(axis=1) > 0

        if valid.sum() > 0:
            cb_pred = cb_model.predict(chunk[valid]).flatten().astype(np.uint8)
            flat_pred = np.full(n_pix, 255, dtype=np.uint8)
            flat_pred[valid] = cb_pred
            pred[r0:r1] = flat_pred.reshape(r1 - r0, W)

    del all_bands
    gc.collect()
    return pred


# ── HTML generation ──────────────────────────────────────────────────────────

def build_html(cities_data):
    """Build HTML with city selector + layer selector."""

    # Build JS data
    cities_js_parts = []
    for cd in cities_data:
        layers_js = []
        for key, name, uri in cd["layers"]:
            layers_js.append(f'{{ key: "{key}", name: "{name}", uri: "{uri}" }}')

        cities_js_parts.append(f"""{{
            name: "{cd['name']}",
            center: [{cd['center_lon']}, {cd['center_lat']}],
            corners: {cd['corners']},
            layers: [{', '.join(layers_js)}],
        }}""")

    all_cities_js = ",\n        ".join(cities_js_parts)

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Test Cities - Model Comparison</title>
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
        #cityPanel {{ top: 16px; left: 16px; min-width: 200px; }}

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

        #stats {{ bottom: 16px; left: 16px; font-size: 11px; color: #999; }}
        #stats strong {{ color: #fff; }}
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
    <h4>Land Cover</h4>
    <div class="legend-item"><div class="swatch" style="background:rgb(20,200,20)"></div>Tree Cover</div>
    <div class="legend-item"><div class="swatch" style="background:rgb(210,180,50)"></div>Shrubland</div>
    <div class="legend-item"><div class="swatch" style="background:rgb(140,255,100)"></div>Grassland</div>
    <div class="legend-item"><div class="swatch" style="background:rgb(255,240,50)"></div>Cropland</div>
    <div class="legend-item"><div class="swatch" style="background:rgb(255,40,40)"></div>Built-up</div>
    <div class="legend-item"><div class="swatch" style="background:rgb(200,200,200)"></div>Bare/Sparse</div>
    <div class="legend-item"><div class="swatch" style="background:rgb(40,100,255)"></div>Water</div>
</div>

<div id="stats" class="panel"></div>

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
let activeLayerKey = null;
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

    // Fly to city
    deckgl.setProps({{
        initialViewState: {{
            longitude: city.center[0],
            latitude: city.center[1],
            zoom: 12,
            pitch: 0, bearing: 0,
            transitionDuration: 800,
        }}
    }});

    // Update city buttons
    document.querySelectorAll('.city-btn').forEach((b, i) => {{
        b.classList.toggle('active', i === idx);
    }});

    // Update layer list
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

    // City buttons
    const cityList = document.getElementById('cityList');
    CITIES.forEach((c, i) => {{
        const btn = document.createElement('button');
        btn.className = 'city-btn' + (i === 0 ? ' active' : '');
        btn.textContent = c.name;
        btn.onclick = () => selectCity(i);
        cityList.appendChild(btn);
    }});

    // Layer toggle
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

    print(f"[{ts()}] Multi-city comparison map builder")
    print(f"  Device: {device}")

    test_cities = [c for c in get_test_cities() if city_has_raw_tifs(c)]
    print(f"  Cities: {[c.name for c in test_cities]}")

    # Load models once
    print(f"\n[{ts()}] Loading models...")
    v3_model, v3_ps, v3_is = load_ssnet_v3(device)
    v5_model, v5_ps, v5_is = load_ssnet_v5(device)
    cb_model = load_catboost()
    print(f"  V3: {v3_model.n_params():,} params")
    print(f"  V5: {v5_model.n_params():,} params")
    print(f"  CB: {cb_model.tree_count_} trees")

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

        # SSNet V3
        print(f"  [{ts()}] SSNet V3 predicting {H*W:,} pixels...")
        v3_pred = predict_ssnet_full(v3_model, v3_ps, v3_is, cube, H, W, device)
        v3_uri = class_map_to_png_b64(v3_pred)
        print(f"    V3 done: {len(v3_uri)//1024:,} KB")
        del v3_pred; gc.collect()

        # SSNet V5
        print(f"  [{ts()}] SSNet V5 predicting...")
        v5_pred = predict_ssnet_full(v5_model, v5_ps, v5_is, cube, H, W, device)
        v5_uri = class_map_to_png_b64(v5_pred)
        print(f"    V5 done: {len(v5_uri)//1024:,} KB")
        del v5_pred; gc.collect()

        del cube; gc.collect()

        # CatBoost
        print(f"  [{ts()}] CatBoost predicting...")
        cb_pred = predict_catboost_full(cb_model, city, H, W)
        cb_uri = class_map_to_png_b64(cb_pred)
        print(f"    CB done: {len(cb_uri)//1024:,} KB")
        del cb_pred; gc.collect()

        # ESA labels
        print(f"  [{ts()}] Loading ESA labels...")
        labels = load_pixel_labels(city, year=2021)
        if labels is not None:
            labels_uri = class_map_to_png_b64(labels)
            print(f"    Labels done: {len(labels_uri)//1024:,} KB")
        else:
            labels_uri = class_map_to_png_b64(np.full((H, W), 255, np.uint8))
            print(f"    No labels available")
        del labels; gc.collect()

        display_name = city.name.replace("_", " ").title()
        cities_data.append({
            "name": display_name,
            "corners": corners,
            "center_lon": clon,
            "center_lat": clat,
            "layers": [
                ("v3", "SSNet V3 (86.5% acc)", v3_uri),
                ("v5", "SSNet V5 (86.5% acc)", v5_uri),
                ("cb", "CatBoost pixel_v5 (85.2%)", cb_uri),
                ("labels", "ESA WorldCover Labels", labels_uri),
            ],
        })
        del v3_uri, v5_uri, cb_uri, labels_uri
        gc.collect()

    # Generate HTML
    print(f"\n[{ts()}] Generating HTML ({len(cities_data)} cities)...")
    html = build_html(cities_data)
    out_path = os.path.join(OUT_DIR, "all_cities_comparison.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"[{ts()}] Done! {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
