import { useState, useEffect, useMemo, useCallback, useRef } from 'react';

const SATELLITE_STYLE = {
    version: 8,
    sources: {
        'esri-satellite': {
            type: 'raster',
            tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'],
            tileSize: 256,
            attribution: '© Esri',
        },
    },
    layers: [{ id: 'esri-satellite-layer', type: 'raster', source: 'esri-satellite', minzoom: 0, maxzoom: 19 }],
};
const DARK_STYLE = 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json';
import { Map } from 'react-map-gl/maplibre';
import DeckGL from '@deck.gl/react';
import { BitmapLayer, GeoJsonLayer } from '@deck.gl/layers';
import 'maplibre-gl/dist/maplibre-gl.css';

const API = import.meta.env.VITE_API_URL || '';

const INITIAL_VIEW = {
    longitude: 11.076,
    latitude: 49.449,
    zoom: 12,
    pitch: 0,
    bearing: 0,
};

// Nuremberg classes (no shrubland)
const CLASS_COLORS_RGB = {
    tree_cover: [45, 106, 79],
    grassland: [149, 213, 178],
    cropland: [244, 162, 97],
    built_up: [231, 111, 81],
    bare_sparse: [142, 68, 173],
    water: [0, 150, 199],
};
const CLASS_ORDER = ['tree_cover', 'grassland', 'cropland', 'built_up', 'bare_sparse', 'water'];
const CLASS_LABELS = {
    tree_cover: 'Tree Cover',
    grassland: 'Grassland',
    cropland: 'Cropland',
    built_up: 'Built-up',
    bare_sparse: 'Bare/Sparse',
    water: 'Water',
};

export default function NurembergMapView({
    meta,
    boundary,
    selectedYear,
    secondaryYear,
    selectedClass,
    resolution,
    classColors,
    loading,
    dataMode = 'labels',
    experimentalView = 'map',
    selectedDistricts = [],
    onDistrictClick,
    hoveredDistrict,
    onDistrictHover,
    districtStats,
    experimentalModel = 'rf',
    predModel = 'catboost',
}) {
    const [labelData, setLabelData] = useState(null);
    const [canvasImage, setCanvasImage] = useState(null);
    const [mapStyle, setMapStyle] = useState('dark');
    const deckRef = useRef(null);

    // Fetch binary label/prediction data when year, resolution, or dataMode changes
    useEffect(() => {
        if (!meta) return;
        let url;
        if (dataMode === 'experimental') {
            const sub = experimentalView === 'map' ? '' : `/${experimentalView}`;
            const query = experimentalView === 'heatmap' ? `?model=${experimentalModel}` : '';
            url = `${API}/api/nuremberg/experimental${sub}/${resolution}${query}`;
        } else if (dataMode === 'predictions' && secondaryYear !== null) {
            const modelQ = predModel !== 'catboost' ? `?model=${predModel}` : '';
            url = `${API}/api/nuremberg/predictions/diff/${selectedYear}/${secondaryYear}/${resolution}${modelQ}`;
        } else {
            const endpoint = dataMode === 'predictions' ? 'predictions' : 'labels';
            const modelQ = dataMode === 'predictions' && predModel !== 'catboost' ? `?model=${predModel}` : '';
            url = `${API}/api/nuremberg/${endpoint}/${selectedYear}/${resolution}${modelQ}`;
        }
        fetch(url)
            .then(res => {
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                return res.arrayBuffer();
            })
            .then(buf => {
                setLabelData(new Uint8Array(buf));
            })
            .catch(err => console.error(`Failed to load nuremberg data:`, err));
    }, [selectedYear, secondaryYear, resolution, meta, dataMode, experimentalView, experimentalModel, predModel]);

    // Generate canvas image from label data
    useEffect(() => {
        if (!labelData || !meta) return;
        const resKey = `res${resolution}`;
        const dims = meta.resolutions[resKey];
        if (!dims) return;

        const { width, height } = dims;
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        const imageData = ctx.createImageData(width, height);

        const colors = CLASS_ORDER.map(c => classColors?.[c] || CLASS_COLORS_RGB[c]);
        const showAll = selectedClass === 'all';
        const selectedIdx = CLASS_ORDER.indexOf(selectedClass);

        const isHeatmap = dataMode === 'experimental' && experimentalView === 'heatmap';
        const isChanges = dataMode === 'experimental' && experimentalView === 'changes';
        const isDiffView = dataMode === 'predictions' && secondaryYear !== null;

        for (let i = 0; i < labelData.length && i < width * height; i++) {
            const val = labelData[i];
            const px = i * 4;

            if (val === 255) {
                imageData.data[px + 3] = 0;
                continue;
            }

            if (isHeatmap) {
                // Power-scale to emphasize high probability (>= 0.8)
                const t = Math.pow(val / 255, 3);
                imageData.data[px] = 255;
                imageData.data[px + 1] = Math.round(255 * (1 - t));
                imageData.data[px + 2] = 0;
                imageData.data[px + 3] = Math.round(20 + t * 235);
                continue;
            }

            if (isDiffView && val === 254) {
                imageData.data[px] = 30;
                imageData.data[px + 1] = 41;
                imageData.data[px + 2] = 59;
                imageData.data[px + 3] = 40;
                continue;
            }

            if (isChanges || (isDiffView && val !== 254)) {
                if (val >= CLASS_ORDER.length) {
                    imageData.data[px + 3] = 0;
                    continue;
                }
                const [r, g, b] = colors[val];
                if (showAll || val === selectedIdx) {
                    imageData.data[px] = r;
                    imageData.data[px + 1] = g;
                    imageData.data[px + 2] = b;
                    imageData.data[px + 3] = 255;
                } else {
                    imageData.data[px] = 40;
                    imageData.data[px + 1] = 40;
                    imageData.data[px + 2] = 50;
                    imageData.data[px + 3] = 120;
                }
                continue;
            }

            const cls = val;
            if (cls >= CLASS_ORDER.length) {
                imageData.data[px + 3] = 0;
                continue;
            }

            const [r, g, b] = colors[cls];

            if (showAll) {
                imageData.data[px] = r;
                imageData.data[px + 1] = g;
                imageData.data[px + 2] = b;
                imageData.data[px + 3] = 220;
            } else if (cls === selectedIdx) {
                imageData.data[px] = r;
                imageData.data[px + 1] = g;
                imageData.data[px + 2] = b;
                imageData.data[px + 3] = 255;
            } else {
                imageData.data[px] = 40;
                imageData.data[px + 1] = 40;
                imageData.data[px + 2] = 50;
                imageData.data[px + 3] = 120;
            }
        }

        ctx.putImageData(imageData, 0, 0);
        setCanvasImage(canvas);
    }, [labelData, meta, resolution, selectedClass, classColors, dataMode, experimentalView]);

    // Build district tooltip HTML from stats
    const buildDistrictTooltip = useCallback((districtId) => {
        if (!districtStats || !districtId) return null;
        const stats = districtStats[districtId];
        if (!stats) return null;

        // Determine which dataset to show based on current data mode
        let dataKey = 'labels_2021';
        if (dataMode === 'labels') {
            dataKey = `labels_${selectedYear}`;
        } else if (dataMode === 'predictions') {
            dataKey = 'predictions_2021';
        } else if (dataMode === 'experimental') {
            dataKey = 'experimental_2021';
        }

        const classData = stats[dataKey] || stats['labels_2021'];
        if (!classData) return null;

        const total = stats.total_pixels || 1;
        const rows = CLASS_ORDER.map(c => {
            const count = classData[c] || 0;
            const pct = ((count / total) * 100).toFixed(1);
            const [r, g, b] = CLASS_COLORS_RGB[c];
            return `<div style="display:flex;justify-content:space-between;align-items:center;padding:2px 0;gap:12px">
                <span style="display:flex;align-items:center;gap:5px">
                    <span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:rgb(${r},${g},${b})"></span>
                    ${CLASS_LABELS[c]}
                </span>
                <span style="font-family:monospace;font-size:10px">${pct}%</span>
            </div>`;
        }).join('');

        return `<div style="font-weight:600;font-size:13px;margin-bottom:6px;border-bottom:1px solid rgba(255,255,255,0.15);padding-bottom:4px">
            ${stats.id} ${stats.name}
        </div>
        <div style="font-size:11px">${rows}</div>
        <div style="font-size:10px;opacity:0.5;margin-top:4px">${total.toLocaleString()} pixels · Click to select</div>`;
    }, [districtStats, dataMode, selectedYear]);

    // DeckGL layers
    const layers = useMemo(() => {
        const result = [];

        if (canvasImage && meta) {
            const bounds = meta.wgs84_corners
                ? [
                    meta.wgs84_corners[3],
                    meta.wgs84_corners[0],
                    meta.wgs84_corners[1],
                    meta.wgs84_corners[2]
                ]
                : meta.wgs84_bounds;

            result.push(new BitmapLayer({
                id: 'nuremberg-labels',
                image: canvasImage,
                bounds: bounds,
                textureParameters: {
                    minFilter: 'nearest',
                    magFilter: 'nearest',
                },
            }));
        }

        if (boundary) {
            // Interactive boundary layer with hover and click
            result.push(new GeoJsonLayer({
                id: 'nuremberg-boundary',
                data: boundary,
                pickable: true,
                stroked: true,
                filled: true,
                getFillColor: (f) => {
                    const id = f.properties?.KRG_DISS;
                    if (selectedDistricts.includes(id)) {
                        return [59, 130, 246, 60]; // Blue highlight for selected
                    }
                    if (hoveredDistrict === id) {
                        return [255, 255, 255, 30]; // Subtle white on hover
                    }
                    return [0, 0, 0, 0]; // Transparent
                },
                getLineColor: (f) => {
                    const id = f.properties?.KRG_DISS;
                    if (selectedDistricts.includes(id)) {
                        return [59, 130, 246, 220]; // Blue for selected
                    }
                    if (hoveredDistrict === id) {
                        return [255, 255, 255, 200]; // White on hover
                    }
                    return [255, 255, 255, 70]; // Default dim
                },
                getLineWidth: (f) => {
                    const id = f.properties?.KRG_DISS;
                    if (selectedDistricts.includes(id) || hoveredDistrict === id) {
                        return 2.5;
                    }
                    return 1;
                },
                lineWidthUnits: 'pixels',
                onClick: (info) => {
                    if (info.object && onDistrictClick) {
                        onDistrictClick(info.object.properties.KRG_DISS);
                    }
                },
                onHover: (info) => {
                    if (onDistrictHover) {
                        onDistrictHover(info.object?.properties?.KRG_DISS || null);
                    }
                },
                updateTriggers: {
                    getFillColor: [selectedDistricts, hoveredDistrict],
                    getLineColor: [selectedDistricts, hoveredDistrict],
                    getLineWidth: [selectedDistricts, hoveredDistrict],
                },
            }));
        }

        return result;
    }, [canvasImage, meta, boundary, selectedDistricts, hoveredDistrict, onDistrictClick, onDistrictHover]);

    // Tooltip
    const getTooltip = useCallback(({ object, bitmap, coordinate, layer }) => {
        // District tooltip (from GeoJsonLayer)
        if (object?.properties?.KRG_DISS) {
            const districtId = object.properties.KRG_DISS;
            const html = buildDistrictTooltip(districtId);
            if (html) {
                return { html, className: 'deck-tooltip district-tooltip' };
            }
            return {
                html: `<div class="tooltip-title">${districtId} ${object.properties.KRG_BEZ || ''}</div>`,
                className: 'deck-tooltip',
            };
        }

        // Pixel tooltip (from BitmapLayer)
        if (!bitmap || !coordinate || !meta || !labelData) return null;
        const resKey = `res${resolution}`;
        const dims = meta.resolutions[resKey];
        if (!dims) return null;

        const [west, south, east, north] = meta.wgs84_bounds;
        const [lng, lat] = coordinate;

        const fracX = (lng - west) / (east - west);
        const fracY = (north - lat) / (north - south);
        const px = Math.floor(fracX * dims.width);
        const py = Math.floor(fracY * dims.height);

        if (px < 0 || px >= dims.width || py < 0 || py >= dims.height) return null;
        const idx = py * dims.width + px;
        const cls = labelData[idx];

        if (cls === 255 || cls >= CLASS_ORDER.length && cls !== 254) return null;

        if (dataMode === 'predictions' && secondaryYear !== null) {
            if (cls === 254) return {
                html: `<div class="tooltip-title">No Change</div><div style="font-size:11px">${selectedYear} → ${secondaryYear}</div>`,
                className: 'deck-tooltip'
            };
            const label = CLASS_ORDER[cls].replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
            return {
                html: `<div class="tooltip-title">New Land Cover: ${label}</div><div style="font-size:11px">Change detected in ${secondaryYear}</div>`,
                className: 'deck-tooltip'
            };
        }

        if (dataMode === 'experimental' && experimentalView === 'heatmap') {
            const probability = (cls / 254 * 100).toFixed(1);
            return {
                html: `<div class="tooltip-title">Change Likelihood</div>
                       <div style="font-size: 11px;">
                         Probability: <strong>${probability}%</strong>
                         <br/>Resolution: ${resolution * 10}m
                       </div>`,
                className: 'deck-tooltip',
            };
        }

        if (dataMode === 'experimental' && experimentalView === 'changes') {
            if (cls === 254) {
                return {
                    html: `<div class="tooltip-title">No Change</div>
                           <div style="font-size: 11px;">Same as 2020 ground truth</div>`,
                    className: 'deck-tooltip',
                };
            }
            if (cls < CLASS_ORDER.length) {
                const label = CLASS_ORDER[cls].replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
                return {
                    html: `<div class="tooltip-title">Predicted: ${label}</div>
                           <div style="font-size: 11px;">Changed from 2020 class</div>`,
                    className: 'deck-tooltip',
                };
            }
            return null;
        }

        const className = CLASS_ORDER[cls];
        const label = className.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
        const resM = resolution * 10;

        return {
            html: `<div class="tooltip-title">${label}</div>
                   <div style="font-size: 11px;">
                     Resolution: ${resM}m
                     <br/>Lat: ${lat.toFixed(5)}, Lng: ${lng.toFixed(5)}
                   </div>`,
            className: 'deck-tooltip',
        };
    }, [meta, labelData, resolution, buildDistrictTooltip, dataMode, experimentalView, selectedYear, secondaryYear]);

    return (
        <div className="map-container">
            {loading && (
                <div className="loading-overlay">
                    <div className="spinner" />
                </div>
            )}
            <DeckGL
                ref={deckRef}
                initialViewState={INITIAL_VIEW}
                controller={true}
                layers={layers}
                getTooltip={getTooltip}
                style={{ width: '100%', height: '100%' }}
            >
                <Map
                    mapStyle={mapStyle === 'satellite' ? SATELLITE_STYLE : DARK_STYLE}
                />
            </DeckGL>
            <button
                onClick={() => setMapStyle(s => s === 'dark' ? 'satellite' : 'dark')}
                style={{
                    position: 'absolute', top: 14, right: 20, zIndex: 10,
                    background: 'rgba(30,30,40,0.85)', color: '#fff',
                    border: '1px solid rgba(255,255,255,0.2)', borderRadius: 8,
                    padding: '8px 14px', cursor: 'pointer', fontSize: 13,
                    backdropFilter: 'blur(8px)', transition: 'all 0.2s',
                }}
                title={mapStyle === 'dark' ? 'Switch to satellite view' : 'Switch to dark map'}
            >
                {mapStyle === 'dark' ? '🛰️ Satellite' : '🗺️ Dark Map'}
            </button>
            {/* Future year placeholder */}
            {selectedYear >= 2026 && dataMode === 'predictions' && (
                <div style={{
                    position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
                    display: 'flex', flexDirection: 'column',
                    alignItems: 'center', justifyContent: 'center',
                    background: 'rgba(15, 23, 42, 0.6)',
                    backdropFilter: 'blur(2px)',
                    pointerEvents: 'none', zIndex: 10,
                }}>
                    <div style={{
                        background: 'rgba(15, 23, 42, 0.9)',
                        borderRadius: 16, padding: '32px 48px',
                        border: '1px solid rgba(59, 130, 246, 0.25)',
                        textAlign: 'center', maxWidth: 420,
                    }}>
                        <div style={{ fontSize: 48, marginBottom: 12 }}>🔮</div>
                        <div style={{
                            fontSize: 22, fontWeight: 600,
                            color: '#e2e8f0', marginBottom: 8,
                        }}>
                            Predicting Future Years
                        </div>
                        <div style={{
                            fontSize: 14, color: '#94a3b8', lineHeight: 1.5,
                        }}>
                            Satellite data for <strong style={{ color: '#e2e8f0' }}>{selectedYear}</strong> is
                            not yet available. Future predictions will appear here once
                            Sentinel-2 imagery is captured and processed.
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
