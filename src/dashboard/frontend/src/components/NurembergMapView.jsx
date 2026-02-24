import { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { Map } from 'react-map-gl/maplibre';
import DeckGL from '@deck.gl/react';
import { BitmapLayer, GeoJsonLayer } from '@deck.gl/layers';
import 'maplibre-gl/dist/maplibre-gl.css';

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';

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
    bare_sparse: [212, 163, 115],
    water: [0, 150, 199],
};
const CLASS_ORDER = ['tree_cover', 'grassland', 'cropland', 'built_up', 'bare_sparse', 'water'];

export default function NurembergMapView({
    meta,
    boundary,
    selectedYear,
    selectedClass,
    resolution,
    classColors,
    loading,
    dataMode = 'labels',
}) {
    const [labelData, setLabelData] = useState(null);
    const [canvasImage, setCanvasImage] = useState(null);
    const deckRef = useRef(null);

    // Fetch binary label/prediction data when year, resolution, or dataMode changes
    useEffect(() => {
        if (!meta) return;
        const endpoint = dataMode === 'predictions' ? 'predictions' : 'labels';
        const url = `${API}/api/nuremberg/${endpoint}/${selectedYear}/${resolution}`;
        fetch(url)
            .then(res => {
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                return res.arrayBuffer();
            })
            .then(buf => {
                setLabelData(new Uint8Array(buf));
            })
            .catch(err => console.error(`Failed to load ${endpoint}:`, err));
    }, [selectedYear, resolution, meta, dataMode]);

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

        for (let i = 0; i < labelData.length && i < width * height; i++) {
            const cls = labelData[i];
            const px = i * 4;

            if (cls === 255 || cls >= CLASS_ORDER.length) {
                // Outside boundary — transparent
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
                // Highlighted class — full opacity
                imageData.data[px] = r;
                imageData.data[px + 1] = g;
                imageData.data[px + 2] = b;
                imageData.data[px + 3] = 255;
            } else {
                // Dimmed
                imageData.data[px] = 40;
                imageData.data[px + 1] = 40;
                imageData.data[px + 2] = 50;
                imageData.data[px + 3] = 120;
            }
        }

        ctx.putImageData(imageData, 0, 0);
        setCanvasImage(canvas);
    }, [labelData, meta, resolution, selectedClass, classColors]);

    // DeckGL layers
    const layers = useMemo(() => {
        const result = [];

        if (canvasImage && meta) {
            const [west, south, east, north] = meta.wgs84_bounds;
            result.push(new BitmapLayer({
                id: 'nuremberg-labels',
                image: canvasImage,
                bounds: [west, south, east, north],
                textureParameters: {
                    minFilter: 'nearest',
                    magFilter: 'nearest',
                },
            }));
        }

        if (boundary) {
            result.push(new GeoJsonLayer({
                id: 'nuremberg-boundary',
                data: boundary,
                pickable: false,
                stroked: true,
                filled: false,
                getLineColor: [255, 255, 255, 100],
                getLineWidth: 1.5,
                lineWidthUnits: 'pixels',
            }));
        }

        return result;
    }, [canvasImage, meta, boundary]);

    // Tooltip
    const getTooltip = useCallback(({ bitmap, coordinate }) => {
        if (!bitmap || !coordinate || !meta || !labelData) return null;
        const resKey = `res${resolution}`;
        const dims = meta.resolutions[resKey];
        if (!dims) return null;

        const [west, south, east, north] = meta.wgs84_bounds;
        const [lng, lat] = coordinate;

        // Calculate pixel position
        const fracX = (lng - west) / (east - west);
        const fracY = (north - lat) / (north - south);
        const px = Math.floor(fracX * dims.width);
        const py = Math.floor(fracY * dims.height);

        if (px < 0 || px >= dims.width || py < 0 || py >= dims.height) return null;
        const idx = py * dims.width + px;
        const cls = labelData[idx];

        if (cls === 255 || cls >= CLASS_ORDER.length) return null;
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
    }, [meta, labelData, resolution]);

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
                    mapStyle="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
                    attributionControl={false}
                />
            </DeckGL>
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

            {/* Resolution legend */}
            <div style={{
                position: 'absolute', bottom: 24, right: 16,
                background: 'rgba(15, 23, 42, 0.85)', borderRadius: 8,
                padding: '8px 14px', color: '#e2e8f0',
                fontSize: 12, backdropFilter: 'blur(8px)',
                border: '1px solid rgba(255,255,255,0.1)',
            }}>
                <strong>{resolution * 10}m</strong> resolution
                &nbsp;&middot;&nbsp;
                {meta?.resolutions?.[`res${resolution}`]
                    ? `${meta.resolutions[`res${resolution}`].width} × ${meta.resolutions[`res${resolution}`].height}`
                    : ''
                } cells
                &nbsp;&middot;&nbsp;
                {dataMode === 'predictions' ? '🤖 Predicted' : '🏷️ Labels'}
            </div>
        </div>
    );
}
