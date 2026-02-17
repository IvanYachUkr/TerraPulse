import { useState, useEffect, useCallback, useRef } from 'react';
import DeckGL from '@deck.gl/react';
import { Map } from 'react-map-gl/maplibre';
import { GeoJsonLayer } from '@deck.gl/layers';
import DeployPanel from './DeployPanel.jsx';

const API = 'http://localhost:8000';

const INITIAL_VIEW = {
    longitude: 11.08,
    latitude: 49.45,
    zoom: 10,
    pitch: 0,
    bearing: 0,
};

const CLASSES = ['tree_cover', 'grassland', 'cropland', 'built_up', 'bare_sparse', 'water'];

const CLASS_COLORS = {
    tree_cover: [45, 106, 79],
    grassland: [149, 213, 178],
    cropland: [244, 162, 97],
    built_up: [231, 111, 81],
    bare_sparse: [212, 163, 115],
    water: [0, 150, 199],
};

const CLASS_LABELS = {
    tree_cover: 'Tree Cover',
    grassland: 'Grassland',
    cropland: 'Cropland',
    built_up: 'Built-up',
    bare_sparse: 'Bare/Sparse',
    water: 'Water',
};

export default function DeployView() {
    // Drawing state
    const [drawMode, setDrawMode] = useState(false);
    const [drawStart, setDrawStart] = useState(null);
    const [drawEnd, setDrawEnd] = useState(null);
    const [bbox, setBbox] = useState(null);

    // Job state
    const [jobId, setJobId] = useState(null);
    const [jobStatus, setJobStatus] = useState(null);
    const [selectedYear, setSelectedYear] = useState(null);
    const [selectedClass, setSelectedClass] = useState('all');

    // Data
    const [grid, setGrid] = useState(null);
    const [results, setResults] = useState({});  // { year: data }
    const [labels, setLabels] = useState({});     // { year: data }
    const [viewMode, setViewMode] = useState('predictions'); // predictions | labels | change

    const pollingRef = useRef(null);
    const mapRef = useRef(null);

    // Polling for job status
    useEffect(() => {
        if (!jobId) return;

        const poll = async () => {
            try {
                const res = await fetch(`${API}/api/deploy/status/${jobId}`);
                const data = await res.json();
                setJobStatus(data);

                if (data.status === 'complete') {
                    clearInterval(pollingRef.current);
                    // Fetch grid
                    const gridRes = await fetch(`${API}/api/deploy/grid/${jobId}`);
                    const gridData = await gridRes.json();
                    setGrid(gridData);

                    // Fetch all available year results
                    for (const year of data.result_years) {
                        const resResult = await fetch(`${API}/api/deploy/results/${jobId}/${year}`);
                        const resData = await resResult.json();
                        setResults(prev => ({ ...prev, [year]: resData }));

                        // Try fetching labels (will 404 for future years)
                        try {
                            const labRes = await fetch(`${API}/api/deploy/labels/${jobId}/${year}`);
                            if (labRes.ok) {
                                const labData = await labRes.json();
                                setLabels(prev => ({ ...prev, [year]: labData }));
                            }
                        } catch (e) { /* no labels for this year */ }
                    }

                    // Also fetch labels for label-only years (2020, 2021) that might not be in result_years
                    for (const year of [2020, 2021]) {
                        if (!data.result_years.includes(year)) {
                            try {
                                const labRes = await fetch(`${API}/api/deploy/labels/${jobId}/${year}`);
                                if (labRes.ok) {
                                    const labData = await labRes.json();
                                    setLabels(prev => ({ ...prev, [year]: labData }));
                                }
                            } catch (e) { /* no labels */ }
                        }
                    }

                    // Select first available year
                    if (data.result_years.length > 0 && !selectedYear) {
                        setSelectedYear(data.result_years[data.result_years.length - 1]);
                    }
                } else if (data.status === 'error') {
                    clearInterval(pollingRef.current);
                }
            } catch (e) {
                console.error('Poll error:', e);
            }
        };

        poll();
        pollingRef.current = setInterval(poll, 2000);
        return () => clearInterval(pollingRef.current);
    }, [jobId]);

    // Handle map clicks for drawing
    const onMapClick = useCallback((info, event) => {
        if (!drawMode) return;
        const [lng, lat] = info.coordinate || [];
        if (!lng) return;

        if (!drawStart) {
            setDrawStart([lng, lat]);
        } else {
            setDrawEnd([lng, lat]);
            const west = Math.min(drawStart[0], lng);
            const south = Math.min(drawStart[1], lat);
            const east = Math.max(drawStart[0], lng);
            const north = Math.max(drawStart[1], lat);
            setBbox([west, south, east, north]);
            setDrawMode(false);
        }
    }, [drawMode, drawStart]);

    // Submit job
    const submitJob = async (yearList) => {
        if (!bbox) return;
        setJobStatus(null);
        setGrid(null);
        setResults({});
        setLabels({});
        setSelectedYear(null);

        try {
            const res = await fetch(`${API}/api/deploy`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ bbox, years: yearList }),
            });
            const data = await res.json();
            setJobId(data.job_id);
        } catch (e) {
            console.error('Submit error:', e);
        }
    };

    // Reset drawing
    const resetDraw = () => {
        setDrawStart(null);
        setDrawEnd(null);
        setBbox(null);
        setJobId(null);
        setJobStatus(null);
        setGrid(null);
        setResults({});
        setLabels({});
        setSelectedYear(null);
    };

    // Get current display data
    const getCurrentData = () => {
        if (!selectedYear) return null;
        if (viewMode === 'labels') return labels[selectedYear] || null;
        return results[selectedYear] || labels[selectedYear] || null;
    };

    // Build layers
    const layers = [];

    // Bbox rectangle layer
    if (bbox) {
        const [west, south, east, north] = bbox;
        layers.push(new GeoJsonLayer({
            id: 'bbox-layer',
            data: {
                type: 'FeatureCollection',
                features: [{
                    type: 'Feature',
                    geometry: {
                        type: 'Polygon',
                        coordinates: [[
                            [west, south], [east, south],
                            [east, north], [west, north],
                            [west, south],
                        ]],
                    },
                    properties: {},
                }],
            },
            filled: true,
            stroked: true,
            getFillColor: [59, 130, 246, 30],
            getLineColor: [59, 130, 246, 200],
            getLineWidth: 2,
            lineWidthUnits: 'pixels',
        }));
    }

    // Draw preview (first point placed)
    if (drawStart && !drawEnd) {
        layers.push(new GeoJsonLayer({
            id: 'draw-start-layer',
            data: {
                type: 'FeatureCollection',
                features: [{
                    type: 'Feature',
                    geometry: { type: 'Point', coordinates: drawStart },
                    properties: {},
                }],
            },
            pointRadiusMinPixels: 6,
            getFillColor: [59, 130, 246, 255],
            getLineColor: [255, 255, 255, 255],
            getLineWidth: 2,
            lineWidthUnits: 'pixels',
        }));
    }

    // Results grid layer
    const viewData = getCurrentData();
    if (grid && viewData) {
        layers.push(new GeoJsonLayer({
            id: 'deploy-grid-layer',
            data: grid,
            filled: true,
            stroked: false,
            pickable: true,
            getFillColor: (feature) => {
                const cellId = String(feature.properties.cell_id);
                const data = viewData[cellId];
                if (!data) return [30, 41, 59, 60];

                if (selectedClass !== 'all') {
                    const val = data[selectedClass] ?? 0;
                    const color = CLASS_COLORS[selectedClass];
                    const intensity = Math.min(val * 100 / 100, 1);
                    return [
                        Math.round(color[0] * intensity + 30 * (1 - intensity)),
                        Math.round(color[1] * intensity + 41 * (1 - intensity)),
                        Math.round(color[2] * intensity + 59 * (1 - intensity)),
                        180,
                    ];
                }

                // Dominant class coloring
                let maxVal = -1, maxClass = null;
                for (const cls of CLASSES) {
                    const v = data[cls] ?? 0;
                    if (v > maxVal) {
                        maxVal = v;
                        maxClass = cls;
                    }
                }
                if (!maxClass) return [30, 41, 59, 60];
                const color = CLASS_COLORS[maxClass];
                const alpha = Math.min(Math.max(maxVal * 200 + 50, 80), 220);
                return [...color, alpha];
            },
            getTooltip: ({ object }) => {
                if (!object) return null;
                const cellId = String(object.properties.cell_id);
                const data = viewData[cellId];
                if (!data) return null;
                const lines = CLASSES.map(c =>
                    `${CLASS_LABELS[c]}: ${((data[c] ?? 0) * 100).toFixed(1)}%`
                ).join('<br/>');
                return {
                    html: `<div class="tooltip-title">Cell ${cellId}</div>
                           <div style="font-size:11px">${lines}</div>`,
                    className: 'deck-tooltip',
                };
            },
            updateTriggers: {
                getFillColor: [selectedYear, selectedClass, viewMode, viewData],
            },
        }));
    }

    const getCursor = ({ isDragging }) => {
        if (drawMode) return 'crosshair';
        return isDragging ? 'grabbing' : 'grab';
    };

    return (
        <div className="deploy-container">
            <DeployPanel
                drawMode={drawMode}
                onToggleDraw={() => {
                    setDrawMode(!drawMode);
                    setDrawStart(null);
                    setDrawEnd(null);
                }}
                bbox={bbox}
                onReset={resetDraw}
                onSubmit={submitJob}
                jobStatus={jobStatus}
                selectedYear={selectedYear}
                onYearChange={setSelectedYear}
                selectedClass={selectedClass}
                onClassChange={setSelectedClass}
                viewMode={viewMode}
                onViewModeChange={setViewMode}
                results={results}
                labels={labels}
                viewData={viewData}
            />
            <div className="deploy-map">
                <DeckGL
                    initialViewState={INITIAL_VIEW}
                    controller={!drawMode}
                    layers={layers}
                    onClick={onMapClick}
                    getCursor={getCursor}
                    getTooltip={({ object }) => {
                        if (!object) return null;
                        const cellId = String(object.properties?.cell_id);
                        const data = viewData?.[cellId];
                        if (!data) return null;
                        const lines = CLASSES.map(c =>
                            `${CLASS_LABELS[c]}: ${((data[c] ?? 0) * 100).toFixed(1)}%`
                        ).join('<br/>');
                        return {
                            html: `<div class="tooltip-title">Cell ${cellId}</div>
                                   <div style="font-size:11px">${lines}</div>`,
                            className: 'deck-tooltip',
                        };
                    }}
                >
                    <Map
                        ref={mapRef}
                        mapStyle="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
                    />
                </DeckGL>

                {drawMode && (
                    <div className="draw-hint">
                        {!drawStart
                            ? '🖱️ Click to set the first corner of your region'
                            : '🖱️ Click to set the opposite corner'}
                    </div>
                )}
            </div>
        </div>
    );
}
