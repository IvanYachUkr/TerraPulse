import { useState, useEffect, useCallback, useRef } from 'react';
import DeckGL from '@deck.gl/react';
import { Map } from 'react-map-gl/maplibre';
import { GeoJsonLayer } from '@deck.gl/layers';
import DeployPanel from './DeployPanel.jsx';
import { REGIONS_GEOJSON, REGION_COLORS } from '../data/trainingRegions.js';

const API = import.meta.env.VITE_API_URL || '';

const INITIAL_VIEW = {
    longitude: 11.08,
    latitude: 49.45,
    zoom: 12,
    pitch: 0,
    bearing: 0,
};

const CLASSES = ['tree_cover', 'shrubland', 'grassland', 'cropland', 'built_up', 'bare_sparse', 'water'];

const CLASS_COLORS = {
    tree_cover: [45, 106, 79],
    shrubland: [106, 153, 78],
    grassland: [149, 213, 178],
    cropland: [244, 162, 97],
    built_up: [231, 111, 81],
    bare_sparse: [212, 163, 115],
    water: [0, 150, 199],
};

const CLASS_LABELS = {
    tree_cover: 'Tree Cover',
    shrubland: 'Shrubland',
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
    const [mapStyle, setMapStyle] = useState('dark'); // dark | satellite
    const [showRegions, setShowRegions] = useState(false);


    // Persistent completed jobs — survives across redraws
    const [completedJobs, setCompletedJobs] = useState([]);

    const pollingRef = useRef(null);
    const mapRef = useRef(null);
    const [viewState, setViewState] = useState(INITIAL_VIEW);

    // Polling for job status
    useEffect(() => {
        if (!jobId) return;

        const poll = async () => {
            try {
                const res = await fetch(`${API}/api/deploy/status/${jobId}`);
                if (!res.ok) return;
                const data = await res.json();
                setJobStatus(data);

                // Fetch grid if not loaded
                if (data.grid_cells > 0 && !grid) {
                    const gRes = await fetch(`${API}/api/deploy/grid/${jobId}`);
                    if (gRes.ok) {
                        const gData = await gRes.json();
                        setGrid(gData);

                        // Auto-center on grid removed as per user request
                    }
                }

                // Fetch newly available results and labels
                for (const year of data.result_years) {
                    if (!results[year]) {
                        console.log(`[Deploy] Fetching results for ${year}...`);
                        const rRes = await fetch(`${API}/api/deploy/results/${jobId}/${year}`);
                        if (rRes.ok) {
                            const rData = await rRes.json();
                            const isEmpty = Object.keys(rData).length === 0;
                            if (!isEmpty) {
                                setResults(prev => ({ ...prev, [year]: rData }));
                                setSelectedYear(current => current || year);
                            }
                        }
                    }
                    if (!labels[year]) {
                        const lRes = await fetch(`${API}/api/deploy/labels/${jobId}/${year}`);
                        if (lRes.ok) {
                            const lData = await lRes.json();
                            setLabels(prev => ({ ...prev, [year]: lData }));
                        }
                    }
                }

                // Periodic check for ground truth (2020, 2021) only if job is complete
                // or if specifically marked available
                if (data.status === 'complete') {
                    for (const year of [2020, 2021]) {
                        if (!labels[year]) {
                            const lRes = await fetch(`${API}/api/deploy/labels/${jobId}/${year}`);
                            if (lRes.ok) {
                                const lData = await lRes.json();
                                setLabels(prev => ({ ...prev, [year]: lData }));
                            }
                        }
                    }
                }


                if (data.status === 'complete' || data.status === 'error') {
                    if (pollingRef.current) clearInterval(pollingRef.current);
                }
            } catch (e) {
                console.error('Poll error:', e);
            }
        };

        const interval = setInterval(poll, 2000);
        pollingRef.current = interval;
        poll();
        return () => clearInterval(interval);
    }, [jobId]);

    // Save completed job to persistent list
    useEffect(() => {
        if (jobStatus?.status === 'complete' && grid && Object.keys(results).length > 0) {
            // Only save if not already saved (check jobId)
            setCompletedJobs(prev => {
                if (prev.some(j => j.jobId === jobId)) return prev;
                return [...prev, {
                    jobId,
                    grid: grid,
                    results: { ...results },
                    labels: { ...labels },
                    bbox: bbox,
                    selectedYear: selectedYear,
                }];
            });
        }
    }, [jobStatus?.status, grid, results]);

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
        // Reset only active job state — keep completedJobs
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

    // Reset drawing (only active job — completed jobs persist)
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

    // Clear all persisted results
    const clearAll = () => {
        resetDraw();
        setCompletedJobs([]);
    };

    // Get current display data
    const getCurrentData = () => {
        if (!selectedYear) return null;
        if (viewMode === 'labels') return labels[selectedYear] || null;
        return results[selectedYear] || labels[selectedYear] || null;
    };

    // Helper: build a grid layer from job data
    const buildGridLayer = (id, gridData, jobResults, jobLabels, jobYear, opacity = 1.0, pickable = true) => {
        const jobViewData = jobYear
            ? (viewMode === 'labels' ? jobLabels[jobYear] : jobResults[jobYear] || jobLabels[jobYear])
            : null;

        return new GeoJsonLayer({
            id,
            data: gridData,
            filled: true,
            stroked: true,
            pickable,
            getFillColor: (feature) => {
                const props = feature.properties;
                const cellIdNum = props.cell_id !== undefined ? props.cell_id : props.cellID;
                const cellId = String(cellIdNum ?? '');

                if (!jobViewData) return [30, 41, 59, Math.round(20 * opacity)];
                const data = jobViewData[cellId];
                if (!data) return [30, 41, 59, Math.round(40 * opacity)];

                // Nodata cells: grey
                if (data._quality === 'nodata') return [80, 80, 80, Math.round(120 * opacity)];

                if (selectedClass !== 'all') {
                    const val = data[selectedClass] ?? 0;
                    const color = CLASS_COLORS[selectedClass] || [71, 85, 105];
                    const intensity = Math.min(val, 1);
                    return [
                        Math.round(color[0] * intensity + 30 * (1 - intensity)),
                        Math.round(color[1] * intensity + 41 * (1 - intensity)),
                        Math.round(color[2] * intensity + 59 * (1 - intensity)),
                        Math.round((180 * intensity + 40 * (1 - intensity)) * opacity),
                    ];
                }

                let maxVal = -1, maxClass = null;
                for (const cls of CLASSES) {
                    const v = data[cls] ?? 0;
                    if (v > maxVal) { maxVal = v; maxClass = cls; }
                }
                if (!maxClass) return [30, 41, 59, Math.round(100 * opacity)];
                const color = CLASS_COLORS[maxClass] || [255, 255, 255];
                const alpha = Math.min(Math.max(maxVal * 150 + 100, 120), 240);
                return [...color, Math.round(alpha * opacity)];
            },
            getLineColor: (feature) => {
                const props = feature.properties;
                const cellIdNum = props.cell_id !== undefined ? props.cell_id : props.cellID;
                const cellId = String(cellIdNum ?? '');
                if (!jobViewData) return [0, 0, 0, 0];
                const data = jobViewData[cellId];
                if (!data) return [0, 0, 0, 0];
                if (data._quality === 'nodata') return [120, 120, 120, Math.round(180 * opacity)];
                if (data._quality === 'low_data') return [220, 50, 50, Math.round(200 * opacity)];
                return [0, 0, 0, 0];
            },
            getLineWidth: (feature) => {
                const props = feature.properties;
                const cellIdNum = props.cell_id !== undefined ? props.cell_id : props.cellID;
                const cellId = String(cellIdNum ?? '');
                if (!jobViewData) return 0;
                const data = jobViewData[cellId];
                if (!data) return 0;
                if (data._quality === 'nodata' || data._quality === 'low_data') return 2;
                return 0;
            },
            lineWidthUnits: 'pixels',
            ...(pickable ? {
                getTooltip: (info) => {
                    const { object } = info;
                    if (!object) return null;
                    const props = object.properties || {};
                    const idNum = props.cell_id !== undefined ? props.cell_id : props.cellID;
                    const cellId = String(idNum ?? '');
                    const data = jobViewData ? jobViewData[cellId] : null;
                    if (!data) return `Cell ${cellId} (no data)`;
                    if (data._quality === 'nodata') {
                        return { html: `<div class="tooltip-title">Cell ${cellId}</div><div style="font-size:11px;color:#ef4444">⚠ No Data</div>`, className: 'deck-tooltip' };
                    }
                    const qualityTag = data._quality === 'low_data'
                        ? '<div style="font-size:10px;color:#f59e0b;margin-top:4px">⚠ Partial data</div>' : '';
                    const lines = CLASSES.map(c =>
                        `${CLASS_LABELS[c]}: ${((data[c] ?? 0) * 100).toFixed(1)}%`
                    ).join('<br/>');
                    return { html: `<div class="tooltip-title">Cell ${cellId}</div><div style="font-size:11px">${lines}</div>${qualityTag}`, className: 'deck-tooltip' };
                },
            } : {}),
            updateTriggers: {
                getFillColor: [jobYear, selectedClass, viewMode, jobViewData, opacity],
                getLineColor: [jobYear, jobViewData, opacity],
                getLineWidth: [jobYear, jobViewData],
            },
        });
    };

    // Build layers
    const layers = [];

    // Completed jobs — rendered as dimmed background layers
    // Training region overlay
    if (showRegions) {
        layers.push(new GeoJsonLayer({
            id: 'training-regions-layer',
            data: REGIONS_GEOJSON,
            filled: true,
            stroked: true,
            pickable: true,
            getFillColor: (f) => REGION_COLORS[f.properties.role]?.fill || [100, 100, 100, 30],
            getLineColor: (f) => REGION_COLORS[f.properties.role]?.stroke || [100, 100, 100, 150],
            getLineWidth: 2,
            lineWidthUnits: 'pixels',
        }));
    }

    completedJobs.forEach((job, idx) => {
        // Skip the active job if it's already in completedJobs
        if (job.jobId === jobId) return;
        const year = job.selectedYear || Object.keys(job.results)[0];
        layers.push(buildGridLayer(
            `completed-grid-${idx}`,
            job.grid,
            job.results,
            job.labels,
            parseInt(year),
            0.6, // dimmed
            true, // pickable — enables hover tooltips on completed jobs
        ));
    });

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

    // Active job results grid layer
    const viewData = getCurrentData();
    if (grid) {
        layers.push(new GeoJsonLayer({
            id: 'deploy-grid-layer',
            data: grid,
            filled: true,
            stroked: true,
            pickable: true,
            getFillColor: (feature) => {
                const props = feature.properties;
                const cellIdNum = props.cell_id !== undefined ? props.cell_id : props.cellID;
                const cellId = String(cellIdNum ?? '');

                if (viewData === null) return [30, 41, 59, 20];
                const data = viewData[cellId];
                if (data === null || data === undefined) return [30, 41, 59, 40];

                // Nodata cells: grey
                if (data._quality === 'nodata') return [80, 80, 80, 120];

                if (selectedClass !== 'all') {
                    const val = data[selectedClass] ?? 0;
                    const color = CLASS_COLORS[selectedClass] || [71, 85, 105];
                    const intensity = Math.min(val, 1);
                    return [
                        Math.round(color[0] * intensity + 30 * (1 - intensity)),
                        Math.round(color[1] * intensity + 41 * (1 - intensity)),
                        Math.round(color[2] * intensity + 59 * (1 - intensity)),
                        Math.round(180 * intensity + 40 * (1 - intensity)),
                    ];
                }

                let maxVal = -1, maxClass = null;
                for (const cls of CLASSES) {
                    const v = data[cls] ?? 0;
                    if (v > maxVal) { maxVal = v; maxClass = cls; }
                }
                if (!maxClass) return [30, 41, 59, 100];
                const color = CLASS_COLORS[maxClass] || [255, 255, 255];
                const alpha = Math.min(Math.max(maxVal * 150 + 100, 120), 240);
                return [...color, alpha];
            },
            getLineColor: (feature) => {
                const props = feature.properties;
                const cellIdNum = props.cell_id !== undefined ? props.cell_id : props.cellID;
                const cellId = String(cellIdNum ?? '');
                if (!viewData) return [0, 0, 0, 0];
                const data = viewData[cellId];
                if (!data) return [0, 0, 0, 0];
                if (data._quality === 'nodata') return [120, 120, 120, 180];
                if (data._quality === 'low_data') return [220, 50, 50, 200];
                return [0, 0, 0, 0];
            },
            getLineWidth: (feature) => {
                const props = feature.properties;
                const cellIdNum = props.cell_id !== undefined ? props.cell_id : props.cellID;
                const cellId = String(cellIdNum ?? '');
                if (!viewData) return 0;
                const data = viewData[cellId];
                if (!data) return 0;
                if (data._quality === 'nodata' || data._quality === 'low_data') return 2;
                return 0;
            },
            lineWidthUnits: 'pixels',
            getTooltip: (info) => {
                const { object } = info;
                if (!object) return null;
                const activeData = getCurrentData();
                const props = object.properties || {};
                const idNum = props.cell_id !== undefined ? props.cell_id : props.cellID;
                const cellId = String(idNum ?? '');
                const data = activeData ? activeData[cellId] : null;

                if (!data) return `Cell ${cellId} (Processing...)`;

                if (data._quality === 'nodata') {
                    return {
                        html: `<div class="tooltip-title">Cell ${cellId}</div>
                               <div style="font-size:11px;color:#ef4444">⚠ No Data — insufficient satellite coverage</div>`,
                        className: 'deck-tooltip',
                    };
                }

                const qualityTag = data._quality === 'low_data'
                    ? '<div style="font-size:10px;color:#f59e0b;margin-top:4px">⚠ Partial data — some features imputed</div>'
                    : '';
                const lines = CLASSES.map(c =>
                    `${CLASS_LABELS[c]}: ${((data[c] ?? 0) * 100).toFixed(1)}%`
                ).join('<br/>');
                return {
                    html: `<div class="tooltip-title">Cell ${cellId}</div>
                           <div style="font-size:11px">${lines}</div>${qualityTag}`,
                    className: 'deck-tooltip',
                };
            },
            updateTriggers: {
                getFillColor: [selectedYear, selectedClass, viewMode, viewData],
                getLineColor: [selectedYear, viewData],
                getLineWidth: [selectedYear, viewData],
            },
        }));
    }

    const getCursor = ({ isDragging }) => {
        if (drawMode) return 'crosshair';
        return isDragging ? 'grabbing' : 'grab';
    };

    // Unified tooltip handler for all pickable layers
    const getTooltip = useCallback((info) => {
        if (!info.object) return null;
        const layerId = info.layer?.id || '';

        // Training regions layer
        if (layerId === 'training-regions-layer') {
            const { name, role } = info.object.properties || {};
            const roleLabel = REGION_COLORS[role]?.label || role;
            return {
                html: `<div class="tooltip-title">${name}</div><div style="font-size:11px;opacity:0.8">${roleLabel} Region</div>`,
                className: 'deck-tooltip',
            };
        }

        // Grid layers (active or completed)
        if (layerId === 'deploy-grid-layer' || layerId.startsWith('completed-grid-')) {
            const props = info.object.properties || {};
            const idNum = props.cell_id !== undefined ? props.cell_id : props.cellID;
            const cellId = String(idNum ?? '');

            // For the active grid, use getCurrentData()
            // For completed grids, find the matching completed job
            let cellData = null;
            if (layerId === 'deploy-grid-layer') {
                const activeData = getCurrentData();
                cellData = activeData ? activeData[cellId] : null;
            } else {
                const idx = parseInt(layerId.replace('completed-grid-', ''), 10);
                const job = completedJobs.filter(j => j.jobId !== jobId)[idx];
                if (job) {
                    const year = job.selectedYear || Object.keys(job.results)[0];
                    const jData = viewMode === 'labels' ? job.labels[year] : (job.results[year] || job.labels[year]);
                    cellData = jData ? jData[cellId] : null;
                }
            }

            if (!cellData) return { html: `<div class="tooltip-title">Cell ${cellId}</div><div style="font-size:11px;opacity:0.6">Processing…</div>`, className: 'deck-tooltip' };
            if (cellData._quality === 'nodata') {
                return { html: `<div class="tooltip-title">Cell ${cellId}</div><div style="font-size:11px;color:#ef4444">⚠ No Data — insufficient satellite coverage</div>`, className: 'deck-tooltip' };
            }
            const qualityTag = cellData._quality === 'low_data'
                ? '<div style="font-size:10px;color:#f59e0b;margin-top:4px">⚠ Partial data — some features imputed</div>' : '';
            const lines = CLASSES.map(c =>
                `${CLASS_LABELS[c]}: ${((cellData[c] ?? 0) * 100).toFixed(1)}%`
            ).join('<br/>');
            return { html: `<div class="tooltip-title">Cell ${cellId}</div><div style="font-size:11px">${lines}</div>${qualityTag}`, className: 'deck-tooltip' };
        }

        return null;
    }, [viewMode, completedJobs, jobId, selectedYear, results, labels, grid, viewData]);

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
                onClearAll={completedJobs.length > 0 ? clearAll : null}
                onSubmit={submitJob}
                jobStatus={jobStatus}
                selectedYear={selectedYear}
                onYearChange={setSelectedYear}
                showRegions={showRegions}
                onToggleRegions={() => setShowRegions(s => !s)}
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
                    viewState={viewState}
                    onViewStateChange={({ viewState }) => setViewState(viewState)}
                    controller={!drawMode}
                    layers={layers}
                    onClick={onMapClick}
                    getCursor={getCursor}
                    getTooltip={getTooltip}
                >
                    <Map
                        ref={mapRef}
                        mapStyle={mapStyle === 'satellite'
                            ? { version: 8, sources: { 'esri-satellite': { type: 'raster', tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'], tileSize: 256, attribution: '© Esri' } }, layers: [{ id: 'esri-satellite-layer', type: 'raster', source: 'esri-satellite', minzoom: 0, maxzoom: 19 }] }
                            : 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json'
                        }
                    />
                </DeckGL>

                <button
                    onClick={() => setMapStyle(s => s === 'dark' ? 'satellite' : 'dark')}
                    style={{
                        position: 'absolute', top: 70, right: 20, zIndex: 10,
                        background: 'rgba(30,30,40,0.85)', color: '#fff',
                        border: '1px solid rgba(255,255,255,0.2)', borderRadius: 8,
                        padding: '8px 14px', cursor: 'pointer', fontSize: 13,
                        backdropFilter: 'blur(8px)', transition: 'all 0.2s',
                    }}
                    title={mapStyle === 'dark' ? 'Switch to satellite view' : 'Switch to dark map'}
                >
                    {mapStyle === 'dark' ? '🛰️ Satellite' : '🗺️ Dark Map'}
                </button>

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
