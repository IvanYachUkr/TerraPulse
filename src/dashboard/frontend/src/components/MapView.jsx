import { useMemo, useCallback } from 'react';
import { Map } from 'react-map-gl/maplibre';
import DeckGL from '@deck.gl/react';
import { GeoJsonLayer } from '@deck.gl/layers';
import 'maplibre-gl/dist/maplibre-gl.css';

const INITIAL_VIEW = {
    longitude: 11.076,
    latitude: 49.449,
    zoom: 12,
    pitch: 0,
    bearing: 0,
};

// Sequential blue scale for proportions
function proportionColor(v) {
    if (v == null || isNaN(v)) return [30, 41, 59, 100];
    const t = Math.max(0, Math.min(1, v));
    return [
        Math.round(30 + t * 29),   // R: 30 -> 59
        Math.round(64 + t * 66),   // G: 64 -> 130
        Math.round(120 + t * 126), // B: 120 -> 246
        Math.round(140 + t * 115), // A: 140 -> 255
    ];
}

// Diverging red-white-green for change
function changeColor(v) {
    if (v == null || isNaN(v)) return [30, 41, 59, 100];
    const clamped = Math.max(-0.3, Math.min(0.3, v));
    const t = (clamped + 0.3) / 0.6; // 0..1 where 0.5 = no change
    if (t < 0.5) {
        const s = t / 0.5;
        return [
            Math.round(239 + s * (240 - 239)),
            Math.round(68 + s * (244 - 68)),
            Math.round(68 + s * (248 - 68)),
            220,
        ];
    } else {
        const s = (t - 0.5) / 0.5;
        return [
            Math.round(240 - s * (240 - 16)),
            Math.round(244 - s * (244 - 185)),
            Math.round(248 - s * (248 - 129)),
            220,
        ];
    }
}

// Get the dominant class and its proportion
function dominantClass(props, classes) {
    let maxVal = -1, maxCls = null;
    for (const c of classes) {
        const v = props[c];
        if (v != null && v > maxVal) {
            maxVal = v;
            maxCls = c;
        }
    }
    return { cls: maxCls, val: maxVal };
}

export default function MapView({
    grid,
    viewData,
    viewMode,
    selectedClass,
    selectedModel,
    predictions,
    labels2020,
    labels2021,
    changeData,
    classColors,
    classes,
    classLabels,
    loading,
    onCellClick,
    selectedCell,
}) {
    const getFillColor = useCallback(
        (feature) => {
            const cellId = String(feature.properties.cell_id);
            let data;

            // Pick data source based on view mode
            if (viewMode === 'labels_2020') data = labels2020?.[cellId];
            else if (viewMode === 'labels_2021') data = labels2021?.[cellId];
            else if (viewMode === 'predictions') data = predictions?.[cellId];
            else if (viewMode === 'change') data = changeData?.[cellId];
            else data = labels2021?.[cellId];

            if (!data) return [30, 41, 59, 80]; // No data - dim

            if (viewMode === 'change') {
                if (selectedClass !== 'all') {
                    const key = `delta_${selectedClass}`;
                    return changeColor(data[key]);
                }
                // "All" for change: show max absolute change
                let maxAbs = 0, maxVal = 0;
                for (const c of classes) {
                    const v = data[`delta_${c}`];
                    if (v != null && Math.abs(v) > maxAbs) {
                        maxAbs = Math.abs(v);
                        maxVal = v;
                    }
                }
                return changeColor(maxVal);
            }

            // Proportion views
            if (selectedClass !== 'all') {
                return proportionColor(data[selectedClass]);
            }

            // "All": color by dominant class
            const dom = dominantClass(data, classes);
            if (dom.cls && classColors[dom.cls]) {
                const [r, g, b] = classColors[dom.cls];
                const alpha = Math.round(120 + dom.val * 135);
                return [r, g, b, alpha];
            }
            return [30, 41, 59, 100];
        },
        [viewMode, selectedClass, labels2020, labels2021, predictions, changeData, classColors, classes]
    );

    const getLineColor = useCallback(
        (feature) => {
            if (feature.properties.cell_id === selectedCell) {
                return [255, 255, 255, 255];
            }
            return [255, 255, 255, 15];
        },
        [selectedCell]
    );

    const getLineWidth = useCallback(
        (feature) => {
            return feature.properties.cell_id === selectedCell ? 3 : 0.5;
        },
        [selectedCell]
    );

    const layer = useMemo(() => {
        if (!grid) return null;
        return new GeoJsonLayer({
            id: 'grid-layer',
            data: grid,
            pickable: true,
            stroked: true,
            filled: true,
            getFillColor,
            getLineColor,
            getLineWidth,
            lineWidthUnits: 'pixels',
            updateTriggers: {
                getFillColor: [viewMode, selectedClass, predictions, labels2020, labels2021, changeData],
                getLineColor: [selectedCell],
                getLineWidth: [selectedCell],
            },
            onClick: (info) => {
                if (info.object) {
                    onCellClick(info.object.properties.cell_id);
                }
            },
        });
    }, [grid, getFillColor, getLineColor, getLineWidth, viewMode, selectedClass, predictions, labels2020, labels2021, changeData, selectedCell, onCellClick]);

    const getTooltip = useCallback(
        ({ object }) => {
            if (!object) return null;
            const cellId = String(object.properties.cell_id);
            let data;
            if (viewMode === 'labels_2020') data = labels2020?.[cellId];
            else if (viewMode === 'labels_2021') data = labels2021?.[cellId];
            else if (viewMode === 'predictions') data = predictions?.[cellId];
            else if (viewMode === 'change') data = changeData?.[cellId];
            else data = labels2021?.[cellId];

            if (!data) return { text: `Cell ${cellId}\nNo data` };

            let rows;
            if (viewMode === 'change') {
                rows = classes
                    .map((c) => {
                        const v = data[`delta_${c}`];
                        return `${classLabels[c]}: ${v != null ? (v > 0 ? '+' : '') + (v * 100).toFixed(1) + 'pp' : 'N/A'}`;
                    })
                    .join('\n');
            } else {
                rows = classes
                    .map((c) => `${classLabels[c]}: ${data[c] != null ? (data[c] * 100).toFixed(1) + '%' : 'N/A'}`)
                    .join('\n');
            }

            return {
                html: `<div class="tooltip-title">Cell ${cellId}</div><div style="white-space: pre-line; font-size: 11px;">${rows}</div>`,
                className: 'deck-tooltip',
            };
        },
        [viewMode, labels2020, labels2021, predictions, changeData, classes, classLabels]
    );

    return (
        <div className="map-container">
            {loading && (
                <div className="loading-overlay">
                    <div className="spinner" />
                </div>
            )}
            <DeckGL
                initialViewState={INITIAL_VIEW}
                controller={true}
                layers={layer ? [layer] : []}
                getTooltip={getTooltip}
                style={{ width: '100%', height: '100%' }}
            >
                <Map
                    mapStyle="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
                    attributionControl={false}
                />
            </DeckGL>
        </div>
    );
}
