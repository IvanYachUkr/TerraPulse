import { useState } from 'react';
import { REGION_COLORS } from '../data/trainingRegions.js';

const CLASS_LABELS = {
    tree_cover: 'Tree Cover',
    shrubland: 'Shrubland',
    grassland: 'Grassland',
    cropland: 'Cropland',
    built_up: 'Built-up',
    bare_sparse: 'Bare/Sparse',
    water: 'Water',
};

const CLASSES = ['tree_cover', 'shrubland', 'grassland', 'cropland', 'built_up', 'bare_sparse', 'water'];

const CLASS_COLORS_HEX = {
    tree_cover: '#2d6a4f',
    shrubland: '#6a994e',
    grassland: '#95d5b2',
    cropland: '#f4a261',
    built_up: '#e76f51',
    bare_sparse: '#d4a373',
    water: '#0096c7',
};

const AVAILABLE_YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025];

export default function DeployPanel({
    drawMode, onToggleDraw, bbox, onReset, onClearAll, onSubmit,
    jobStatus, selectedYear, onYearChange,
    showRegions, onToggleRegions,
    selectedClass, onClassChange,
    viewMode, onViewModeChange,
    results, labels, viewData,
}) {
    const [selectedYears, setSelectedYears] = useState(
        new Set([2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025])
    );

    const toggleYear = (y) => {
        const next = new Set(selectedYears);
        if (next.has(y)) next.delete(y);
        else next.add(y);
        setSelectedYears(next);
    };

    const isRunning = jobStatus?.status === 'running';
    const isComplete = jobStatus?.status === 'complete';
    const isError = jobStatus?.status === 'error';
    const availableResultYears = jobStatus?.result_years || [];
    const hasLabels = selectedYear && labels?.[selectedYear];

    // Compute summary stats for current view
    const computeSummary = () => {
        if (!viewData || !selectedYear) return null;
        const cells = Object.values(viewData);
        if (cells.length === 0) return null;

        const totals = {};
        for (const c of CLASSES) {
            const values = cells.map(d => d[c] ?? 0);
            totals[c] = values.reduce((a, b) => a + b, 0) / values.length;
        }
        return totals;
    };

    const summary = computeSummary();

    return (
        <div className="deploy-panel">
            <div className="deploy-panel-section">
                <h3 className="deploy-section-title">
                    <span className="deploy-icon">🚀</span> Deploy Pipeline
                </h3>
                <p className="deploy-hint">
                    Draw a rectangle on the map, then run the pipeline to generate predictions.
                </p>
            </div>

            {/* Training regions toggle + legend */}
            <div className="deploy-panel-section">
                <h4 className="deploy-section-subtitle">Training Data</h4>
                <button
                    className={`deploy-btn ${showRegions ? 'deploy-btn-active' : 'deploy-btn-ghost'}`}
                    onClick={onToggleRegions}
                    style={{ marginBottom: showRegions ? 8 : 0 }}
                >
                    {showRegions ? '🗺️ Hide Training Regions' : '🗺️ Show Training Regions'}
                </button>
                {showRegions && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 5, marginTop: 4 }}>
                        {Object.entries(REGION_COLORS).map(([role, cfg]) => (
                            <div key={role} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 12 }}>
                                <span style={{
                                    width: 14, height: 14, borderRadius: 3,
                                    background: cfg.hex, opacity: 0.85,
                                    border: `2px solid ${cfg.hex}`,
                                    display: 'inline-block', flexShrink: 0,
                                }} />
                                <span style={{ color: 'rgba(255,255,255,0.85)' }}>{cfg.label}</span>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            {/* Drawing controls */}
            <div className="deploy-panel-section">
                <h4 className="deploy-section-subtitle">Region</h4>
                {!bbox ? (
                    <button
                        className={`deploy-btn ${drawMode ? 'deploy-btn-active' : 'deploy-btn-primary'}`}
                        onClick={onToggleDraw}
                    >
                        {drawMode ? '✋ Cancel Drawing' : '✏️ Draw Region'}
                    </button>
                ) : (
                    <div className="deploy-bbox-info">
                        <div className="deploy-bbox-coords">
                            <span>W: {bbox[0].toFixed(4)}°</span>
                            <span>S: {bbox[1].toFixed(4)}°</span>
                            <span>E: {bbox[2].toFixed(4)}°</span>
                            <span>N: {bbox[3].toFixed(4)}°</span>
                        </div>
                        <button className="deploy-btn deploy-btn-ghost" onClick={() => { onReset(); }}>
                            ↺ Redraw
                        </button>
                        {onClearAll && (
                            <button className="deploy-btn deploy-btn-ghost" onClick={onClearAll}
                                style={{ marginTop: '4px', opacity: 0.7, fontSize: '0.85em' }}>
                                🗑️ Clear All Regions
                            </button>
                        )}
                    </div>
                )}
            </div>

            {/* Year selection */}
            {bbox && !isRunning && !isComplete && (
                <div className="deploy-panel-section">
                    <h4 className="deploy-section-subtitle">Years</h4>
                    <div className="deploy-year-grid">
                        {AVAILABLE_YEARS.map(y => (
                            <button
                                key={y}
                                className={`deploy-year-btn ${selectedYears.has(y) ? 'selected' : ''}`}
                                onClick={() => toggleYear(y)}
                            >
                                {y}
                                <span className="deploy-year-badge">
                                    {(y === 2020 || y === 2021) ? 'Label' : 'Pred'}
                                </span>
                            </button>
                        ))}
                    </div>
                    <button
                        className="deploy-btn deploy-btn-primary deploy-btn-run"
                        onClick={() => onSubmit(Array.from(selectedYears).sort())}
                        disabled={selectedYears.size === 0}
                    >
                        ▶ Run Pipeline ({selectedYears.size} years)
                    </button>
                </div>
            )}

            {/* Progress */}
            {isRunning && jobStatus && (
                <div className="deploy-panel-section">
                    <h4 className="deploy-section-subtitle">Pipeline Progress</h4>
                    <div className="deploy-progress-bar">
                        <div
                            className="deploy-progress-fill"
                            style={{ width: `${jobStatus.progress}%` }}
                        />
                    </div>
                    <div className="deploy-stage">{jobStatus.stage}</div>
                    <div className="deploy-messages">
                        {(jobStatus.messages || []).slice(-5).map((msg, i) => (
                            <div key={i} className="deploy-message">{msg}</div>
                        ))}
                    </div>
                </div>
            )}

            {/* Error */}
            {isError && (
                <div className="deploy-panel-section">
                    <div className="deploy-error">
                        <strong>Error:</strong> {jobStatus.error}
                    </div>
                    <button className="deploy-btn deploy-btn-ghost" onClick={onReset}>
                        ↺ Try Again
                    </button>
                </div>
            )}

            {/* Results */}
            {isComplete && (
                <>
                    <div className="deploy-panel-section">
                        <h4 className="deploy-section-subtitle">
                            ✅ Results ({jobStatus.grid_cells} cells)
                        </h4>

                        {/* View mode toggle */}
                        <div className="deploy-view-modes">
                            <button
                                className={`deploy-mode-btn ${viewMode === 'predictions' ? 'active' : ''}`}
                                onClick={() => onViewModeChange('predictions')}
                            >
                                Predicted
                            </button>
                            <button
                                className={`deploy-mode-btn ${viewMode === 'labels' ? 'active' : ''}`}
                                onClick={() => onViewModeChange('labels')}
                            >
                                Labels
                            </button>
                        </div>
                    </div>

                    {/* Year selector */}
                    <div className="deploy-panel-section">
                        <h4 className="deploy-section-subtitle">Year</h4>
                        <div className="deploy-year-results">
                            {availableResultYears.map(y => (
                                <button
                                    key={y}
                                    className={`deploy-year-btn ${y === selectedYear ? 'selected' : ''}`}
                                    onClick={() => onYearChange(y)}
                                >
                                    {y}
                                    <span className="deploy-year-badge">
                                        {labels?.[y] ? '✓ Label' : 'Pred'}
                                    </span>
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Class filter */}
                    <div className="deploy-panel-section">
                        <h4 className="deploy-section-subtitle">Class Filter</h4>
                        <div className="deploy-class-list">
                            <button
                                className={`deploy-class-btn ${selectedClass === 'all' ? 'active' : ''}`}
                                onClick={() => onClassChange('all')}
                            >
                                <span className="deploy-class-dot" style={{ background: '#94a3b8' }} />
                                Dominant
                            </button>
                            {CLASSES.map(c => (
                                <button
                                    key={c}
                                    className={`deploy-class-btn ${selectedClass === c ? 'active' : ''}`}
                                    onClick={() => onClassChange(c)}
                                >
                                    <span
                                        className="deploy-class-dot"
                                        style={{ background: CLASS_COLORS_HEX[c] }}
                                    />
                                    {CLASS_LABELS[c]}
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Summary stats */}
                    {summary && (
                        <div className="deploy-panel-section">
                            <h4 className="deploy-section-subtitle">
                                Region Summary ({selectedYear})
                            </h4>
                            <div className="deploy-summary">
                                {CLASSES.map(c => {
                                    const pct = (summary[c] * 100).toFixed(1);
                                    return (
                                        <div key={c} className="deploy-summary-row">
                                            <span
                                                className="deploy-class-dot"
                                                style={{ background: CLASS_COLORS_HEX[c] }}
                                            />
                                            <span className="deploy-summary-label">
                                                {CLASS_LABELS[c]}
                                            </span>
                                            <div className="deploy-summary-bar-bg">
                                                <div
                                                    className="deploy-summary-bar-fill"
                                                    style={{
                                                        width: `${Math.min(pct, 100)}%`,
                                                        background: CLASS_COLORS_HEX[c],
                                                    }}
                                                />
                                            </div>
                                            <span className="deploy-summary-pct">{pct}%</span>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    )}

                    {/* Accuracy when labels available */}
                    {hasLabels && results[selectedYear] && (
                        <div className="deploy-panel-section">
                            <h4 className="deploy-section-subtitle">
                                Accuracy vs Labels ({selectedYear})
                            </h4>
                            <div className="deploy-accuracy">
                                {CLASSES.map(c => {
                                    const cells = Object.keys(labels[selectedYear]);
                                    const maes = cells.map(id => {
                                        const pred = results[selectedYear]?.[id]?.[c] ?? 0;
                                        const lab = labels[selectedYear]?.[id]?.[c] ?? 0;
                                        return Math.abs(pred - lab);
                                    });
                                    const mae = (maes.reduce((a, b) => a + b, 0) / maes.length * 100).toFixed(2);
                                    return (
                                        <div key={c} className="deploy-summary-row">
                                            <span
                                                className="deploy-class-dot"
                                                style={{ background: CLASS_COLORS_HEX[c] }}
                                            />
                                            <span className="deploy-summary-label">
                                                {CLASS_LABELS[c]}
                                            </span>
                                            <span className="deploy-accuracy-val">
                                                MAE: {mae}pp
                                            </span>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    )}
                </>
            )}
        </div>
    );
}
