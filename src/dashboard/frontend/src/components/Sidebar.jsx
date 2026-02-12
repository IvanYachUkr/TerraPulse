const VIEW_MODES = [
    { key: 'labels', label: 'Labels' },
    { key: 'predictions', label: 'Predicted' },
    { key: 'change', label: 'Change' },
    { key: 'folds', label: 'Folds' },
];

const MODEL_DISPLAY = {
    ridge: 'Ridge',
    elasticnet: 'ElasticNet',
    extratrees: 'ExtraTrees',
    rf: 'Random Forest',
    catboost: 'CatBoost',
    mlp: 'MLP',
};

const FOLD_LABELS = [
    { fold: 0, label: 'Fold 0 (Holdout)', color: '#3b82f6' },
    { fold: 1, label: 'Fold 1', color: '#ef4444' },
    { fold: 2, label: 'Fold 2', color: '#10b981' },
    { fold: 3, label: 'Fold 3', color: '#f59e0b' },
    { fold: 4, label: 'Fold 4', color: '#8b5cf6' },
];

export default function Sidebar({
    models,
    selectedModel,
    onModelChange,
    viewMode,
    onViewModeChange,
    selectedYear,
    onYearChange,
    selectedClass,
    onClassChange,
    classes,
    classLabels,
    classColors,
    labelYears,
    allYears,
    isFutureYear,
    searchCellId,
    onSearchCellId,
}) {
    const showModels = viewMode === 'predictions' || viewMode === 'labels';
    const showYears = viewMode === 'labels';
    const showClasses = viewMode !== 'folds';

    return (
        <aside className="sidebar">
            {/* View Mode */}
            <div className="section">
                <div className="section-title">View Mode</div>
                <div className="toggle-group">
                    {VIEW_MODES.map((m) => (
                        <button
                            key={m.key}
                            className={`toggle-btn ${viewMode === m.key ? 'active' : ''}`}
                            onClick={() => onViewModeChange(m.key)}
                        >
                            {m.label}
                        </button>
                    ))}
                </div>
            </div>

            {showYears && (
                <div className="section">
                    <div className="section-title">Year</div>
                    <select
                        className="select"
                        value={selectedYear}
                        onChange={(e) => onYearChange(Number(e.target.value))}
                    >
                        {allYears.map((y) => (
                            <option key={y} value={y}>
                                {y}{!labelYears.includes(y) ? ' (predicted)' : ''}
                            </option>
                        ))}
                    </select>
                    {isFutureYear && viewMode === 'labels' && (
                        <div className="info-badge">
                            No labels for {selectedYear} &mdash; showing model predictions
                        </div>
                    )}
                </div>
            )}

            {/* Change mode note */}
            {viewMode === 'change' && (
                <div className="section">
                    <div className="section-title">Year Range</div>
                    <div className="info-badge">
                        Showing change: 2020 &rarr; 2021
                    </div>
                </div>
            )}

            {/* Model Selector — only for prediction-relevant views */}
            {showModels && (
                <div className="section">
                    <div className="section-title">Model</div>
                    <div className="model-list">
                        {models &&
                            models.map((m) => (
                                <div
                                    key={m.model}
                                    className={`model-item ${selectedModel === m.model ? 'active' : ''}`}
                                    onClick={() => onModelChange(m.model)}
                                >
                                    <span className="model-name">{MODEL_DISPLAY[m.model] || m.model}</span>
                                    <span className="model-r2">{m.r2_uniform.toFixed(3)}</span>
                                    <span className="model-mae">{m.mae_mean_pp.toFixed(1)} pp</span>
                                </div>
                            ))}
                    </div>
                </div>
            )}

            {/* Fold Legend — only in folds view */}
            {viewMode === 'folds' && (
                <div className="section">
                    <div className="section-title">Spatial CV Folds</div>
                    <div className="fold-legend">
                        {FOLD_LABELS.map(({ fold, label, color }) => (
                            <div key={fold} className="fold-item">
                                <span className="fold-swatch" style={{ backgroundColor: color }} />
                                <span>{label}</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Class Filter */}
            {showClasses && (
                <div className="section">
                    <div className="section-title">Land-Cover Class</div>
                    <div className="class-chips">
                        <button
                            className={`class-chip ${selectedClass === 'all' ? 'active' : ''}`}
                            onClick={() => onClassChange('all')}
                        >
                            All
                        </button>
                        {classes.map((c) => {
                            const [r, g, b] = classColors[c];
                            return (
                                <button
                                    key={c}
                                    className={`class-chip ${selectedClass === c ? 'active' : ''}`}
                                    style={{ '--chip-color': `rgb(${r},${g},${b})` }}
                                    onClick={() => onClassChange(c)}
                                >
                                    <span
                                        className="class-dot"
                                        style={{ backgroundColor: `rgb(${r},${g},${b})` }}
                                    />
                                    {classLabels[c]}
                                </button>
                            );
                        })}
                    </div>
                </div>
            )}

            {/* Legend */}
            {showClasses && (
                <div className="section">
                    <div className="section-title">Legend</div>
                    {viewMode === 'change' ? (
                        <div className="legend">
                            <div className="legend-bar diverging" />
                            <div className="legend-labels">
                                <span>-30%</span>
                                <span>0</span>
                                <span>+30%</span>
                            </div>
                        </div>
                    ) : (
                        <div className="legend">
                            <div className="legend-bar" />
                            <div className="legend-labels">
                                <span>0%</span>
                                <span>50%</span>
                                <span>100%</span>
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* Search Cell */}
            <div className="section">
                <div className="section-title">Search Cell</div>
                <input
                    className="select"
                    type="number"
                    min="0"
                    max="29945"
                    placeholder="Cell ID (0-29945)"
                    value={searchCellId ?? ''}
                    onChange={(e) => {
                        const val = e.target.value;
                        onSearchCellId(val === '' ? null : Number(val));
                    }}
                />
            </div>

            {/* Disclaimer */}
            <div className="section">
                <div className="disclaimer">
                    <span className="disclaimer-icon">&#9888;</span>
                    <strong>Caveat:</strong> Labels use ESA WorldCover v100 (2020) vs v200 (2021).
                    Algorithm differences may create apparent change that is not real land-cover change.
                    Predictions shown only for the holdout fold (6,100 cells).
                </div>
            </div>
        </aside>
    );
}
