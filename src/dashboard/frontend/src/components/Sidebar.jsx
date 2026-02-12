const VIEW_MODES = [
    { key: 'labels', label: 'Labels' },
    { key: 'predictions', label: 'Predicted' },
    { key: 'change', label: 'Change' },
];

const MODEL_DISPLAY = {
    ridge: 'Ridge',
    elasticnet: 'ElasticNet',
    extratrees: 'ExtraTrees',
    rf: 'Random Forest',
    catboost: 'CatBoost',
    mlp: 'MLP',
};

export default function Sidebar({
    models,
    selectedModel,
    onModelChange,
    viewMode,
    onViewModeChange,
    selectedYear,
    onYearChange,
    changeYearFrom,
    onChangeYearFromChange,
    changeYearTo,
    onChangeYearToChange,
    selectedClass,
    onClassChange,
    classes,
    classLabels,
    classColors,
    labelYears,
    allYears,
    isFutureYear,
}) {
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

            {/* Year Selector */}
            <div className="section">
                <div className="section-title">
                    {viewMode === 'change' ? 'Year Range' : 'Year'}
                </div>
                {viewMode === 'change' ? (
                    <div className="year-range">
                        <select
                            className="select"
                            value={changeYearFrom}
                            onChange={(e) => onChangeYearFromChange(Number(e.target.value))}
                        >
                            {allYears.map((y) => (
                                <option key={y} value={y}>{y}{!labelYears.includes(y) ? ' (pred)' : ''}</option>
                            ))}
                        </select>
                        <span className="year-range-arrow">&rarr;</span>
                        <select
                            className="select"
                            value={changeYearTo}
                            onChange={(e) => onChangeYearToChange(Number(e.target.value))}
                        >
                            {allYears.map((y) => (
                                <option key={y} value={y}>{y}{!labelYears.includes(y) ? ' (pred)' : ''}</option>
                            ))}
                        </select>
                    </div>
                ) : (
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
                )}
                {isFutureYear && viewMode === 'labels' && (
                    <div className="info-badge">
                        No labels for {selectedYear} &mdash; showing model predictions
                    </div>
                )}
            </div>

            {/* Model Selector */}
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

            {/* Class Filter */}
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

            {/* Legend */}
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
