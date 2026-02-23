const VIEW_MODES = [
    { key: 'labels', label: 'ESA' },
    { key: 'predictions', label: 'Prediction' },
    { key: 'change', label: 'Change' },
    { key: 'future', label: 'Future' },
];

const MODEL_DISPLAY = {
    mlp: 'MLP',
    tree: 'LightGBM',
    ridge: 'Ridge',
};



// Years available for predictions (OOF 2021 + pipeline 2022-2025)
const PREDICTION_YEARS = [2021, 2022, 2023, 2024, 2025];
const FUTURE_YEARS = [2026, 2027]

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
    changeYearFrom,
    changeYearTo,
    onChangeYearFrom,
    onChangeYearTo,
    searchCellId,
    onSearchCellId,
}) {

    // Show model selector when predictions or change involves predicted years
    const needsModel = viewMode === 'predictions' ||
        (viewMode === 'change' && (changeYearFrom > 2021 || changeYearTo > 2021));

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

            {/* Labels view: year selector — only ground-truth years */}
            {viewMode === 'labels' && (
                <div className="section">
                    <div className="section-title">Year (Ground Truth)</div>
                    <div className="toggle-group">
                        {labelYears.map((y) => (
                            <button
                                key={y}
                                className={`toggle-btn ${selectedYear === y ? 'active' : ''}`}
                                onClick={() => onYearChange(y)}
                            >
                                {y}
                            </button>
                        ))}
                    </div>
                    <div className="info-badge" style={{ marginTop: 6 }}>
                        ESA WorldCover labels
                    </div>
                </div>
            )}

            {/* Predictions view: year + model selectors */}
            {viewMode === 'predictions' && (
                <div className="section">
                    <div className="section-title">Prediction Year</div>
                    <div className="toggle-group">
                        {PREDICTION_YEARS.map((y) => (
                            <button
                                key={y}
                                className={`toggle-btn ${selectedYear === y ? 'active' : ''}`}
                                onClick={() => onYearChange(y)}
                            >
                                {y}
                            </button>
                        ))}
                    </div>
                    {selectedYear === 2021 && (
                        <div className="info-badge" style={{ marginTop: 6 }}>
                            Out-of-fold predictions (holdout cells)
                        </div>
                    )}
                    {selectedYear > 2021 && (
                        <div className="info-badge" style={{ marginTop: 6 }}>
                            Pipeline predictions &mdash; all 29,946 cells
                        </div>
                    )}
                </div>
            )}

            {/* Change view: from/to year pickers */}
            {viewMode === 'change' && (
                <div className="section">
                    <div className="section-title">Compare Years</div>
                    <div className="change-year-row">
                        <div className="change-year-picker">
                            <label className="change-year-label">From</label>
                            <div className="toggle-group compact">
                                {allYears.map((y) => (
                                    <button
                                        key={y}
                                        className={`toggle-btn mini ${changeYearFrom === y ? 'active' : ''}`}
                                        onClick={() => onChangeYearFrom(y)}
                                        disabled={y === changeYearTo}
                                    >
                                        {y}
                                    </button>
                                ))}
                            </div>
                        </div>
                        <div className="change-year-picker">
                            <label className="change-year-label">To</label>
                            <div className="toggle-group compact">
                                {allYears.map((y) => (
                                    <button
                                        key={y}
                                        className={`toggle-btn mini ${changeYearTo === y ? 'active' : ''}`}
                                        onClick={() => onChangeYearTo(y)}
                                        disabled={y === changeYearFrom}
                                    >
                                        {y}
                                    </button>
                                ))}
                            </div>
                        </div>
                    </div>
                    <div className="info-badge" style={{ marginTop: 6 }}>
                        {changeYearFrom} &rarr; {changeYearTo}
                        {(changeYearFrom > 2021 || changeYearTo > 2021) && (
                            <span> &middot; using {MODEL_DISPLAY[selectedModel] || selectedModel} predictions</span>
                        )}
                    </div>
                </div>
            )}

            {/* Future view: year picker */}
            {viewMode === 'future' && (
                <div className="section">
                    <div className="section-title">Prediction Year</div>
                    <div className="toggle-group">
                        {FUTURE_YEARS.map((y) => (
                            <button
                                key={y}
                                className={`toggle-btn ${selectedYear === y ? 'active' : ''}`}
                                onClick={() => onYearChange(y)}
                            >
                                {y}
                            </button>
                        ))}
                    </div>
                    <div>
                        TODO
                    </div>
                </div>
            )}



            {/* Model Selector — for predictions view or change view with predicted years */}
            {needsModel && (
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



            {/* Class Filter */}
            {(
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
            {(
                <div className="section">
                    {viewMode === 'change' ? (
                        <div className="legend">
                            <div className="section-title">Probability to change</div>
                            <div className="legend-bar diverging" />
                            <div className="legend-labels">
                                <span>0%</span> /* TODO */
                                <span>100%</span>
                            </div>
                        </div>
                    ) : (
                        <div className="legend">
                            <div className="section-title">Purity of majority label</div>
                            <div className="legend-bar" />
                            <div className="legend-labels">
                                <span>0%</span> /* TODO */
                                <span>50%</span>
                                <span>100%</span>
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* Search Cell */ /* TODO */}
            <div className="section">
                <div className="section-title">Coordinate Search</div>
                <input
                    className="select"
                    type="number"
                    min="0"
                    max="29945"
                    placeholder="TODO"
                    value={searchCellId ?? ''}
                    onChange={(e) => {
                        const val = e.target.value;
                        onSearchCellId(val === '' ? null : Number(val));
                    }}
                />
            </div>

            {/* Nuremberg District Selection */}
            <div className="section">
                <div className="section-title">Districts</div>
                <div className="nuremberg-district-selection">
                    {/* TODO: Add real district selection logic here */}
                    <select multiple className="select">
                        <option value="TODO">TODO</option>
                        <option value="north">North</option>
                        <option value="south">South</option>
                        <option value="east">East</option>
                        <option value="west">West</option>
                    </select>
                </div>
            </div>

            {/* Disclaimer */}
            <div className="section">
                <div className="disclaimer">
                    <span className="disclaimer-icon">&#9888;</span>
                    <strong>Caveat:</strong> Labels use ESA WorldCover v100 (2020) vs v200 (2021).
                    Algorithm differences may create apparent change that is not real land-cover change.
                </div>
            </div>
        </aside>
    );
}
