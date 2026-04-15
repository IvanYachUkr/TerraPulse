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
    appMode,
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
    nurembergResolution,
    onResolutionChange,
    nurembergYear,
    onNurembergYearChange,
    nurembergSecondaryYear,
    onNurembergSecondaryYearChange,
    nurembergDataMode,
    onNurembergDataModeChange,
    nurembergMeta,
    selectedDistricts,
    onDistrictChange,
    nurembergDistricts,
    experimentalMetrics,
    nurembergExperimentalView,
    onNurembergExperimentalViewChange,
    nurembergExperimentalModel,
    onNurembergExperimentalModelChange,
    districtStats,
    hoveredDistrict,
    nurembergDataMode_forStats,
    predictionAccuracy,
    changeMetrics,
    nurembergPredModel,
    onNurembergPredModelChange,
}) {
    if (appMode === 'analytical') {
        const labelYearsN = nurembergMeta?.label_years || [2020, 2021];
        const predYearsN = nurembergMeta?.prediction_years || [];
        const visibleYears = nurembergDataMode === 'labels' ? labelYearsN : predYearsN;

        return (
            <aside className="sidebar">
                {/* Data Mode Toggle */}
                <div className="section">
                    <div className="section-title">Data Source</div>
                    <div className="toggle-group">
                        <button
                            className={`toggle-btn ${nurembergDataMode === 'labels' ? 'active' : ''}`}
                            onClick={() => {
                                onNurembergDataModeChange('labels');
                                if (!labelYearsN.includes(nurembergYear)) {
                                    onNurembergYearChange(labelYearsN[labelYearsN.length - 1]);
                                }
                            }}
                        >
                            🏷️ Labels
                        </button>
                        <button
                            className={`toggle-btn ${nurembergDataMode === 'predictions' ? 'active' : ''}`}
                            onClick={() => {
                                onNurembergDataModeChange('predictions');
                                if (!predYearsN.includes(nurembergYear)) {
                                    onNurembergYearChange(predYearsN[predYearsN.length - 1] || 2021);
                                }
                            }}
                            disabled={predYearsN.length === 0}
                        >
                            🤖 Predictions
                        </button>
                        <button
                            className={`toggle-btn ${nurembergDataMode === 'experimental' ? 'active' : ''}`}
                            onClick={() => {
                                onNurembergDataModeChange('experimental');
                                onNurembergYearChange(2021);
                            }}
                        >
                            🧪 Experimental
                        </button>
                    </div>
                    <div className="info-badge" style={{ marginTop: 6 }}>
                        {nurembergDataMode === 'labels'
                            ? 'ESA WorldCover · Ground Truth'
                            : nurembergDataMode === 'experimental'
                                ? 'Experimental · 4-Fold Spatial CV'
                                : nurembergPredModel === 'ssnet_v8'
                                    ? 'Spectral-Spatial Net · Deep Pixel Classifier'
                                    : 'CatBoost · Gradient Boosted Pixel Classifier'}
                    </div>
                </div>

                {/* Year Selection */}
                {nurembergDataMode === 'labels' && (
                    <div className="section">
                        <div className="section-title">Year</div>
                        {visibleYears.length <= 3 ? (
                            <div className="toggle-group">
                                {visibleYears.map((y) => (
                                    <button
                                        key={y}
                                        className={`toggle-btn ${nurembergYear === y ? 'active' : ''}`}
                                        onClick={() => onNurembergYearChange(y)}
                                    >
                                        {y}
                                    </button>
                                ))}
                            </div>
                        ) : (
                            <select
                                className="select"
                                value={nurembergYear}
                                onChange={(e) => onNurembergYearChange(Number(e.target.value))}
                            >
                                {visibleYears.map((y) => (
                                    <option key={y} value={y}>{y}</option>
                                ))}
                            </select>
                        )}
                    </div>
                )}

                {nurembergDataMode === 'predictions' && (
                    <div className="section">
                        <div className="section-title">Select Year | Compare Years</div>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                            <div>
                                <div style={{ fontSize: 10, opacity: 0.5, marginBottom: 4, fontWeight: 600, textTransform: 'uppercase' }}>Year</div>
                                <select
                                    className="select"
                                    value={nurembergYear}
                                    onChange={(e) => onNurembergYearChange(Number(e.target.value))}
                                >
                                    {visibleYears.map((y) => (
                                        <option key={y} value={y}>{y}</option>
                                    ))}
                                </select>
                            </div>
                            <div>
                                <div style={{ fontSize: 10, opacity: 0.5, marginBottom: 4, fontWeight: 600, textTransform: 'uppercase' }}>Compare to</div>
                                <select
                                    className="select"
                                    value={nurembergSecondaryYear ?? 'none'}
                                    onChange={(e) => onNurembergSecondaryYearChange(e.target.value === 'none' ? null : Number(e.target.value))}
                                >
                                    <option value="none">None</option>
                                    {visibleYears
                                        .filter((y) => y !== nurembergYear)
                                        .map((y) => (
                                            <option key={y} value={y}>{y}</option>
                                        ))}
                                </select>
                            </div>
                        </div>
                        {nurembergSecondaryYear !== null && (
                            <div className="info-badge" style={{ marginTop: 8, background: 'rgba(59, 130, 246, 0.15)', borderColor: 'rgba(59, 130, 246, 0.3)' }}>
                                Showing changes: {nurembergYear} &rarr; {nurembergSecondaryYear}
                            </div>
                        )}
                        <div style={{ marginTop: 10 }}>
                            <div style={{ fontSize: 10, opacity: 0.5, marginBottom: 4, fontWeight: 600, textTransform: 'uppercase' }}>Model</div>
                            <div className="toggle-group">
                                <button
                                    className={`toggle-btn ${nurembergPredModel === 'catboost' ? 'active' : ''}`}
                                    onClick={() => onNurembergPredModelChange('catboost')}
                                >
                                    CatBoost
                                </button>
                                <button
                                    className={`toggle-btn ${nurembergPredModel === 'ssnet_v8' ? 'active' : ''}`}
                                    onClick={() => onNurembergPredModelChange('ssnet_v8')}
                                >
                                    Spectral-Spatial Net
                                </button>
                            </div>
                            <div style={{ fontSize: 10, opacity: 0.5, marginTop: 4, lineHeight: 1.4 }}>
                                {nurembergPredModel === 'ssnet_v8'
                                    ? 'Deep neural network using raw spectral bands with 3×3 spatial context'
                                    : 'Gradient boosting with 217 engineered features per pixel'}
                            </div>
                        </div>
                    </div>
                )}

                {/* Prediction Accuracy Card */}
                {nurembergDataMode === 'predictions' && predictionAccuracy && Object.keys(predictionAccuracy).length > 0 && (
                    <div className="section" style={{ borderLeft: '3px solid #10b981', paddingLeft: 12 }}>
                        <div className="section-title">📊 Model Accuracy vs Ground Truth</div>
                        <div style={{
                            background: 'linear-gradient(135deg, rgba(16,185,129,0.1), rgba(59,130,246,0.08))',
                            border: '1px solid rgba(16,185,129,0.25)',
                            borderRadius: 10, padding: '12px 14px',
                            fontSize: 12, color: '#e2e8f0',
                        }}>
                            {Object.entries(predictionAccuracy).map(([year, data]) => (
                                <div key={year} style={{ marginBottom: 10 }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 6 }}>
                                        <span style={{ fontWeight: 600, fontSize: 13 }}>{year}</span>
                                        <span style={{
                                            fontSize: 22, fontWeight: 800,
                                            color: data.accuracy >= 0.8 ? '#10b981' : data.accuracy >= 0.6 ? '#f59e0b' : '#ef4444',
                                            fontFamily: 'monospace',
                                            letterSpacing: '-0.5px',
                                        }}>
                                            {(data.accuracy * 100).toFixed(1)}%
                                        </span>
                                    </div>
                                    <div style={{
                                        height: 6, borderRadius: 3, background: 'rgba(255,255,255,0.08)',
                                        overflow: 'hidden', marginBottom: 8,
                                    }}>
                                        <div style={{
                                            width: `${data.accuracy * 100}%`, height: '100%', borderRadius: 3,
                                            background: data.accuracy >= 0.8
                                                ? 'linear-gradient(90deg, #10b981, #34d399)'
                                                : data.accuracy >= 0.6
                                                    ? 'linear-gradient(90deg, #f59e0b, #fbbf24)'
                                                    : 'linear-gradient(90deg, #ef4444, #f87171)',
                                            transition: 'width 0.5s ease',
                                        }} />
                                    </div>
                                    <div style={{ fontSize: 10, opacity: 0.5, marginBottom: 4 }}>
                                        {data.correct_pixels?.toLocaleString()} / {data.total_pixels?.toLocaleString()} pixels correct
                                    </div>
                                    {data.per_class && (
                                        <div style={{ borderTop: '1px solid rgba(255,255,255,0.06)', paddingTop: 4 }}>
                                            {Object.entries(data.per_class).map(([cls, m]) => (
                                                <div key={cls} style={{
                                                    display: 'flex', justifyContent: 'space-between',
                                                    padding: '1px 0', fontSize: 10, opacity: 0.8,
                                                }}>
                                                    <span>{cls.replace(/_/g, ' ')}</span>
                                                    <span style={{ fontFamily: 'monospace' }}>F1 {(m.f1 * 100).toFixed(1)}%</span>
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* Change-Specific Metrics Card */}
                {nurembergDataMode === 'predictions' && changeMetrics && (
                    <div className="section" style={{ borderLeft: '3px solid #8b5cf6', paddingLeft: 12 }}>
                        <div className="section-title">📈 Change Metrics (2018–2025)</div>
                        <div style={{
                            background: 'linear-gradient(135deg, rgba(139,92,246,0.1), rgba(59,130,246,0.08))',
                            border: '1px solid rgba(139,92,246,0.25)',
                            borderRadius: 10, padding: '12px 14px',
                            fontSize: 12, color: '#e2e8f0',
                        }}>
                            {/* Pixel Stability */}
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 8 }}>
                                <span style={{ fontWeight: 600 }}>Pixel Stability</span>
                                <span style={{
                                    fontSize: 20, fontWeight: 800,
                                    color: changeMetrics.pixel_stability_pct >= 70 ? '#10b981' : '#f59e0b',
                                    fontFamily: 'monospace',
                                }}>
                                    {changeMetrics.pixel_stability_pct}%
                                </span>
                            </div>
                            <div style={{ fontSize: 10, opacity: 0.5, marginBottom: 10 }}>
                                Pixels with same class across all {changeMetrics.years?.length || 8} years
                            </div>

                            {/* Annual Change Rates */}
                            <div style={{ borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: 8, marginBottom: 8 }}>
                                <div style={{ fontWeight: 600, fontSize: 11, marginBottom: 4 }}>Annual Change Rate</div>
                                {changeMetrics.annual_changes && Object.entries(changeMetrics.annual_changes).map(([key, d]) => (
                                    <div key={key} style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0', fontSize: 11 }}>
                                        <span style={{ opacity: 0.7 }}>{d.from} → {d.to}</span>
                                        <span style={{
                                            fontFamily: 'monospace',
                                            color: d.change_rate > 0.08 ? '#f59e0b' : d.change_rate === 0 ? '#64748b' : '#94a3b8',
                                        }}>
                                            {(d.change_rate * 100).toFixed(1)}%
                                        </span>
                                    </div>
                                ))}
                            </div>

                            {/* False Change Validation */}
                            {changeMetrics.false_change && (
                                <div style={{ borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: 8, marginBottom: 8 }}>
                                    <div style={{ fontWeight: 600, fontSize: 11, marginBottom: 4 }}>Change Validation (2020→2021 vs Labels)</div>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0', fontSize: 11 }}>
                                        <span style={{ opacity: 0.7 }}>False Change Rate</span>
                                        <span style={{ fontFamily: 'monospace', color: changeMetrics.false_change.false_change_rate > 0.3 ? '#ef4444' : '#10b981' }}>
                                            {(changeMetrics.false_change.false_change_rate * 100).toFixed(1)}%
                                        </span>
                                    </div>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0', fontSize: 11 }}>
                                        <span style={{ opacity: 0.7 }}>Precision / Recall</span>
                                        <span style={{ fontFamily: 'monospace' }}>
                                            {(changeMetrics.false_change.change_precision * 100).toFixed(1)}% / {(changeMetrics.false_change.change_recall * 100).toFixed(1)}%
                                        </span>
                                    </div>
                                </div>
                            )}

                            {/* Per-Class Stability */}
                            {changeMetrics.per_class_stability && (
                                <div style={{ borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: 8, marginBottom: 8 }}>
                                    <div style={{ fontWeight: 600, fontSize: 11, marginBottom: 4 }}>Per-Class Stability</div>
                                    {Object.entries(changeMetrics.per_class_stability)
                                        .sort((a, b) => b[1].stable_pct - a[1].stable_pct)
                                        .map(([cls, data]) => (
                                        <div key={cls} style={{
                                            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                                            padding: '2px 0', fontSize: 11,
                                        }}>
                                            <span style={{ opacity: 0.8 }}>{cls.replace(/_/g, ' ')}</span>
                                            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                                <div style={{ width: 40, height: 5, borderRadius: 2, background: 'rgba(255,255,255,0.08)', overflow: 'hidden' }}>
                                                    <div style={{
                                                        width: `${data.stable_pct}%`, height: '100%', borderRadius: 2,
                                                        background: data.stable_pct >= 80 ? '#10b981' : data.stable_pct >= 50 ? '#f59e0b' : '#ef4444',
                                                    }} />
                                                </div>
                                                <span style={{ fontFamily: 'monospace', width: 38, textAlign: 'right' }}>{data.stable_pct}%</span>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}

                            {/* Top Transitions */}
                            {changeMetrics.top_transitions && changeMetrics.top_transitions.length > 0 && (
                                <div style={{ borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: 8 }}>
                                    <div style={{ fontWeight: 600, fontSize: 11, marginBottom: 4 }}>Top Transitions</div>
                                    {changeMetrics.top_transitions.slice(0, 5).map((t, i) => (
                                        <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0', fontSize: 10 }}>
                                            <span style={{ opacity: 0.7 }}>
                                                {t.from.replace(/_/g, ' ')} → {t.to.replace(/_/g, ' ')}
                                            </span>
                                            <span style={{ fontFamily: 'monospace', opacity: 0.8 }}>{t.pct}%</span>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>
                )}

                {/* Resolution Slider */}
                <div className="section">
                    <div className="section-title">
                        Resolution: {nurembergResolution * 10}m
                        <span style={{ opacity: 0.5, fontSize: 11, marginLeft: 6 }}>
                            ({nurembergResolution === 1 ? 'pixel' : `${nurembergResolution}×${nurembergResolution}`})
                        </span>
                    </div>
                    <input
                        type="range"
                        min="1"
                        max="10"
                        value={nurembergResolution}
                        onChange={(e) => onResolutionChange(Number(e.target.value))}
                        className="resolution-slider"
                        style={{ width: '100%' }}
                    />
                    <div className="legend-labels" style={{ fontSize: 10, opacity: 0.5 }}>
                        <span>10m (pixel)</span>
                        <span>100m (cell)</span>
                    </div>
                </div>

                {/* Model Stage Selection */}
                {nurembergDataMode === 'experimental' && (
                    <div className="section" style={{ borderLeft: '3px solid #10b981', paddingLeft: 12 }}>
                        <div className="section-title">Model Stage Selection</div>
                        <div className="toggle-group vertical">
                            <button
                                className={`toggle-btn ${nurembergExperimentalView === 'heatmap' ? 'active' : ''}`}
                                onClick={() => onNurembergExperimentalViewChange('heatmap')}
                            >
                                🔥 Change Likelihood
                            </button>
                            <button
                                className={`toggle-btn ${nurembergExperimentalView === 'changes' ? 'active' : ''}`}
                                onClick={() => onNurembergExperimentalViewChange('changes')}
                            >
                                📍 Predicted Changes
                            </button>
                            <button
                                className={`toggle-btn ${nurembergExperimentalView === 'map' ? 'active' : ''}`}
                                onClick={() => onNurembergExperimentalViewChange('map')}
                            >
                                🗺️ Full Prediction Map
                            </button>
                        </div>

                    </div>
                )}

                {/* Class Filter / Model Toggle */}
                <div className="section">
                    {nurembergDataMode === 'experimental' && nurembergExperimentalView === 'heatmap' ? (
                        <>
                            <div className="section-title">Experimental Model Selection</div>
                            <div className="toggle-group">
                                <button
                                    className={`toggle-btn ${nurembergExperimentalModel === 'rf' ? 'active' : ''}`}
                                    onClick={() => onNurembergExperimentalModelChange('rf')}
                                >
                                    🌳 Random Forest
                                </button>
                                <button
                                    className={`toggle-btn ${nurembergExperimentalModel === 'linear' ? 'active' : ''}`}
                                    onClick={() => onNurembergExperimentalModelChange('linear')}
                                >
                                    📈 Explainable (LR)
                                </button>
                            </div>
                            <div style={{ fontSize: 11, opacity: 0.6, marginTop: 8, lineHeight: 1.4 }}>
                                {nurembergExperimentalModel === 'rf'
                                    ? 'Multi-stage RF ensemble trained on spatial folds to minimize leakage.'
                                    : 'Log. Regression on standardized spectral & socioeconomic features for max transparency.'}
                            </div>
                        </>
                    ) : (
                        <>
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
                        </>
                    )}
                </div>

                {/* Nuremberg legend */}
                <div className="section">
                    <div className="section-title">Legend</div>
                    <div className="legend">
                        {classes.map((c) => {
                            const [r, g, b] = classColors[c];
                            return (
                                <div key={c} style={{
                                    display: 'flex', alignItems: 'center',
                                    gap: 8, padding: '2px 0', fontSize: 12,
                                }}>
                                    <span style={{
                                        width: 14, height: 14, borderRadius: 3,
                                        backgroundColor: `rgb(${r},${g},${b})`,
                                        flexShrink: 0,
                                    }} />
                                    <span style={{ color: '#e2e8f0' }}>{classLabels[c]}</span>
                                </div>
                            );
                        })}
                    </div>
                </div>

                {/* Experimental metrics panel */}
                {nurembergDataMode === 'experimental' && experimentalMetrics && (
                    <div className="section">
                        <div className="section-title">Model Accuracy</div>
                        <div style={{
                            background: nurembergExperimentalModel === 'rf' ? 'rgba(16, 185, 129, 0.1)' : 'rgba(59, 130, 246, 0.1)',
                            border: `1px solid ${nurembergExperimentalModel === 'rf' ? 'rgba(16, 185, 129, 0.3)' : 'rgba(59, 130, 246, 0.3)'}`,
                            borderRadius: 8, padding: '10px 12px',
                            fontSize: 12, color: '#e2e8f0',
                        }}>
                            <div style={{ fontSize: 18, fontWeight: 700, color: nurembergExperimentalModel === 'rf' ? '#10b981' : '#3b82f6', marginBottom: 6 }}>
                                {experimentalMetrics.overall_accuracy !== undefined
                                    ? `${(experimentalMetrics.overall_accuracy * 100).toFixed(1)}% Overall`
                                    : 'Calculating...'}
                            </div>
                            <div style={{ fontSize: 11, opacity: 0.7, marginBottom: 8 }}>
                                {nurembergExperimentalModel === 'rf'
                                    ? `${experimentalMetrics.n_folds || 4}-fold spatial CV · ${experimentalMetrics.model || 'Random Forest'}`
                                    : `${experimentalMetrics.model_info || 'Explainable Model'} (${experimentalMetrics.training_info || 'Loading...'})`}
                            </div>

                            {nurembergExperimentalModel === 'rf' ? (
                                <>
                                    <div style={{ marginTop: 8, borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: 6, marginBottom: 8 }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0' }}>
                                            <span>Macro Precision</span>
                                            <span style={{ fontFamily: 'monospace' }}>
                                                {(Object.values(experimentalMetrics.per_class || {}).reduce((acc, m) => acc + (m.precision || 0), 0) /
                                                    Object.keys(experimentalMetrics.per_class || {}).length).toFixed(3)}
                                            </span>
                                        </div>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0' }}>
                                            <span>Macro Recall</span>
                                            <span style={{ fontFamily: 'monospace' }}>
                                                {(Object.values(experimentalMetrics.per_class || {}).reduce((acc, m) => acc + (m.recall || 0), 0) /
                                                    Object.keys(experimentalMetrics.per_class || {}).length).toFixed(3)}
                                            </span>
                                        </div>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0' }}>
                                            <span>Macro F1-Score</span>
                                            <span style={{ fontFamily: 'monospace' }}>
                                                {(Object.values(experimentalMetrics.per_class || {}).reduce((acc, m) => acc + (m.f1 || 0), 0) /
                                                    Object.keys(experimentalMetrics.per_class || {}).length).toFixed(3)}
                                            </span>
                                        </div>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0', borderTop: '1px solid rgba(255,255,255,0.05)', marginTop: 4, paddingTop: 4 }}>
                                            <span>False Change Rate</span>
                                            <span style={{ fontFamily: 'monospace', opacity: experimentalMetrics.false_change_rate !== undefined ? 1 : 0.5 }}>
                                                {experimentalMetrics.false_change_rate !== undefined
                                                    ? `${(experimentalMetrics.false_change_rate * 100).toFixed(2)}%`
                                                    : 'N/A (Multiclass)'}
                                            </span>
                                        </div>
                                    </div>

                                    {experimentalMetrics.fold_metrics && (
                                        <div style={{ marginBottom: 10 }}>
                                            <div style={{ fontWeight: 600, fontSize: 10, opacity: 0.6, marginBottom: 4 }}>Per-Fold Accuracy</div>
                                            {experimentalMetrics.fold_metrics.map((fm) => (
                                                <div key={fm.fold} style={{ display: 'flex', justifyContent: 'space-between', padding: '1px 0', fontSize: 11 }}>
                                                    <span>Fold {fm.fold}</span>
                                                    <span style={{ fontFamily: 'monospace' }}>{(fm.accuracy * 100).toFixed(1)}%</span>
                                                </div>
                                            ))}
                                        </div>
                                    )}

                                    <div style={{ fontWeight: 600, fontSize: 10, opacity: 0.6, marginBottom: 4 }}>Per-Class F1 Scores</div>
                                    {experimentalMetrics.per_class && Object.entries(experimentalMetrics.per_class).map(([cls, m]) => (
                                        <div key={cls} style={{
                                            display: 'flex', justifyContent: 'space-between',
                                            padding: '2px 0', borderTop: '1px solid rgba(255,255,255,0.05)',
                                        }}>
                                            <span>{cls.replace('_', ' ')}</span>
                                            <span style={{ fontFamily: 'monospace' }}>F1={m.f1?.toFixed(3) || '0.000'}</span>
                                        </div>
                                    ))}
                                </>
                            ) : (
                                <div style={{ marginTop: 8, borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: 6 }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0' }}>
                                        <span>F1-Score</span>
                                        <span style={{ fontFamily: 'monospace' }}>{experimentalMetrics.macro_f1?.toFixed(3) || '0.000'}</span>
                                    </div>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0' }}>
                                        <span>Precision</span>
                                        <span style={{ fontFamily: 'monospace' }}>{experimentalMetrics.precision?.toFixed(3) || '0.000'}</span>
                                    </div>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0' }}>
                                        <span>Recall</span>
                                        <span style={{ fontFamily: 'monospace' }}>{experimentalMetrics.recall?.toFixed(3) || '0.000'}</span>
                                    </div>
                                    {experimentalMetrics.false_change_rate !== undefined && (
                                        <div style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0', borderTop: '1px solid rgba(255,255,255,0.05)', marginTop: 4, paddingTop: 4 }}>
                                            <span>False Change Rate</span>
                                            <span style={{ fontFamily: 'monospace' }}>{(experimentalMetrics.false_change_rate * 100).toFixed(1)}%</span>
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    </div>
                )}



                {/* District Selection & Stats */}
                {districtStats && (() => {
                    // Build list of districts from the stats
                    const districtEntries = Object.entries(districtStats)
                        .sort((a, b) => a[1].id.localeCompare(b[1].id, undefined, { numeric: true }));

                    // Determine data key for stats
                    let dataKey = 'labels_2021';
                    if (nurembergDataMode_forStats === 'labels') {
                        dataKey = `labels_${nurembergYear}`;
                    } else if (nurembergDataMode_forStats === 'predictions') {
                        dataKey = 'predictions_2021';
                    } else if (nurembergDataMode_forStats === 'experimental') {
                        dataKey = 'experimental_2021';
                    }

                    // Aggregate stats for selected districts
                    const aggStats = {};
                    let aggTotal = 0;
                    const CLASS_ORDER_LOCAL = ['tree_cover', 'grassland', 'cropland', 'built_up', 'bare_sparse', 'water'];
                    CLASS_ORDER_LOCAL.forEach(c => aggStats[c] = 0);
                    for (const id of selectedDistricts) {
                        const d = districtStats[id];
                        if (!d || !d[dataKey]) continue;
                        aggTotal += d.total_pixels || 0;
                        CLASS_ORDER_LOCAL.forEach(c => aggStats[c] += (d[dataKey][c] || 0));
                    }

                    return (<>
                        <div className="section">
                            <div className="section-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <span>Districts ({selectedDistricts.length})</span>
                                <div style={{ display: 'flex', gap: 4 }}>
                                    {selectedDistricts.length > 0 && (
                                        <button
                                            onClick={() => onDistrictChange('__clear__')}
                                            style={{
                                                background: 'rgba(239,68,68,0.2)', border: '1px solid rgba(239,68,68,0.4)',
                                                color: '#fca5a5', borderRadius: 4, padding: '2px 8px',
                                                fontSize: 10, cursor: 'pointer',
                                            }}
                                        >
                                            Clear
                                        </button>
                                    )}
                                </div>
                            </div>
                            <div style={{
                                maxHeight: 200, overflowY: 'auto', border: '1px solid rgba(255,255,255,0.1)',
                                borderRadius: 8, padding: 4, background: 'rgba(15,23,42,0.5)',
                            }}>
                                {districtEntries.map(([id, d]) => {
                                    const isSelected = selectedDistricts.includes(id);
                                    const isHovered = hoveredDistrict === id;
                                    return (
                                        <div
                                            key={id}
                                            onClick={() => onDistrictChange(id)}
                                            style={{
                                                display: 'flex', alignItems: 'center', gap: 6,
                                                padding: '4px 8px', cursor: 'pointer', borderRadius: 4,
                                                fontSize: 11, color: isSelected ? '#93c5fd' : '#e2e8f0',
                                                background: isHovered ? 'rgba(255,255,255,0.08)'
                                                    : isSelected ? 'rgba(59,130,246,0.12)' : 'transparent',
                                                transition: 'background 0.15s',
                                            }}
                                        >
                                            <span style={{
                                                width: 14, height: 14, borderRadius: 3, flexShrink: 0,
                                                border: isSelected ? '2px solid #3b82f6' : '1px solid rgba(255,255,255,0.2)',
                                                background: isSelected ? '#3b82f6' : 'transparent',
                                                display: 'flex', alignItems: 'center', justifyContent: 'center',
                                                fontSize: 9, color: '#fff',
                                            }}>
                                                {isSelected ? '✓' : ''}
                                            </span>
                                            <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                                {d.id} {d.name}
                                            </span>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>

                        {/* Stats panel for selected districts */}
                        {selectedDistricts.length > 0 && aggTotal > 0 && (
                            <div className="section">
                                <div className="section-title">
                                    District Stats
                                    <span style={{ fontSize: 10, opacity: 0.5, marginLeft: 6 }}>
                                        ({selectedDistricts.length} selected · {aggTotal.toLocaleString()} px)
                                    </span>
                                </div>
                                <div style={{
                                    background: 'rgba(59, 130, 246, 0.08)',
                                    border: '1px solid rgba(59, 130, 246, 0.2)',
                                    borderRadius: 8, padding: '8px 10px',
                                }}>
                                    {CLASS_ORDER_LOCAL.map(c => {
                                        const count = aggStats[c];
                                        const pct = aggTotal > 0 ? (count / aggTotal * 100) : 0;
                                        const [r, g, b] = classColors[c];
                                        return (
                                            <div key={c} style={{
                                                display: 'flex', alignItems: 'center', gap: 6,
                                                padding: '3px 0', fontSize: 11,
                                            }}>
                                                <span style={{
                                                    width: 10, height: 10, borderRadius: 2,
                                                    background: `rgb(${r},${g},${b})`, flexShrink: 0,
                                                }} />
                                                <span style={{ flex: 1, color: '#e2e8f0' }}>{classLabels[c]}</span>
                                                <div style={{
                                                    width: 60, height: 6, borderRadius: 3,
                                                    background: 'rgba(255,255,255,0.08)', overflow: 'hidden',
                                                }}>
                                                    <div style={{
                                                        width: `${pct}%`, height: '100%', borderRadius: 3,
                                                        background: `rgb(${r},${g},${b})`,
                                                        transition: 'width 0.3s ease',
                                                    }} />
                                                </div>
                                                <span style={{
                                                    fontFamily: 'monospace', fontSize: 10, color: '#94a3b8',
                                                    width: 36, textAlign: 'right',
                                                }}>
                                                    {pct.toFixed(1)}%
                                                </span>
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>
                        )}
                    </>);
                })()}

                <div className="section">
                    <div className="disclaimer">
                        <span className="disclaimer-icon">&#9888;</span>
                        <strong>Note:</strong> Shrubland is remapped to grassland
                        for Nuremberg. WorldCover v100 (2020) vs v200 (2021) may
                        show algorithm differences.
                    </div>
                </div>
            </aside>
        );
    }

    // Original sidebar for other modes (unused for now but kept for compatibility)
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

            {/* Model Selector */}
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
                                <span>0%</span> {/* TODO */}
                                <span>100%</span>
                            </div>
                        </div>
                    ) : (
                        <div className="legend">
                            <div className="section-title">Purity of majority label</div>
                            <div className="legend-bar" />
                            <div className="legend-labels">
                                <span>0%</span> {/* TODO */}
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
                    <select multiple
                        className="select"
                        value={selectedDistricts}
                        onChange={onDistrictChange}
                    >
                        <option>TODO - GEOJSON in Map Component</option>
                        <option key="All" value="All">All</option>
                        {nurembergDistricts.map(d => (
                            <option key={d} value={d}>
                                {d}
                            </option>
                        ))}
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
