import { useEffect, useRef, useState } from 'react';
import { Chart, registerables } from 'chart.js';

Chart.register(...registerables);

const CLASS_LABELS = {
    tree_cover: 'Tree Cover',
    shrubland: 'Shrubland',
    grassland: 'Grassland',
    cropland: 'Cropland',
    built_up: 'Built-up',
    bare_sparse: 'Bare/Sparse',
    water: 'Water',
};

const CLASS_COLORS_HEX = {
    tree_cover: '#2d6a4f',
    shrubland: '#6a994e',
    grassland: '#95d5b2',
    cropland: '#f4a261',
    built_up: '#e76f51',
    bare_sparse: '#d4a373',
    water: '#0096c7',
};

const MODEL_COLORS = {
    MLP: { bg: 'rgba(236,72,153,0.6)', border: 'rgb(236,72,153)' },
    LightGBM: { bg: 'rgba(16,185,129,0.6)', border: 'rgb(16,185,129)' },
    Ridge: { bg: 'rgba(59,130,246,0.6)', border: 'rgb(59,130,246)' },
};

const TABS = [
    { key: 'metrics', label: 'Per-Class' },
    { key: 'stress', label: 'Stress Tests' },
    { key: 'change', label: 'Change Det.' },
    { key: 'failure', label: 'Failure' },
];

const DARK_CHART_OPTS = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: {
            labels: { color: '#94a3b8', font: { size: 10, family: 'Inter' }, boxWidth: 12 },
        },
    },
    scales: {
        x: {
            ticks: { color: '#94a3b8', font: { size: 10 } },
            grid: { color: 'rgba(255,255,255,0.04)' },
        },
        y: {
            ticks: { color: '#94a3b8', font: { size: 10 } },
            grid: { color: 'rgba(255,255,255,0.04)' },
        },
    },
};

function useChart(ref, chartRef, config) {
    useEffect(() => {
        if (!ref.current || !config) return;
        if (chartRef.current) chartRef.current.destroy();
        chartRef.current = new Chart(ref.current, config);
        return () => {
            if (chartRef.current) { chartRef.current.destroy(); chartRef.current = null; }
        };
    }, [config]);
}

// ── Per-class R² chart ──
function PerClassChart({ evaluation }) {
    const canvasRef = useRef(null);
    const chartRef = useRef(null);
    const classes = ['tree_cover', 'grassland', 'cropland', 'built_up', 'bare_sparse', 'water'];
    const labels = classes.map((c) => CLASS_LABELS[c]);

    const config = evaluation ? {
        type: 'bar',
        data: {
            labels,
            datasets: ['MLP', 'LightGBM', 'Ridge'].map((model) => ({
                label: model,
                data: classes.map((c) => {
                    const row = evaluation.per_class.find((r) => r.model === model && r.class === c);
                    return row ? row.r2 : 0;
                }),
                backgroundColor: MODEL_COLORS[model].bg,
                borderColor: MODEL_COLORS[model].border,
                borderWidth: 1,
            })),
        },
        options: {
            ...DARK_CHART_OPTS,
            scales: {
                ...DARK_CHART_OPTS.scales,
                y: { ...DARK_CHART_OPTS.scales.y, min: 0, max: 1, title: { display: true, text: 'R²', color: '#64748b', font: { size: 10 } } },
                x: { ...DARK_CHART_OPTS.scales.x, ticks: { ...DARK_CHART_OPTS.scales.x.ticks, maxRotation: 30 } },
            },
        },
    } : null;

    useChart(canvasRef, chartRef, config);
    return <canvas ref={canvasRef} />;
}

// ── Noise robustness chart ──
function NoiseChart({ stressTests }) {
    const canvasRef = useRef(null);
    const chartRef = useRef(null);

    const noise = stressTests?.noise;
    // Separate by model
    const mlpNoise = noise?.filter((r) => r.model === 'MLP');
    const treeNoise = noise?.filter((r) => r.model === 'LightGBM');
    const sigmas = mlpNoise?.map((r) => r.noise_sigma.toString());

    const config = mlpNoise ? {
        type: 'line',
        data: {
            labels: sigmas,
            datasets: [
                {
                    label: 'MLP R²',
                    data: mlpNoise.map((r) => r.r2),
                    borderColor: MODEL_COLORS.MLP.border,
                    backgroundColor: 'rgba(236,72,153,0.1)',
                    fill: false,
                    tension: 0.3,
                    pointRadius: 5,
                },
                {
                    label: 'LightGBM R²',
                    data: treeNoise.map((r) => r.r2),
                    borderColor: MODEL_COLORS.LightGBM.border,
                    backgroundColor: 'rgba(16,185,129,0.1)',
                    fill: false,
                    tension: 0.3,
                    pointRadius: 5,
                    borderDash: [5, 5],
                },
                {
                    label: 'MLP MAE',
                    data: mlpNoise.map((r) => r.mae_pp),
                    borderColor: 'rgba(236,72,153,0.5)',
                    fill: false,
                    tension: 0.3,
                    pointRadius: 3,
                    yAxisID: 'y1',
                },
                {
                    label: 'LightGBM MAE',
                    data: treeNoise.map((r) => r.mae_pp),
                    borderColor: 'rgba(16,185,129,0.5)',
                    fill: false,
                    tension: 0.3,
                    pointRadius: 3,
                    borderDash: [5, 5],
                    yAxisID: 'y1',
                },
            ],
        },
        options: {
            ...DARK_CHART_OPTS,
            scales: {
                x: { ...DARK_CHART_OPTS.scales.x, title: { display: true, text: 'Noise σ (× feature std)', color: '#64748b', font: { size: 10 } } },
                y: { ...DARK_CHART_OPTS.scales.y, title: { display: true, text: 'R²', color: '#64748b', font: { size: 10 } }, position: 'left' },
                y1: { ...DARK_CHART_OPTS.scales.y, title: { display: true, text: 'MAE (pp)', color: '#64748b', font: { size: 10 } }, position: 'right', grid: { drawOnChartArea: false } },
            },
        },
    } : null;

    useChart(canvasRef, chartRef, config);
    return <canvas ref={canvasRef} />;
}

// ── Season/Feature ablation chart ──
function AblationChart({ stressTests }) {
    const canvasRef = useRef(null);
    const chartRef = useRef(null);
    const [ablModel, setAblModel] = useState('MLP');

    const season = stressTests?.season_dropout?.filter((r) => r.model === ablModel && r.season_dropped !== 'none');
    const feature = stressTests?.feature_ablation?.filter((r) => r.model === ablModel && r.group_dropped !== 'none');
    const baseR2 = stressTests?.season_dropout?.find((r) => r.model === ablModel && r.season_dropped === 'none')?.r2 || 0;

    const allItems = [
        ...(season || []).map((r) => ({ label: r.season_dropped.replace('_', ' '), delta: r.r2 - baseR2, type: 'season' })),
        ...(feature || []).map((r) => ({ label: `${r.group_dropped} (${r.n_zeroed}f)`, delta: r.r2 - baseR2, type: 'feature' })),
    ].sort((a, b) => a.delta - b.delta);

    const modelColor = ablModel === 'MLP' ? MODEL_COLORS.MLP.border : MODEL_COLORS.LightGBM.border;

    const config = allItems.length ? {
        type: 'bar',
        data: {
            labels: allItems.map((r) => r.label),
            datasets: [{
                label: `${ablModel} R² change`,
                data: allItems.map((r) => r.delta),
                backgroundColor: allItems.map((r) =>
                    r.delta < -0.3 ? 'rgba(239,68,68,0.7)' : r.delta < -0.1 ? 'rgba(245,158,11,0.7)' : 'rgba(16,185,129,0.7)'
                ),
                borderWidth: 0,
            }],
        },
        options: {
            ...DARK_CHART_OPTS,
            indexAxis: 'y',
            plugins: { ...DARK_CHART_OPTS.plugins, legend: { display: false } },
            scales: {
                x: { ...DARK_CHART_OPTS.scales.x, title: { display: true, text: `${ablModel} R² Δ from baseline (${baseR2.toFixed(4)})`, color: '#64748b', font: { size: 10 } } },
                y: { ...DARK_CHART_OPTS.scales.y, ticks: { color: '#f0f4f8', font: { size: 10 } } },
            },
        },
    } : null;

    useChart(canvasRef, chartRef, config);
    return (
        <>
            <div className="toggle-group" style={{ marginBottom: 6 }}>
                <button className={`toggle-btn ${ablModel === 'MLP' ? 'active' : ''}`} onClick={() => setAblModel('MLP')}>MLP</button>
                <button className={`toggle-btn ${ablModel === 'LightGBM' ? 'active' : ''}`} onClick={() => setAblModel('LightGBM')}>LightGBM</button>
            </div>
            <canvas ref={canvasRef} />
        </>
    );
}

// ── Failure by land cover chart ──
function FailureChart({ failureAnalysis }) {
    const canvasRef = useRef(null);
    const chartRef = useRef(null);

    // Group by model — show both MLP and LightGBM side by side
    const models = ['MLP', 'LightGBM'];
    const classOrder = ['tree_cover', 'grassland', 'cropland', 'built_up', 'bare_sparse', 'water'];
    const labels = classOrder.map((c) => CLASS_LABELS[c] || c);

    const config = failureAnalysis ? {
        type: 'bar',
        data: {
            labels,
            datasets: models.map((model) => {
                const modelData = failureAnalysis.filter((r) => r.model === model);
                return {
                    label: `${model} MAE`,
                    data: classOrder.map((c) => {
                        const row = modelData.find((r) => r.dominant_class === c);
                        return row ? row.mae_pp : 0;
                    }),
                    backgroundColor: model === 'MLP' ? MODEL_COLORS.MLP.bg : MODEL_COLORS.LightGBM.bg,
                    borderColor: model === 'MLP' ? MODEL_COLORS.MLP.border : MODEL_COLORS.LightGBM.border,
                    borderWidth: 1,
                };
            }),
        },
        options: {
            ...DARK_CHART_OPTS,
            scales: {
                x: { ...DARK_CHART_OPTS.scales.x, ticks: { ...DARK_CHART_OPTS.scales.x.ticks, maxRotation: 30 } },
                y: { ...DARK_CHART_OPTS.scales.y, title: { display: true, text: 'MAE (pp)', color: '#64748b', font: { size: 10 } } },
            },
        },
    } : null;

    useChart(canvasRef, chartRef, config);
    return <canvas ref={canvasRef} />;
}

// ── Change detection tradeoff chart ──
function ChangeDetectionChart({ evaluation }) {
    const canvasRef = useRef(null);
    const chartRef = useRef(null);

    const changeData = evaluation?.change_detection;
    if (!changeData || !changeData.length) return <div className="info-badge">No change detection data</div>;

    const config = {
        type: 'line',
        data: {
            datasets: ['MLP', 'LightGBM', 'Ridge'].map((model) => {
                const rows = changeData.filter((r) => r.model === model).sort((a, b) => a.threshold - b.threshold);
                return {
                    label: model,
                    data: rows.map((r) => ({ x: r.false_change_pct, y: r.missed_change_pct })),
                    borderColor: MODEL_COLORS[model].border,
                    backgroundColor: MODEL_COLORS[model].bg,
                    pointRadius: 5,
                    tension: 0.3,
                    showLine: true,
                };
            }),
        },
        options: {
            ...DARK_CHART_OPTS,
            scales: {
                x: { ...DARK_CHART_OPTS.scales.x, type: 'linear', title: { display: true, text: 'False Change Rate (%)', color: '#64748b', font: { size: 10 } } },
                y: { ...DARK_CHART_OPTS.scales.y, title: { display: true, text: 'Missed Change Rate (%)', color: '#64748b', font: { size: 10 } } },
            },
        },
    };

    useChart(canvasRef, chartRef, config);
    return <canvas ref={canvasRef} />;
}

// ── Permutation Importance chart ──
function PermutationChart({ explainability }) {
    const canvasRef = useRef(null);
    const chartRef = useRef(null);

    const mlpPerm = explainability?.models?.mlp?.permutation?.slice(0, 15) || [];
    const treePerm = explainability?.models?.tree?.permutation?.slice(0, 15) || [];

    const allFeatures = [...new Set([...mlpPerm.map(f => f.feature), ...treePerm.map(f => f.feature)])];
    const topFeatures = allFeatures.slice(0, 15);
    const labels = topFeatures.map(f => f.replace(/_/g, ' ').substring(0, 28));

    const config = topFeatures.length ? {
        type: 'bar',
        data: {
            labels,
            datasets: [
                {
                    label: 'MLP',
                    data: topFeatures.map(f => { const it = mlpPerm.find(p => p.feature === f); return it ? it.importance : 0; }),
                    backgroundColor: MODEL_COLORS.MLP.bg, borderColor: MODEL_COLORS.MLP.border, borderWidth: 1,
                },
                {
                    label: 'LightGBM',
                    data: topFeatures.map(f => { const it = treePerm.find(p => p.feature === f); return it ? it.importance : 0; }),
                    backgroundColor: MODEL_COLORS.LightGBM.bg, borderColor: MODEL_COLORS.LightGBM.border, borderWidth: 1,
                },
            ],
        },
        options: {
            ...DARK_CHART_OPTS, indexAxis: 'y',
            scales: {
                x: { ...DARK_CHART_OPTS.scales.x, title: { display: true, text: 'R² decrease', color: '#64748b', font: { size: 10 } } },
                y: { ...DARK_CHART_OPTS.scales.y, ticks: { color: '#f0f4f8', font: { size: 9 } } },
            },
        },
    } : null;

    useChart(canvasRef, chartRef, config);
    return <canvas ref={canvasRef} />;
}

// ── SHAP per-class importance chart ──
function ShapChart({ explainability }) {
    const canvasRef = useRef(null);
    const chartRef = useRef(null);
    const [shapModel, setShapModel] = useState('mlp');

    const classKeys = ['tree_cover', 'shrubland', 'grassland', 'cropland', 'built_up', 'bare_sparse', 'water'];
    const classColorsList = [
        { bg: '#2d6a4faa', border: '#2d6a4f' },
        { bg: '#6a994eaa', border: '#6a994e' },
        { bg: '#95d5b2aa', border: '#95d5b2' },
        { bg: '#f4a261aa', border: '#f4a261' },
        { bg: '#e76f51aa', border: '#e76f51' },
        { bg: '#d4a373aa', border: '#d4a373' },
        { bg: '#0096c7aa', border: '#0096c7' },
    ];

    const shapData = explainability?.models?.[shapModel]?.shap?.slice(0, 12) || [];
    const labels = shapData.map(f => f.feature.replace(/_/g, ' ').substring(0, 28));

    const config = shapData.length ? {
        type: 'bar',
        data: {
            labels,
            datasets: classKeys.map((cls, i) => ({
                label: CLASS_LABELS[cls],
                data: shapData.map(f => f[cls] || 0),
                backgroundColor: classColorsList[i].bg,
                borderColor: classColorsList[i].border,
                borderWidth: 1,
            })),
        },
        options: {
            ...DARK_CHART_OPTS, indexAxis: 'y',
            scales: {
                x: { ...DARK_CHART_OPTS.scales.x, stacked: true, title: { display: true, text: 'Mean |SHAP|', color: '#64748b', font: { size: 10 } } },
                y: { ...DARK_CHART_OPTS.scales.y, stacked: true, ticks: { color: '#f0f4f8', font: { size: 9 } } },
            },
            plugins: { ...DARK_CHART_OPTS.plugins, legend: { ...DARK_CHART_OPTS.plugins.legend, labels: { ...DARK_CHART_OPTS.plugins.legend.labels, boxWidth: 8 } } },
        },
    } : null;

    useChart(canvasRef, chartRef, config);

    return (
        <>
            <div className="toggle-group" style={{ marginBottom: 6 }}>
                <button className={`toggle-btn ${shapModel === 'mlp' ? 'active' : ''}`} onClick={() => setShapModel('mlp')}>MLP</button>
                <button className={`toggle-btn ${shapModel === 'tree' ? 'active' : ''}`} onClick={() => setShapModel('tree')}>LightGBM</button>
            </div>
            <canvas ref={canvasRef} />
        </>
    );
}

// ── Explanation cards ──
// ── SHAP Deep Dive (beeswarm + dependence plots) ──
function ShapDeepDive() {
    const [manifest, setManifest] = useState(null);
    const [activeModel, setActiveModel] = useState('mlp');
    const [activeClass, setActiveClass] = useState('tree_cover');
    const [viewMode, setViewMode] = useState('beeswarm'); // 'beeswarm' | 'dependence'
    const [activeDep, setActiveDep] = useState(0);

    useEffect(() => {
        fetch('/api/shap-plots/manifest')
            .then(r => r.json())
            .then(d => setManifest(d))
            .catch(() => { });
    }, []);

    if (!manifest) return <div style={{ color: '#94a3b8', padding: 12, fontSize: 12 }}>Loading SHAP deep-dive plots...</div>;

    const modelData = manifest[activeModel];
    if (!modelData) return null;

    const classInfo = modelData.classes?.find(c => c.class === activeClass);
    const depPlots = classInfo?.dependence || [];

    const CLASS_LABELS_MAP = {
        tree_cover: 'Tree Cover', grassland: 'Grassland', cropland: 'Cropland',
        built_up: 'Built-up', bare_sparse: 'Bare/Sparse', water: 'Water',
    };

    return (
        <div>
            {/* Model toggle */}
            <div style={{ display: 'flex', gap: 6, marginBottom: 8 }}>
                <button className={`toggle-btn ${activeModel === 'mlp' ? 'active' : ''}`}
                    onClick={() => { setActiveModel('mlp'); setActiveDep(0); }}>MLP</button>
                <button className={`toggle-btn ${activeModel === 'tree' ? 'active' : ''}`}
                    onClick={() => { setActiveModel('tree'); setActiveDep(0); }}>LightGBM</button>
                <div style={{ flex: 1 }} />
                <button className={`toggle-btn ${viewMode === 'beeswarm' ? 'active' : ''}`}
                    onClick={() => setViewMode('beeswarm')}
                    style={{ fontSize: 10 }}>Beeswarm</button>
                <button className={`toggle-btn ${viewMode === 'dependence' ? 'active' : ''}`}
                    onClick={() => setViewMode('dependence')}
                    style={{ fontSize: 10 }}>Dependence</button>
            </div>

            {/* Class tabs */}
            <div style={{ display: 'flex', gap: 4, marginBottom: 10, flexWrap: 'wrap' }}>
                {modelData.classes?.map(c => (
                    <button key={c.class}
                        className={`toggle-btn ${activeClass === c.class ? 'active' : ''}`}
                        onClick={() => { setActiveClass(c.class); setActiveDep(0); }}
                        style={{ fontSize: 10, padding: '3px 8px' }}>
                        {CLASS_LABELS_MAP[c.class] || c.class}
                    </button>
                ))}
            </div>

            {/* Plot display */}
            {viewMode === 'beeswarm' && classInfo && (
                <div style={{ textAlign: 'center' }}>
                    <img
                        src={`/api/shap-plots/${classInfo.beeswarm}`}
                        alt={`SHAP Beeswarm - ${activeModel} - ${activeClass}`}
                        style={{ maxWidth: '100%', borderRadius: 6, border: '1px solid #334155' }}
                    />
                    <div style={{ color: '#94a3b8', fontSize: 10, marginTop: 6 }}>
                        Each dot = one sample. X-axis = SHAP impact on prediction. Color = feature value (blue=low, red=high).
                    </div>
                </div>
            )}

            {viewMode === 'dependence' && classInfo && (
                <div>
                    {/* Feature selector for dependence plots */}
                    <div style={{ display: 'flex', gap: 4, marginBottom: 8, flexWrap: 'wrap' }}>
                        {depPlots.map((d, i) => {
                            const shortFeat = d.feature.length > 22
                                ? d.feature.substring(0, 21) + '.'
                                : d.feature;
                            return (
                                <button key={i}
                                    className={`toggle-btn ${activeDep === i ? 'active' : ''}`}
                                    onClick={() => setActiveDep(i)}
                                    style={{ fontSize: 9, padding: '2px 6px' }}>
                                    {shortFeat}
                                </button>
                            );
                        })}
                    </div>
                    {depPlots[activeDep] && (
                        <div style={{ textAlign: 'center' }}>
                            <img
                                src={`/api/shap-plots/${depPlots[activeDep].file}`}
                                alt={`SHAP Dependence - ${depPlots[activeDep].feature}`}
                                style={{ maxWidth: '100%', borderRadius: 6, border: '1px solid #334155' }}
                            />
                            <div style={{ color: '#94a3b8', fontSize: 10, marginTop: 6 }}>
                                X-axis = feature value. Y-axis = SHAP impact. Color = interaction feature value.
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}


function ExplanationCards({ explainability }) {
    const mlp = explainability?.models?.mlp || {};
    const tree = explainability?.models?.tree || {};

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {mlp.helpful && (
                <div style={{ padding: '8px 10px', borderRadius: 6, background: 'rgba(16,185,129,0.12)', border: '1px solid rgba(16,185,129,0.3)' }}>
                    <div style={{ color: '#10b981', fontWeight: 600, fontSize: 11, marginBottom: 4 }}>✓ Helpful — MLP</div>
                    <div style={{ color: '#e2e8f0', fontSize: 11 }}>{mlp.helpful.explanation}</div>
                    <div style={{ color: '#94a3b8', fontSize: 10, marginTop: 2 }}>{mlp.helpful.rationale}</div>
                </div>
            )}
            {mlp.misleading && (
                <div style={{ padding: '8px 10px', borderRadius: 6, background: 'rgba(245,158,11,0.12)', border: '1px solid rgba(245,158,11,0.3)' }}>
                    <div style={{ color: '#f59e0b', fontWeight: 600, fontSize: 11, marginBottom: 4 }}>⚠ Misleading — MLP</div>
                    <div style={{ color: '#e2e8f0', fontSize: 11 }}>{mlp.misleading.explanation}</div>
                    <div style={{ color: '#94a3b8', fontSize: 10, marginTop: 2 }}>{mlp.misleading.pitfall || mlp.misleading.note}</div>
                </div>
            )}
            {tree.helpful && (
                <div style={{ padding: '8px 10px', borderRadius: 6, background: 'rgba(16,185,129,0.12)', border: '1px solid rgba(16,185,129,0.3)' }}>
                    <div style={{ color: '#10b981', fontWeight: 600, fontSize: 11, marginBottom: 4 }}>✓ Helpful — LightGBM</div>
                    <div style={{ color: '#e2e8f0', fontSize: 11 }}>{tree.helpful.explanation}</div>
                    <div style={{ color: '#94a3b8', fontSize: 10, marginTop: 2 }}>{tree.helpful.rationale}</div>
                </div>
            )}
        </div>
    );
}

// ── Main component ──
export default function EvaluationPanel({ evaluation, stressTests, failureAnalysis, onClose }) {
    const [activeTab, setActiveTab] = useState('metrics');

    return (
        <div className="evaluation-panel">
            <div className="inspector-header">
                <span className="inspector-title">Evaluation</span>
                <button className="inspector-close" onClick={onClose}>
                    &times;
                </button>
            </div>

            <div className="toggle-group">
                {TABS.map((t) => (
                    <button
                        key={t.key}
                        className={`toggle-btn ${activeTab === t.key ? 'active' : ''}`}
                        onClick={() => setActiveTab(t.key)}
                    >
                        {t.label}
                    </button>
                ))}
            </div>

            {activeTab === 'metrics' && (
                <>
                    {/* Aggregate summary */}
                    {evaluation?.aggregate && (
                        <div className="card">
                            <div className="card-title">Aggregate Metrics <span style={{ color: '#64748b', fontWeight: 400, fontSize: 10 }}>(Holdout fold 0)</span></div>
                            <div className="metric-grid">
                                {evaluation.aggregate.map((m) => (
                                    <div className="metric-item" key={m.model}>
                                        <span className="metric-value" style={{ fontSize: 14, color: (MODEL_COLORS[m.model] || MODEL_COLORS.MLP).border }}>
                                            {m.r2_uniform.toFixed(4)}
                                        </span>
                                        <span className="metric-label">
                                            {m.model} R² &middot; MAE {m.mae_mean_pp.toFixed(1)}pp
                                        </span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    <div className="card">
                        <div className="card-title">Per-Class R² (All Models)</div>
                        <div style={{ height: 200, position: 'relative' }}>
                            <PerClassChart evaluation={evaluation} />
                        </div>
                    </div>
                </>
            )}

            {activeTab === 'stress' && (
                <>
                    <div className="card">
                        <div className="card-title">Noise Robustness</div>
                        <div style={{ height: 200, position: 'relative' }}>
                            <NoiseChart stressTests={stressTests} />
                        </div>
                    </div>

                    <div className="card">
                        <div className="card-title">Season &amp; Feature Importance</div>
                        <div style={{ height: 260, position: 'relative' }}>
                            <AblationChart stressTests={stressTests} />
                        </div>
                    </div>
                </>
            )}

            {activeTab === 'change' && (
                <div className="card">
                    <div className="card-title">False vs Missed Change Tradeoff</div>
                    <div style={{ height: 260, position: 'relative' }}>
                        <ChangeDetectionChart evaluation={evaluation} />
                    </div>
                </div>
            )}

            {activeTab === 'failure' && (
                <>
                    <div className="card">
                        <div className="card-title">Error by Dominant Land Cover</div>
                        <div style={{ height: 220, position: 'relative' }}>
                            <FailureChart failureAnalysis={failureAnalysis} />
                        </div>
                    </div>

                    {failureAnalysis && (
                        <div className="card">
                            <div className="card-title">Failure Details (MLP)</div>
                            <div className="metric-grid">
                                {failureAnalysis.filter((r) => r.model === 'MLP').map((r) => (
                                    <div className="metric-item" key={r.dominant_class}>
                                        <span className="metric-value" style={{
                                            fontSize: 13,
                                            color: r.mae_pp < 3 ? '#10b981' : r.mae_pp < 6 ? '#f59e0b' : '#ef4444',
                                        }}>
                                            {r.mae_pp.toFixed(1)}pp
                                        </span>
                                        <span className="metric-label">
                                            {CLASS_LABELS[r.dominant_class] || r.dominant_class}
                                            <br />
                                            <span style={{ color: '#64748b' }}>
                                                {r.n_cells.toLocaleString()} cells &middot; R²={r.r2_uniform.toFixed(3)}
                                            </span>
                                        </span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </>
            )}
        </div>
    );
}
