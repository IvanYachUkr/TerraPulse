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

const MODEL_COLORS_PERM = {
    mlp: 'rgba(236,72,153,0.6)',
    tree: 'rgba(16,185,129,0.6)',
};

const CLASS_COLORS_SHAP = {
    tree_cover: 'rgba(45,106,79,0.7)',
    grassland: 'rgba(149,213,178,0.7)',
    cropland: 'rgba(244,162,97,0.7)',
    built_up: 'rgba(231,111,81,0.7)',
    bare_sparse: 'rgba(212,163,115,0.7)',
    water: 'rgba(0,150,199,0.7)',
};

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
            ticks: { color: '#94a3b8', font: { size: 9 } },
            grid: { color: 'rgba(255,255,255,0.04)' },
        },
        y: {
            ticks: { color: '#94a3b8', font: { size: 9 } },
            grid: { color: 'rgba(255,255,255,0.04)' },
        },
    },
};

// ── Fullscreen image overlay ──
function FullscreenOverlay({ src, alt, onClose }) {
    if (!src) return null;
    return (
        <div className="fullscreen-overlay" onClick={onClose}>
            <img src={src} alt={alt} className="fullscreen-image" onClick={e => e.stopPropagation()} />
            <button className="fullscreen-close" onClick={onClose}>&times;</button>
        </div>
    );
}

// ── Clickable plot image with fullscreen button ──
function PlotImage({ src, alt, style }) {
    const [fullscreen, setFullscreen] = useState(false);
    return (
        <>
            <div style={{ position: 'relative', display: 'inline-block', ...style }}>
                <img src={src} alt={alt}
                    style={{ maxWidth: '100%', borderRadius: 6, border: '1px solid #334155', cursor: 'pointer' }}
                    onClick={() => setFullscreen(true)}
                />
                <button
                    className="plot-fullscreen-btn"
                    onClick={() => setFullscreen(true)}
                    title="View fullscreen"
                >
                    ⛶
                </button>
            </div>
            {fullscreen && <FullscreenOverlay src={src} alt={alt} onClose={() => setFullscreen(false)} />}
        </>
    );
}

// ── Permutation Importance chart ──
function PermutationChart({ explainability }) {
    const canvasRef = useRef(null);
    const chartRef = useRef(null);

    useEffect(() => {
        if (!canvasRef.current || !explainability) return;
        if (chartRef.current) chartRef.current.destroy();

        const mlpPerm = explainability?.models?.mlp?.permutation?.slice(0, 15) || [];
        const treePerm = explainability?.models?.tree?.permutation?.slice(0, 15) || [];

        const allFeatures = [...new Set([...mlpPerm.map(f => f.feature), ...treePerm.map(f => f.feature)])].slice(0, 15);
        const mlpMap = Object.fromEntries(mlpPerm.map(f => [f.feature, f.importance]));
        const treeMap = Object.fromEntries(treePerm.map(f => [f.feature, f.importance]));

        chartRef.current = new Chart(canvasRef.current, {
            type: 'bar',
            data: {
                labels: allFeatures.map(f => f.replace(/_/g, ' ').substring(0, 25)),
                datasets: [
                    { label: 'MLP', data: allFeatures.map(f => mlpMap[f] || 0), backgroundColor: MODEL_COLORS_PERM.mlp, borderWidth: 0 },
                    { label: 'LightGBM', data: allFeatures.map(f => treeMap[f] || 0), backgroundColor: MODEL_COLORS_PERM.tree, borderWidth: 0 },
                ],
            },
            options: {
                ...DARK_CHART_OPTS,
                indexAxis: 'y',
                scales: {
                    ...DARK_CHART_OPTS.scales,
                    x: { ...DARK_CHART_OPTS.scales.x, title: { display: true, text: 'Δ R² when shuffled', color: '#64748b', font: { size: 10 } } },
                    y: { ...DARK_CHART_OPTS.scales.y, ticks: { ...DARK_CHART_OPTS.scales.y.ticks, font: { size: 8 } } },
                },
            },
        });

        return () => { if (chartRef.current) { chartRef.current.destroy(); chartRef.current = null; } };
    }, [explainability]);

    return (
        <div style={{ width: '100%', height: '100%', position: 'relative' }}>
            <canvas ref={canvasRef} />
        </div>
    );
}

// ── SHAP per-class importance chart ──
function ShapChart({ explainability }) {
    const canvasRef = useRef(null);
    const chartRef = useRef(null);
    const [shapModel, setShapModel] = useState('mlp');

    useEffect(() => {
        if (!canvasRef.current || !explainability) return;
        if (chartRef.current) chartRef.current.destroy();

        const shapData = explainability?.models?.[shapModel]?.shap?.slice(0, 12) || [];
        const labels = shapData.map(f => f.feature.replace(/_/g, ' ').substring(0, 28));

        const config = shapData.length ? {
            type: 'bar',
            data: {
                labels,
                datasets: Object.keys(CLASS_COLORS_SHAP).map(cls => ({
                    label: CLASS_LABELS[cls] || cls,
                    data: shapData.map(f => f[cls] || 0),
                    backgroundColor: CLASS_COLORS_SHAP[cls],
                })),
            },
            options: {
                ...DARK_CHART_OPTS,
                indexAxis: 'y',
                scales: {
                    ...DARK_CHART_OPTS.scales,
                    x: { ...DARK_CHART_OPTS.scales.x, stacked: true, title: { display: true, text: 'Mean |SHAP|', color: '#64748b', font: { size: 10 } } },
                    y: { ...DARK_CHART_OPTS.scales.y, stacked: true, ticks: { ...DARK_CHART_OPTS.scales.y.ticks, font: { size: 8 } } },
                },
            },
        } : null;

        if (config) chartRef.current = new Chart(canvasRef.current, config);

        return () => { if (chartRef.current) { chartRef.current.destroy(); chartRef.current = null; } };
    }, [explainability, shapModel]);

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
            <div style={{ display: 'flex', gap: 6, marginBottom: 8, flexShrink: 0 }}>
                <button className={`toggle-btn ${shapModel === 'mlp' ? 'active' : ''}`} onClick={() => setShapModel('mlp')}>MLP</button>
                <button className={`toggle-btn ${shapModel === 'tree' ? 'active' : ''}`} onClick={() => setShapModel('tree')}>LightGBM</button>
            </div>
            <div style={{ flex: 1, position: 'relative', minHeight: 0 }}>
                <canvas ref={canvasRef} />
            </div>
        </div>
    );
}

// ── SHAP Deep Dive (beeswarm + dependence plots) ──
function ShapDeepDive() {
    const [manifest, setManifest] = useState(null);
    const [activeModel, setActiveModel] = useState('mlp');
    const [activeClass, setActiveClass] = useState('tree_cover');
    const [viewMode, setViewMode] = useState('beeswarm');
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

    return (
        <div>
            {/* Model + view toggle */}
            <div style={{ display: 'flex', gap: 6, marginBottom: 8 }}>
                <button className={`toggle-btn ${activeModel === 'mlp' ? 'active' : ''}`}
                    onClick={() => { setActiveModel('mlp'); setActiveDep(0); }}>MLP</button>
                <button className={`toggle-btn ${activeModel === 'tree' ? 'active' : ''}`}
                    onClick={() => { setActiveModel('tree'); setActiveDep(0); }}>LightGBM</button>
                <div style={{ flex: 1 }} />
                <button className={`toggle-btn ${viewMode === 'beeswarm' ? 'active' : ''}`}
                    onClick={() => setViewMode('beeswarm')} style={{ fontSize: 10 }}>Beeswarm</button>
                <button className={`toggle-btn ${viewMode === 'dependence' ? 'active' : ''}`}
                    onClick={() => setViewMode('dependence')} style={{ fontSize: 10 }}>Dependence</button>
            </div>

            {/* Class tabs */}
            <div style={{ display: 'flex', gap: 4, marginBottom: 10, flexWrap: 'wrap' }}>
                {modelData.classes?.map(c => (
                    <button key={c.class}
                        className={`toggle-btn ${activeClass === c.class ? 'active' : ''}`}
                        onClick={() => { setActiveClass(c.class); setActiveDep(0); }}
                        style={{ fontSize: 10, padding: '3px 8px' }}>
                        {CLASS_LABELS[c.class] || c.class}
                    </button>
                ))}
            </div>

            {/* Plot display */}
            {viewMode === 'beeswarm' && classInfo && (
                <div style={{ textAlign: 'center' }}>
                    <PlotImage
                        src={`/api/shap-plots/${classInfo.beeswarm}`}
                        alt={`SHAP Beeswarm - ${activeModel} - ${activeClass}`}
                    />
                    <div style={{ color: '#94a3b8', fontSize: 10, marginTop: 6 }}>
                        Each dot = one sample. X-axis = SHAP impact on prediction. Color = feature value (blue=low, red=high).
                    </div>
                </div>
            )}

            {viewMode === 'dependence' && classInfo && (
                <div>
                    <div style={{ display: 'flex', gap: 4, marginBottom: 8, flexWrap: 'wrap' }}>
                        {depPlots.map((d, i) => {
                            const shortFeat = d.feature.length > 22 ? d.feature.substring(0, 21) + '.' : d.feature;
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
                            <PlotImage
                                src={`/api/shap-plots/${depPlots[activeDep].file}`}
                                alt={`SHAP Dependence - ${depPlots[activeDep].feature}`}
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

// ── Explanation cards ──
function ExplanationCards({ explainability }) {
    const mlp = explainability?.models?.mlp || {};
    const tree = explainability?.models?.tree || {};
    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {mlp.helpful && (
                <div style={{ background: 'rgba(16,185,129,0.08)', borderRadius: 6, padding: 10 }}>
                    <div style={{ color: '#10b981', fontWeight: 600, fontSize: 11, marginBottom: 4 }}>✓ Helpful — MLP</div>
                    <div style={{ color: '#e2e8f0', fontSize: 11 }}>{mlp.helpful.explanation}</div>
                    <div style={{ color: '#94a3b8', fontSize: 10, marginTop: 2 }}>{mlp.helpful.rationale}</div>
                </div>
            )}
            {mlp.misleading && (
                <div style={{ background: 'rgba(245,158,11,0.08)', borderRadius: 6, padding: 10 }}>
                    <div style={{ color: '#f59e0b', fontWeight: 600, fontSize: 11, marginBottom: 4 }}>⚠ Misleading — MLP</div>
                    <div style={{ color: '#e2e8f0', fontSize: 11 }}>{mlp.misleading.explanation}</div>
                    <div style={{ color: '#94a3b8', fontSize: 10, marginTop: 2 }}>{mlp.misleading.pitfall || mlp.misleading.note}</div>
                </div>
            )}
            {tree.helpful && (
                <div style={{ background: 'rgba(16,185,129,0.08)', borderRadius: 6, padding: 10 }}>
                    <div style={{ color: '#10b981', fontWeight: 600, fontSize: 11, marginBottom: 4 }}>✓ Helpful — LightGBM</div>
                    <div style={{ color: '#e2e8f0', fontSize: 11 }}>{tree.helpful.explanation}</div>
                    <div style={{ color: '#94a3b8', fontSize: 10, marginTop: 2 }}>{tree.helpful.rationale}</div>
                </div>
            )}
        </div>
    );
}


// ── Main ExplainabilityPanel (full-screen overlay) ──
export default function ExplainabilityPanel({ explainability, onClose }) {
    return (
        <div className="explainability-panel">
            <div className="explainability-header">
                <span className="inspector-title">🔍 Model Explainability</span>
                <button className="inspector-close" onClick={onClose}>&times;</button>
            </div>
            <div className="explainability-content">
                <div className="explainability-grid">
                    <div className="card">
                        <div className="card-title">Permutation Importance (Top 15)</div>
                        <div style={{ height: 380, position: 'relative', overflow: 'hidden' }}>
                            <PermutationChart explainability={explainability} />
                        </div>
                    </div>

                    <div className="card">
                        <div className="card-title">SHAP per Class (Top 12)</div>
                        <div style={{ height: 380, position: 'relative', overflow: 'hidden' }}>
                            <ShapChart explainability={explainability} />
                        </div>
                    </div>

                    <div className="card" style={{ gridColumn: '1 / -1' }}>
                        <div className="card-title">SHAP Deep Dive — Beeswarm &amp; Dependence Plots</div>
                        <ShapDeepDive />
                    </div>

                    <div className="card" style={{ gridColumn: '1 / -1' }}>
                        <div className="card-title">Explanations</div>
                        <ExplanationCards explainability={explainability} />
                    </div>
                </div>
            </div>
        </div>
    );
}
