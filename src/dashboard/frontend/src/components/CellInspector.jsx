import { useEffect, useRef } from 'react';
import { Chart, registerables } from 'chart.js';

Chart.register(...registerables);

const MODEL_DISPLAY = {
    ridge: 'Ridge',
    elasticnet: 'ElasticNet',
    extratrees: 'ExtraTrees',
    rf: 'Random Forest',
    catboost: 'CatBoost',
    mlp: 'MLP',
};

// Distinct palette for each model (not tied to class colors)
const MODEL_COLORS = {
    mlp: { bg: 'rgba(236,72,153,0.5)', border: 'rgb(236,72,153)' },
    tree: { bg: 'rgba(16,185,129,0.5)', border: 'rgb(16,185,129)' },
    ridge: { bg: 'rgba(59,130,246,0.5)', border: 'rgb(59,130,246)' },
};

export default function CellInspector({
    cellDetail,
    selectedCell,
    onClose,
    classLabels,
    classColors,
    classes,
    models,
    selectedModel,
    conformal,
}) {
    const barRef = useRef(null);
    const chartRef = useRef(null);

    useEffect(() => {
        if (!cellDetail || !barRef.current) return;

        // Destroy previous chart
        if (chartRef.current) {
            chartRef.current.destroy();
        }

        const labels = classes.map((c) => classLabels[c]);
        const bgColors = classes.map((c) => {
            const [r, g, b] = classColors[c];
            return `rgba(${r},${g},${b},0.8)`;
        });
        const borderColors = classes.map((c) => {
            const [r, g, b] = classColors[c];
            return `rgb(${r},${g},${b})`;
        });

        const datasets = [];

        // True labels 2021
        if (cellDetail.labels_2021) {
            datasets.push({
                label: 'True 2021',
                data: classes.map((c) => (cellDetail.labels_2021[c] || 0) * 100),
                backgroundColor: bgColors,
                borderColor: borderColors,
                borderWidth: 1,
            });
        }

        // Show selected model's prediction with conformal error bars
        if (cellDetail.predictions && cellDetail.predictions[selectedModel]) {
            const preds = cellDetail.predictions[selectedModel];
            const mc = MODEL_COLORS[selectedModel] || { bg: 'rgba(148,163,184,0.5)', border: 'rgb(148,163,184)' };

            // Get conformal interval widths for this model (half-width for symmetric bars)
            const conformalData = conformal?.[selectedModel];
            const errorBars = classes.map((c) => {
                if (!conformalData?.[c]) return 0;
                return conformalData[c].median_width_pp / 2; // half-width in pp
            });

            const predData = classes.map((c) => (preds[c] || 0) * 100);

            datasets.push({
                label: `${MODEL_DISPLAY[selectedModel] || selectedModel} Pred`,
                data: predData,
                backgroundColor: mc.bg,
                borderColor: mc.border,
                borderWidth: 1,
            });

            // Conformal interval band (low and high as a separate floating bar dataset)
            if (conformalData) {
                const floatingData = classes.map((c, i) => {
                    const pred = predData[i];
                    const halfW = errorBars[i];
                    return [Math.max(0, pred - halfW), Math.min(100, pred + halfW)];
                });
                datasets.push({
                    label: `Conformal 90% CI`,
                    data: floatingData,
                    backgroundColor: 'rgba(255,255,255,0.07)',
                    borderColor: 'rgba(255,255,255,0.3)',
                    borderWidth: 1,
                    borderSkipped: false,
                });
            }
        }

        chartRef.current = new Chart(barRef.current, {
            type: 'bar',
            data: { labels, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'bottom',
                        labels: {
                            color: '#94a3b8',
                            font: { size: 10, family: 'Inter' },
                            boxWidth: 12,
                        },
                    },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => {
                                if (ctx.dataset.label.startsWith('Conformal')) {
                                    const [lo, hi] = ctx.raw;
                                    return `${ctx.dataset.label}: [${lo.toFixed(1)}%, ${hi.toFixed(1)}%]`;
                                }
                                return `${ctx.dataset.label}: ${ctx.formattedValue}%`;
                            },
                        },
                    },
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: { display: true, text: '%', color: '#64748b', font: { size: 11 } },
                        ticks: { color: '#94a3b8', font: { size: 10 } },
                        grid: { color: 'rgba(255,255,255,0.04)' },
                    },
                    x: {
                        ticks: { color: '#94a3b8', font: { size: 10 }, maxRotation: 45 },
                        grid: { display: false },
                    },
                },
            },
        });

        return () => {
            if (chartRef.current) {
                chartRef.current.destroy();
                chartRef.current = null;
            }
        };
    }, [cellDetail, selectedModel, conformal, classes, classLabels, classColors]);

    if (selectedCell == null) return null;

    const isHoldout = cellDetail?.split?.fold === 0;

    // Conformal coverage summary for the selected model
    const conformalSummary = conformal?.[selectedModel];

    return (
        <div className={`inspector ${selectedCell == null ? 'hidden' : ''}`}>
            <div className="inspector-header">
                <span className="inspector-title">Cell {selectedCell}</span>
                <button className="inspector-close" onClick={onClose}>
                    &times;
                </button>
            </div>

            {!cellDetail ? (
                <div style={{ textAlign: 'center', padding: '40px 0' }}>
                    <div className="spinner" style={{ margin: '0 auto' }} />
                </div>
            ) : (
                <>
                    {/* Proportions chart */}
                    <div className="card">
                        <div className="card-title">True vs Predicted Proportions</div>
                        {!isHoldout && (
                            <div className="info-badge" style={{ marginBottom: 8 }}>
                                Training cell &mdash; no predictions available
                            </div>
                        )}
                        <div style={{ height: 220, position: 'relative' }}>
                            <canvas ref={barRef} />
                        </div>
                    </div>

                    {/* Conformal summary */}
                    {conformalSummary && isHoldout && (
                        <div className="card">
                            <div className="card-title">
                                Conformal Intervals ({MODEL_DISPLAY[selectedModel] || selectedModel})
                            </div>
                            <div className="metric-grid">
                                {classes.map((c) => {
                                    const ci = conformalSummary[c];
                                    if (!ci) return null;
                                    return (
                                        <div className="metric-item" key={c}>
                                            <span className="metric-value" style={{
                                                fontSize: 13,
                                                color: ci.coverage_pct >= 85 ? '#10b981' : ci.coverage_pct >= 70 ? '#f59e0b' : '#ef4444',
                                            }}>
                                                {ci.coverage_pct.toFixed(0)}%
                                            </span>
                                            <span className="metric-label" style={{ fontSize: 10 }}>
                                                {classLabels[c]}
                                                <br />
                                                <span style={{ color: '#64748b' }}>
                                                    &plusmn;{(ci.median_width_pp / 2).toFixed(1)}pp
                                                </span>
                                            </span>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    )}

                    {/* Labels comparison */}
                    <div className="card">
                        <div className="card-title">Label Change (2020 &rarr; 2021)</div>
                        <div className="metric-grid">
                            {classes.map((c) => {
                                const delta = cellDetail.change?.[`delta_${c}`];
                                return (
                                    <div className="metric-item" key={c}>
                                        <span className="metric-value" style={{
                                            fontSize: 14,
                                            color: delta > 0.01 ? '#10b981' : delta < -0.01 ? '#ef4444' : '#94a3b8'
                                        }}>
                                            {delta != null ? (delta > 0 ? '+' : '') + (delta * 100).toFixed(1) + 'pp' : '-'}
                                        </span>
                                        <span className="metric-label">
                                            {classLabels[c]}
                                        </span>
                                    </div>
                                );
                            })}
                        </div>
                    </div>

                    {/* Cell metadata */}
                    <div className="card">
                        <div className="card-title">Metadata</div>
                        <div className="metric-grid">
                            <div className="metric-item">
                                <span className="metric-value" style={{ fontSize: 14 }}>
                                    {cellDetail.split?.fold ?? '-'}
                                </span>
                                <span className="metric-label">CV Fold</span>
                            </div>
                            <div className="metric-item">
                                <span className="metric-value" style={{ fontSize: 14 }}>
                                    {cellDetail.split?.tile_group ?? '-'}
                                </span>
                                <span className="metric-label">Tile Group</span>
                            </div>
                            <div className="metric-item">
                                <span className="metric-value" style={{ fontSize: 14 }}>
                                    {Object.keys(cellDetail.predictions || {}).length}
                                </span>
                                <span className="metric-label">Models Available</span>
                            </div>
                            <div className="metric-item">
                                <span className="metric-value" style={{
                                    fontSize: 14,
                                    color: isHoldout ? '#3b82f6' : '#94a3b8'
                                }}>
                                    {isHoldout ? 'Holdout' : 'Training'}
                                </span>
                                <span className="metric-label">Split Role</span>
                            </div>
                        </div>
                    </div>
                </>
            )}
        </div>
    );
}
