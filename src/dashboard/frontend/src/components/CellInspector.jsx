import { useEffect, useRef, useMemo } from 'react';
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
    ridge: { bg: 'rgba(59,130,246,0.5)', border: 'rgb(59,130,246)' },
    elasticnet: { bg: 'rgba(139,92,246,0.5)', border: 'rgb(139,92,246)' },
    extratrees: { bg: 'rgba(16,185,129,0.5)', border: 'rgb(16,185,129)' },
    rf: { bg: 'rgba(245,158,11,0.5)', border: 'rgb(245,158,11)' },
    catboost: { bg: 'rgba(239,68,68,0.5)', border: 'rgb(239,68,68)' },
    mlp: { bg: 'rgba(236,72,153,0.5)', border: 'rgb(236,72,153)' },
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

        // Show selected model's prediction (not all models — avoids clutter)
        if (cellDetail.predictions && cellDetail.predictions[selectedModel]) {
            const preds = cellDetail.predictions[selectedModel];
            const mc = MODEL_COLORS[selectedModel] || { bg: 'rgba(148,163,184,0.5)', border: 'rgb(148,163,184)' };
            datasets.push({
                label: `${MODEL_DISPLAY[selectedModel] || selectedModel} Pred`,
                data: classes.map((c) => (preds[c] || 0) * 100),
                backgroundColor: mc.bg,
                borderColor: mc.border,
                borderWidth: 1,
            });
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
    }, [cellDetail, selectedModel, classes, classLabels, classColors]);

    if (selectedCell == null) return null;

    const isHoldout = cellDetail?.split?.fold === 0;

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
