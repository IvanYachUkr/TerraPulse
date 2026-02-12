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

export default function CellInspector({
    cellDetail,
    selectedCell,
    onClose,
    classLabels,
    classColors,
    classes,
    models,
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

        // Predictions from all models
        if (cellDetail.predictions) {
            const modelKeys = Object.keys(cellDetail.predictions);
            // Just show all model predictions as grouped bars
            for (const mKey of modelKeys) {
                const preds = cellDetail.predictions[mKey];
                datasets.push({
                    label: MODEL_DISPLAY[mKey] || mKey,
                    data: classes.map((c) => (preds[c] || 0) * 100),
                    backgroundColor: bgColors.map((c) => c.replace('0.8', '0.4')),
                    borderColor: borderColors,
                    borderWidth: 1,
                    borderDash: [3, 3],
                });
            }
        }

        chartRef.current = new Chart(barRef.current, {
            type: 'bar',
            data: { labels, datasets: datasets.length > 2 ? [datasets[0], datasets[datasets.length - 1]] : datasets },
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
    }, [cellDetail, classes, classLabels, classColors]);

    if (selectedCell == null) return null;

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
                        <div style={{ height: 220, position: 'relative' }}>
                            <canvas ref={barRef} />
                        </div>
                    </div>

                    {/* Labels comparison */}
                    <div className="card">
                        <div className="card-title">Label Change (2020 &rarr; 2021)</div>
                        <div className="metric-grid">
                            {classes.map((c) => {
                                const v2020 = cellDetail.labels_2020?.[c];
                                const v2021 = cellDetail.labels_2021?.[c];
                                const delta = cellDetail.change?.[`delta_${c}`];
                                return (
                                    <div className="metric-item" key={c}>
                                        <span className="metric-value" style={{ fontSize: 14 }}>
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
                                <span className="metric-value" style={{ fontSize: 14 }}>
                                    {cellDetail.split?.fold === 0 ? 'Holdout' : 'Training'}
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
