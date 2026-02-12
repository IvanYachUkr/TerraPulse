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

const MODEL_COLORS_BAR = {
    ridge: { bg: 'rgba(59,130,246,0.6)', border: 'rgb(59,130,246)' },
    elasticnet: { bg: 'rgba(139,92,246,0.6)', border: 'rgb(139,92,246)' },
    extratrees: { bg: 'rgba(16,185,129,0.6)', border: 'rgb(16,185,129)' },
    rf: { bg: 'rgba(245,158,11,0.6)', border: 'rgb(245,158,11)' },
    catboost: { bg: 'rgba(239,68,68,0.6)', border: 'rgb(239,68,68)' },
    mlp: { bg: 'rgba(236,72,153,0.6)', border: 'rgb(236,72,153)' },
};

export default function ModelComparison({ models }) {
    const r2Ref = useRef(null);
    const maeRef = useRef(null);
    const r2ChartRef = useRef(null);
    const maeChartRef = useRef(null);

    useEffect(() => {
        if (!models || !r2Ref.current || !maeRef.current) return;

        // Destroy previous
        if (r2ChartRef.current) r2ChartRef.current.destroy();
        if (maeChartRef.current) maeChartRef.current.destroy();

        const sorted = [...models].sort((a, b) => b.r2_uniform - a.r2_uniform);
        const labels = sorted.map((m) => MODEL_DISPLAY[m.model] || m.model);
        const bgColors = sorted.map((m) => MODEL_COLORS_BAR[m.model]?.bg || 'rgba(148,163,184,0.6)');
        const borderColors = sorted.map((m) => MODEL_COLORS_BAR[m.model]?.border || 'rgb(148,163,184)');

        const chartOpts = {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: 'y',
            plugins: {
                legend: { display: false },
            },
            scales: {
                x: {
                    ticks: { color: '#94a3b8', font: { size: 10 } },
                    grid: { color: 'rgba(255,255,255,0.04)' },
                },
                y: {
                    ticks: { color: '#f0f4f8', font: { size: 11, family: 'Inter' } },
                    grid: { display: false },
                },
            },
        };

        r2ChartRef.current = new Chart(r2Ref.current, {
            type: 'bar',
            data: {
                labels,
                datasets: [{
                    label: 'R2',
                    data: sorted.map((m) => m.r2_uniform),
                    backgroundColor: bgColors,
                    borderColor: borderColors,
                    borderWidth: 1,
                }],
            },
            options: {
                ...chartOpts,
                scales: {
                    ...chartOpts.scales,
                    x: { ...chartOpts.scales.x, min: 0, max: 1, title: { display: true, text: 'R2 (uniform)', color: '#64748b', font: { size: 10 } } },
                },
            },
        });

        maeChartRef.current = new Chart(maeRef.current, {
            type: 'bar',
            data: {
                labels,
                datasets: [{
                    label: 'MAE (pp)',
                    data: sorted.map((m) => m.mae_mean_pp),
                    backgroundColor: bgColors,
                    borderColor: borderColors,
                    borderWidth: 1,
                }],
            },
            options: {
                ...chartOpts,
                scales: {
                    ...chartOpts.scales,
                    x: { ...chartOpts.scales.x, min: 0, title: { display: true, text: 'MAE (pp)', color: '#64748b', font: { size: 10 } } },
                },
            },
        });

        return () => {
            if (r2ChartRef.current) { r2ChartRef.current.destroy(); r2ChartRef.current = null; }
            if (maeChartRef.current) { maeChartRef.current.destroy(); maeChartRef.current = null; }
        };
    }, [models]);

    if (!models) return null;

    return (
        <div className="model-comparison">
            <div className="card">
                <div className="card-title">Model R&sup2; Comparison</div>
                <div style={{ height: 160, position: 'relative' }}>
                    <canvas ref={r2Ref} />
                </div>
            </div>
            <div className="card" style={{ marginTop: 12 }}>
                <div className="card-title">Model MAE Comparison</div>
                <div style={{ height: 160, position: 'relative' }}>
                    <canvas ref={maeRef} />
                </div>
            </div>
        </div>
    );
}
