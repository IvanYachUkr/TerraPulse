export default function Header({
    sidebarOpen,
    onToggleSidebar,
    appMode,
    onAppModeChange,
    nurembergDataMode,
    nurembergYear,
    nurembergSecondaryYear,
    nurembergPredModel,
}) {
    // Build context info text
    let infoText = '';
    if (appMode === 'analytical') {
        if (nurembergDataMode === 'labels') {
            infoText = `Ground Truth \u00b7 ESA WorldCover ${nurembergYear}`;
        } else if (nurembergDataMode === 'experimental') {
            infoText = `Experimental \u00b7 4-Fold Spatial CV \u00b7 2020\u21922021`;
        } else if (nurembergSecondaryYear !== null) {
            infoText = `Predictions · Nuremberg ${nurembergYear} → ${nurembergSecondaryYear} · Change Detection`;
        } else if (nurembergYear >= 2026) {
            infoText = `Predictions · Nuremberg ${nurembergYear} · Future Forecast`;
        } else {
            const modelName = nurembergPredModel === 'ssnet_v8' ? 'Spectral-Spatial Net' : 'CatBoost';
            infoText = `${modelName} Predictions \u00b7 Nuremberg ${nurembergYear}`;
        }
    } else if (appMode === 'deploy') {
        infoText = `Global Predictions \u00b7 Regional Predictor`;
    }

    return (
        <header className="header">
            {appMode !== 'deploy' && (
                <button
                    className="sidebar-toggle"
                    onClick={onToggleSidebar}
                    title={sidebarOpen ? 'Hide sidebar' : 'Show sidebar'}
                >
                    {sidebarOpen ? '\u2190' : '\u2192'}
                </button>
            )}
            <img
                src="/logo.png"
                alt="TerraPulse"
                className="header-logo-img"
            />
            <span className="header-logo">TerraPulse</span>
            <div className="header-mode-toggle">
                <button
                    className={`header-mode-btn ${appMode === 'analytical' ? 'active' : ''}`}
                    onClick={() => onAppModeChange('analytical')}
                >
                    🏰 Nuremberg
                </button>
                <button
                    className={`header-mode-btn ${appMode === 'deploy' ? 'active' : ''}`}
                    onClick={() => onAppModeChange('deploy')}
                >
                    🌍 Global
                </button>
            </div>

            {/* Centered context info */}
            <div className="header-info">
                {infoText && (
                    <span className="header-info-text">{infoText}</span>
                )}
            </div>

            {appMode === 'analytical' && (
                <span className="header-badge">Pixel-Level Land Cover</span>
            )}
        </header>
    );
}
