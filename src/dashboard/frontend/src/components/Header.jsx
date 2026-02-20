export default function Header({ sidebarOpen, onToggleSidebar, showComparison, onToggleComparison, showEvaluation, onToggleEvaluation, showExplainability, onToggleExplainability, appMode, onAppModeChange }) {
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
                    📊 Analytical
                </button>
                <button
                    className={`header-mode-btn ${appMode === 'deploy' ? 'active' : ''}`}
                    onClick={() => onAppModeChange('deploy')}
                >
                    🚀 Deploy
                </button>
            </div>
            <div className="header-spacer" />
            {appMode !== 'deploy' && (
                <>
                    <button
                        className={`header-btn ${showEvaluation ? 'active' : ''}`}
                        onClick={onToggleEvaluation}
                        title="Evaluation metrics"
                    >
                        &#128202; Eval
                    </button>
                    <button
                        className={`header-btn ${showExplainability ? 'active' : ''}`}
                        onClick={onToggleExplainability}
                        title="Model explainability"
                    >
                        🔍 Explain
                    </button>
                    <button
                        className={`header-btn ${showComparison ? 'active' : ''}`}
                        onClick={onToggleComparison}
                        title="Model comparison charts"
                    >
                        &#9776; Compare
                    </button>
                    <span className="header-badge">100m Grid &middot; 29,946 Cells</span>
                </>
            )}
        </header>
    );
}
