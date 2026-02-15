export default function Header({ sidebarOpen, onToggleSidebar, showComparison, onToggleComparison, showEvaluation, onToggleEvaluation }) {
    return (
        <header className="header">
            <button
                className="sidebar-toggle"
                onClick={onToggleSidebar}
                title={sidebarOpen ? 'Hide sidebar' : 'Show sidebar'}
            >
                {sidebarOpen ? '\u2190' : '\u2192'}
            </button>
            <img
                src="/logo.png"
                alt="TerraPulse"
                className="header-logo-img"
            />
            <span className="header-logo">TerraPulse</span>
            <span className="header-subtitle">Urban Land Cover Dashboard &mdash; Nuremberg</span>
            <div className="header-spacer" />
            <button
                className={`header-btn ${showEvaluation ? 'active' : ''}`}
                onClick={onToggleEvaluation}
                title="Evaluation metrics"
            >
                &#128202; Eval
            </button>
            <button
                className={`header-btn ${showComparison ? 'active' : ''}`}
                onClick={onToggleComparison}
                title="Model comparison charts"
            >
                &#9776; Compare
            </button>
            <span className="header-badge">100m Grid &middot; 29,946 Cells</span>
        </header>
    );
}
