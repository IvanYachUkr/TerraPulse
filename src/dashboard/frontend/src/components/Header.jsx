export default function Header({ sidebarOpen, onToggleSidebar, appMode, onAppModeChange }) {
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
            <div className="header-spacer" />
            {appMode === 'analytical' && (
                <span className="header-badge">Pixel-Level Land Cover</span>
            )}
        </header>
    );
}
