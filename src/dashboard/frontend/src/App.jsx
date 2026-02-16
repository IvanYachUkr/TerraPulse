import { useState } from 'react';
import Header from './components/Header.jsx';
import Sidebar from './components/Sidebar.jsx';
import MapView from './components/MapView.jsx';
import CellInspector from './components/CellInspector.jsx';
import ModelComparison from './components/ModelComparison.jsx';
import EvaluationPanel from './components/EvaluationPanel.jsx';
import { useApi } from './hooks/useApi.js';

const CLASSES = ['tree_cover', 'grassland', 'cropland', 'built_up', 'bare_sparse', 'water'];

const CLASS_COLORS = {
    tree_cover: [45, 106, 79],
    grassland: [149, 213, 178],
    cropland: [244, 162, 97],
    built_up: [231, 111, 81],
    bare_sparse: [212, 163, 115],
    water: [0, 150, 199],
};

const CLASS_LABELS = {
    tree_cover: 'Tree Cover',
    grassland: 'Grassland',
    cropland: 'Cropland',
    built_up: 'Built-up',
    bare_sparse: 'Bare/Sparse',
    water: 'Water',
};

// Years with actual labels (ground truth)
const LABEL_YEARS = [2020, 2021];

export default function App() {
    const [selectedModel, setSelectedModel] = useState('mlp');
    const [viewMode, setViewMode] = useState('labels');
    const [selectedYear, setSelectedYear] = useState(2021);

    const [selectedClass, setSelectedClass] = useState('all');
    const [selectedCell, setSelectedCell] = useState(null);
    const [sidebarOpen, setSidebarOpen] = useState(true);
    const [searchCellId, setSearchCellId] = useState(null);
    const [showComparison, setShowComparison] = useState(false);
    const [showEvaluation, setShowEvaluation] = useState(false);

    // Data fetching
    const { data: grid, loading: gridLoading } = useApi('/api/grid');
    const { data: labels2020 } = useApi('/api/labels/2020');
    const { data: labels2021 } = useApi('/api/labels/2021');
    const { data: changeData } = useApi('/api/change');
    const { data: models } = useApi('/api/models');

    // Predictions: year-aware fetch
    // For 2021 use the OOF endpoint, for 2022+ use the year-keyed endpoint
    const predUrl = viewMode === 'predictions' && selectedYear > 2021
        ? `/api/predictions/${selectedModel}/${selectedYear}`
        : `/api/predictions/${selectedModel}`;
    const { data: predictions } = useApi(predUrl);

    const { data: conformal } = useApi('/api/conformal');
    const { data: splitData } = useApi('/api/split');
    const { data: evaluationData } = useApi('/api/evaluation');
    const { data: stressTestsData } = useApi('/api/stress-tests');
    const { data: failureData } = useApi('/api/failure-analysis');
    const { data: explainData } = useApi('/api/explainability');
    const { data: cellDetail } = useApi(
        selectedCell != null ? `/api/cell/${selectedCell}` : null
    );

    // Pick the right data based on view mode
    const getViewData = () => {
        switch (viewMode) {
            case 'labels':
                return selectedYear === 2020 ? labels2020 : labels2021;
            case 'predictions':
                return predictions;
            case 'change':
                return changeData;
            case 'folds':
                return splitData;
            default:
                return labels2021;
        }
    };

    // When user searches a cell, also select it for inspection
    const handleSearchCell = (id) => {
        setSearchCellId(id);
        if (id != null) setSelectedCell(id);
    };

    return (
        <>
            <Header
                sidebarOpen={sidebarOpen}
                onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
                showComparison={showComparison}
                onToggleComparison={() => setShowComparison(!showComparison)}
                showEvaluation={showEvaluation}
                onToggleEvaluation={() => setShowEvaluation(!showEvaluation)}
            />
            <div className="app-layout">
                {sidebarOpen && (
                    <Sidebar
                        models={models}
                        selectedModel={selectedModel}
                        onModelChange={setSelectedModel}
                        viewMode={viewMode}
                        onViewModeChange={setViewMode}
                        selectedYear={selectedYear}
                        onYearChange={setSelectedYear}
                        selectedClass={selectedClass}
                        onClassChange={setSelectedClass}
                        classes={CLASSES}
                        classLabels={CLASS_LABELS}
                        classColors={CLASS_COLORS}
                        labelYears={LABEL_YEARS}
                        searchCellId={searchCellId}
                        onSearchCellId={handleSearchCell}
                    />
                )}
                <MapView
                    grid={grid}
                    viewData={getViewData()}
                    viewMode={viewMode}
                    selectedYear={selectedYear}
                    selectedClass={selectedClass}
                    selectedModel={selectedModel}
                    predictions={predictions}
                    labels2020={labels2020}
                    labels2021={labels2021}
                    changeData={changeData}
                    splitData={splitData}
                    classColors={CLASS_COLORS}
                    classes={CLASSES}
                    classLabels={CLASS_LABELS}
                    loading={gridLoading}
                    onCellClick={setSelectedCell}
                    selectedCell={selectedCell}
                    isFutureYear={false}
                    searchCellId={searchCellId}
                />
                {showComparison && (
                    <div className="comparison-panel">
                        <div className="inspector-header">
                            <span className="inspector-title">Model Comparison</span>
                            <button className="inspector-close" onClick={() => setShowComparison(false)}>
                                &times;
                            </button>
                        </div>
                        <ModelComparison models={models} evaluation={evaluationData} />
                    </div>
                )}
                {showEvaluation && (
                    <EvaluationPanel
                        evaluation={evaluationData}
                        stressTests={stressTestsData}
                        failureAnalysis={failureData}
                        explainability={explainData}
                        onClose={() => setShowEvaluation(false)}
                    />
                )}
                <CellInspector
                    cellDetail={cellDetail}
                    selectedCell={selectedCell}
                    onClose={() => setSelectedCell(null)}
                    classLabels={CLASS_LABELS}
                    classColors={CLASS_COLORS}
                    classes={CLASSES}
                    models={models}
                    selectedModel={selectedModel}
                    conformal={conformal}
                />
            </div>
        </>
    );
}
