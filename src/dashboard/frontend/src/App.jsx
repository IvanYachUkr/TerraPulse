import { useState, useEffect } from 'react';
import Header from './components/Header.jsx';
import Sidebar from './components/Sidebar.jsx';
import MapView from './components/MapView.jsx';
import CellInspector from './components/CellInspector.jsx';
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

// Years with actual labels
const LABEL_YEARS = [2020, 2021];
// Future years where we can only predict (no labels yet)
const ALL_YEARS = [2020, 2021, 2022, 2023, 2024, 2025];

export default function App() {
    const [selectedModel, setSelectedModel] = useState('mlp');
    const [viewMode, setViewMode] = useState('labels');
    const [selectedYear, setSelectedYear] = useState(2021);
    const [changeYearFrom, setChangeYearFrom] = useState(2020);
    const [changeYearTo, setChangeYearTo] = useState(2021);
    const [selectedClass, setSelectedClass] = useState('all');
    const [selectedCell, setSelectedCell] = useState(null);
    const [sidebarOpen, setSidebarOpen] = useState(true);

    // Data fetching
    const { data: grid, loading: gridLoading } = useApi('/api/grid');
    const { data: labels2020 } = useApi('/api/labels/2020');
    const { data: labels2021 } = useApi('/api/labels/2021');
    const { data: changeData } = useApi('/api/change');
    const { data: models } = useApi('/api/models');
    const { data: conformal } = useApi('/api/conformal');
    const { data: predictions } = useApi(`/api/predictions/${selectedModel}`);
    const { data: cellDetail } = useApi(
        selectedCell != null ? `/api/cell/${selectedCell}` : null
    );

    // Pick the right data based on view mode + year
    const getViewData = () => {
        switch (viewMode) {
            case 'labels':
                if (selectedYear === 2020) return labels2020;
                if (selectedYear === 2021) return labels2021;
                // Future years: show predictions instead
                return predictions;
            case 'predictions': return predictions;
            case 'change': return changeData;
            default: return labels2021;
        }
    };

    const isFutureYear = selectedYear > 2021;

    return (
        <>
            <Header
                sidebarOpen={sidebarOpen}
                onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
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
                        changeYearFrom={changeYearFrom}
                        onChangeYearFromChange={setChangeYearFrom}
                        changeYearTo={changeYearTo}
                        onChangeYearToChange={setChangeYearTo}
                        selectedClass={selectedClass}
                        onClassChange={setSelectedClass}
                        classes={CLASSES}
                        classLabels={CLASS_LABELS}
                        classColors={CLASS_COLORS}
                        conformal={conformal}
                        labelYears={LABEL_YEARS}
                        allYears={ALL_YEARS}
                        isFutureYear={isFutureYear}
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
                    classColors={CLASS_COLORS}
                    classes={CLASSES}
                    classLabels={CLASS_LABELS}
                    loading={gridLoading}
                    onCellClick={setSelectedCell}
                    selectedCell={selectedCell}
                    isFutureYear={selectedYear > 2021}
                />
                <CellInspector
                    cellDetail={cellDetail}
                    selectedCell={selectedCell}
                    onClose={() => setSelectedCell(null)}
                    classLabels={CLASS_LABELS}
                    classColors={CLASS_COLORS}
                    classes={CLASSES}
                    models={models}
                />
            </div>
        </>
    );
}
