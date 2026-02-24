import { useState, useMemo } from 'react';
import Header from './components/Header.jsx';
import Sidebar from './components/Sidebar.jsx';
import MapView from './components/MapView.jsx';
import CellInspector from './components/CellInspector.jsx';
import DeployView from './components/DeployView.jsx';
import NurembergMapView from './components/NurembergMapView.jsx';
import { useApi } from './hooks/useApi.js';

// TODO load nuremberg polygon
//\src\dashboard\frontend\public\nuremberg_stat_bezirke_wgs84.geojson

const CLASSES = ['tree_cover', 'shrubland', 'grassland', 'cropland', 'built_up', 'bare_sparse', 'water']; // TODO: remove Shrubland for Nuremberg header selection

// Nuremberg classes (no shrubland — remapped to grassland)
const NUREMBERG_CLASSES = ['tree_cover', 'grassland', 'cropland', 'built_up', 'bare_sparse', 'water'];

const CLASS_COLORS = {
    tree_cover: [45, 106, 79],
    shrubland: [106, 153, 78],
    grassland: [149, 213, 178],
    cropland: [244, 162, 97],
    built_up: [231, 111, 81],
    bare_sparse: [212, 163, 115],
    water: [0, 150, 199],
};

const CLASS_LABELS = {
    tree_cover: 'Tree Cover',
    shrubland: 'Shrubland_TODO_REMOVE',
    grassland: 'Grassland',
    cropland: 'Cropland',
    built_up: 'Built-up',
    bare_sparse: 'Bare/Sparse',
    water: 'Water',
};

// Years with actual labels (ground truth)
const LABEL_YEARS = [2020, 2021];
// All years available (labels + predictions)
const ALL_YEARS = [2020, 2021, 2022, 2023, 2024, 2025];

export default function App() {
    const [appMode, setAppMode] = useState('analytical');
    const [selectedModel, setSelectedModel] = useState('mlp');
    const [viewMode, setViewMode] = useState('labels');
    const [selectedYear, setSelectedYear] = useState(2021);

    // Change view: from/to year pickers
    const [changeYearFrom, setChangeYearFrom] = useState(2020);
    const [changeYearTo, setChangeYearTo] = useState(2021);

    const [selectedClass, setSelectedClass] = useState('all');
    const [selectedCell, setSelectedCell] = useState(null);
    const [sidebarOpen, setSidebarOpen] = useState(true);
    const [searchCellId, setSearchCellId] = useState(null);

    // Nuremberg pixel mode state
    const [nurembergResolution, setNurembergResolution] = useState(5);
    const [nurembergYear, setNurembergYear] = useState(2021);
    const [nurembergDataMode, setNurembergDataMode] = useState('labels');

    // Data fetching — labels (always loaded)
    const { data: grid, loading: gridLoading } = useApi('/api/grid');
    const { data: labels2020 } = useApi('/api/labels/2020');
    const { data: labels2021 } = useApi('/api/labels/2021');
    const { data: models } = useApi('/api/models');

    // Nuremberg pixel mode data
    const { data: nurembergMeta } = useApi('/api/nuremberg/meta');
    const { data: nurembergBoundary } = useApi('/api/nuremberg/boundary');

    // Predictions: year-aware fetch for predictions view
    const predUrl = viewMode === 'predictions' && selectedYear > 2021
        ? `/api/predictions/${selectedModel}/${selectedYear}`
        : `/api/predictions/${selectedModel}`;
    const { data: predictions } = useApi(predUrl);

    // Change view: fetch prediction data for "from" and "to" years (only when > 2021)
    const changeFromUrl = changeYearFrom > 2021
        ? `/api/predictions/${selectedModel}/${changeYearFrom}` : null;
    const changeToUrl = changeYearTo > 2021
        ? `/api/predictions/${selectedModel}/${changeYearTo}` : null;
    const { data: changeFromPred } = useApi(changeFromUrl);
    const { data: changeToPred } = useApi(changeToUrl);

    const { data: conformal } = useApi('/api/conformal');
    const { data: cellDetail } = useApi(
        selectedCell != null ? `/api/cell/${selectedCell}` : null
    );

    // Resolve year data: labels for 2020/2021, predictions for 2022+
    const resolveYearData = (year) => {
        if (year === 2020) return labels2020;
        if (year === 2021) return labels2021;
        // For 2022+, use the fetched prediction data
        if (year === changeYearFrom && changeFromPred) return changeFromPred;
        if (year === changeYearTo && changeToPred) return changeToPred;
        return null;
    };

    // Compute change data dynamically from two years
    const computedChangeData = useMemo(() => {
        const fromData = resolveYearData(changeYearFrom);
        const toData = resolveYearData(changeYearTo);
        if (!fromData || !toData) return null;

        const result = {};
        // Use all cell IDs from the "from" dataset
        for (const cellId of Object.keys(fromData)) {
            const from = fromData[cellId];
            const to = toData[cellId];
            if (!from || !to) continue;
            const entry = {};
            for (const c of CLASSES) {
                entry[`delta_${c}`] = (to[c] ?? 0) - (from[c] ?? 0);
            }
            result[cellId] = entry;
        }
        return result;
    }, [changeYearFrom, changeYearTo, labels2020, labels2021, changeFromPred, changeToPred]);

    // Pick the right data based on view mode
    const getViewData = () => {
        switch (viewMode) {
            case 'labels':
                return selectedYear === 2020 ? labels2020 : labels2021;
            case 'predictions':
                return predictions;
            case 'change':
                return computedChangeData;
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
                appMode={appMode}
                onAppModeChange={setAppMode}
                nurembergDataMode={nurembergDataMode}
                nurembergYear={nurembergYear}
            />
            {appMode === 'deploy' ? (
                <DeployView />
            ) : (
                <div className="app-layout">
                    {sidebarOpen && (
                        <Sidebar
                            appMode={appMode}
                            models={models}
                            selectedModel={selectedModel}
                            onModelChange={setSelectedModel}
                            viewMode={viewMode}
                            onViewModeChange={setViewMode}
                            selectedYear={selectedYear}
                            onYearChange={setSelectedYear}
                            selectedClass={selectedClass}
                            onClassChange={setSelectedClass}
                            classes={NUREMBERG_CLASSES}
                            classLabels={CLASS_LABELS}
                            classColors={CLASS_COLORS}
                            labelYears={LABEL_YEARS}
                            allYears={ALL_YEARS}
                            changeYearFrom={changeYearFrom}
                            changeYearTo={changeYearTo}
                            onChangeYearFrom={setChangeYearFrom}
                            onChangeYearTo={setChangeYearTo}
                            searchCellId={searchCellId}
                            onSearchCellId={handleSearchCell}
                            nurembergResolution={nurembergResolution}
                            onResolutionChange={setNurembergResolution}
                            nurembergYear={nurembergYear}
                            onNurembergYearChange={setNurembergYear}
                            nurembergDataMode={nurembergDataMode}
                            onNurembergDataModeChange={setNurembergDataMode}
                            nurembergMeta={nurembergMeta}
                        />
                    )}
                    <NurembergMapView
                        meta={nurembergMeta}
                        boundary={nurembergBoundary}
                        selectedYear={nurembergYear}
                        selectedClass={selectedClass}
                        resolution={nurembergResolution}
                        classColors={CLASS_COLORS}
                        loading={!nurembergMeta}
                        dataMode={nurembergDataMode}
                    />
                </div>
            )}
        </>
    );
}
