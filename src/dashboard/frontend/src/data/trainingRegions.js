/**
 * Training / Validation / Test city bounding boxes.
 * Generated from mlp/config.py — used to overlay regions on the global map.
 *
 * Each entry: { name, bbox: [west, south, east, north], role: 'train'|'val'|'test' }
 * Split: 87 train, 23 val, 11 test
 */

export const TRAINING_REGIONS = [
    // === TRAINING CITIES (87) ===
    { name: 'Bremen', bbox: [8.65, 53.0, 8.9, 53.14], role: 'train' },
    { name: 'Hamburg', bbox: [9.8, 53.4, 10.15, 53.58], role: 'train' },
    { name: 'Düsseldorf', bbox: [6.7, 51.15, 6.9, 51.28], role: 'train' },
    { name: 'Leipzig', bbox: [12.25, 51.27, 12.5, 51.4], role: 'train' },
    { name: 'Amsterdam', bbox: [4.75, 52.3, 4.95, 52.4], role: 'train' },
    { name: 'Hambach Mine', bbox: [6.4, 50.85, 6.6, 50.98], role: 'train' },
    { name: 'Welzow Mine', bbox: [14.1, 51.5, 14.35, 51.65], role: 'train' },
    { name: 'Salzburg', bbox: [12.95, 47.73, 13.15, 47.87], role: 'train' },
    { name: 'Malmö', bbox: [12.9, 55.53, 13.15, 55.68], role: 'train' },
    { name: 'London', bbox: [-0.25, 51.42, 0.05, 51.56], role: 'train' },
    { name: 'Brussels', bbox: [4.3, 50.8, 4.5, 50.92], role: 'train' },
    { name: 'Vienna', bbox: [16.3, 48.15, 16.55, 48.28], role: 'train' },
    { name: 'Zurich', bbox: [8.45, 47.32, 8.65, 47.44], role: 'train' },
    { name: 'Munich North', bbox: [11.45, 48.22, 11.7, 48.36], role: 'train' },
    { name: 'Stuttgart', bbox: [9.1, 48.72, 9.3, 48.84], role: 'train' },
    { name: 'Innsbruck', bbox: [11.3, 47.22, 11.55, 47.35], role: 'train' },
    { name: 'Kraków', bbox: [19.85, 50.02, 20.1, 50.14], role: 'train' },
    { name: 'Budapest', bbox: [19.0, 47.42, 19.2, 47.55], role: 'train' },
    { name: 'Bratislava', bbox: [17.05, 48.1, 17.25, 48.22], role: 'train' },
    { name: 'Copenhagen', bbox: [12.48, 55.62, 12.68, 55.74], role: 'train' },
    { name: 'Gothenburg', bbox: [11.9, 57.65, 12.1, 57.78], role: 'train' },
    { name: 'Barcelona', bbox: [2.05, 41.32, 2.3, 41.45], role: 'train' },
    { name: 'Lisbon', bbox: [-9.2, 38.68, -8.95, 38.82], role: 'train' },
    { name: 'Rome', bbox: [12.4, 41.82, 12.6, 41.95], role: 'train' },
    { name: 'Milan', bbox: [9.1, 45.42, 9.35, 45.55], role: 'train' },
    { name: 'Lyon', bbox: [4.75, 45.7, 4.95, 45.82], role: 'train' },
    { name: 'Toulouse', bbox: [1.35, 43.55, 1.55, 43.68], role: 'train' },
    { name: 'Athens', bbox: [23.65, 37.92, 23.85, 38.05], role: 'train' },
    { name: 'Almería Coast', bbox: [-2.5, 36.78, -2.3, 36.9], role: 'train' },
    { name: 'Central Hungary', bbox: [19.5, 47.1, 19.75, 47.24], role: 'train' },
    { name: 'Finnish Lakeland', bbox: [27.5, 61.8, 27.8, 61.95], role: 'train' },
    { name: 'Swedish Forest', bbox: [15.3, 57.3, 15.55, 57.45], role: 'train' },
    { name: 'Scottish Highlands', bbox: [-5.2, 57.05, -4.95, 57.18], role: 'train' },
    { name: 'Sicily Interior', bbox: [14.1, 37.4, 14.35, 37.53], role: 'train' },
    { name: 'Carpathian Romania', bbox: [24.6, 45.5, 24.85, 45.65], role: 'train' },
    { name: 'Danish Farmland', bbox: [9.8, 55.3, 10.05, 55.45], role: 'train' },
    { name: 'Dublin', bbox: [-6.35, 53.3, -6.15, 53.42], role: 'train' },
    { name: 'Naples', bbox: [14.18, 40.8, 14.38, 40.93], role: 'train' },
    { name: 'Valencia', bbox: [-0.45, 39.42, -0.25, 39.55], role: 'train' },
    { name: 'Oslo', bbox: [10.65, 59.87, 10.85, 60.0], role: 'train' },
    { name: 'Gdańsk', bbox: [18.55, 54.32, 18.75, 54.45], role: 'train' },
    { name: 'Castilla Meseta', bbox: [-3.1, 39.2, -2.85, 39.35], role: 'train' },
    { name: 'Extremadura Dehesa', bbox: [-6.15, 39.1, -5.9, 39.25], role: 'train' },
    { name: 'Aragón Steppe', bbox: [-0.7, 41.05, -0.45, 41.2], role: 'train' },
    { name: 'Murcia Drylands', bbox: [-1.6, 38.0, -1.35, 38.15], role: 'train' },
    { name: 'Tabernas Desert', bbox: [-2.4, 37.0, -2.15, 37.15], role: 'train' },
    { name: 'Bardenas Reales', bbox: [-1.57, 42.13, -1.32, 42.27], role: 'train' },
    { name: 'Sardinia Maquis', bbox: [9.0, 39.8, 9.25, 39.95], role: 'train' },
    { name: 'Crete Phrygana', bbox: [24.8, 35.2, 25.05, 35.35], role: 'train' },
    { name: 'Thessaly Scrubland', bbox: [22.3, 39.5, 22.55, 39.65], role: 'train' },
    { name: 'Thrace Steppe', bbox: [26.4, 41.0, 26.65, 41.15], role: 'train' },
    { name: 'El Ejido Greenhouses', bbox: [-2.94, 36.7, -2.69, 36.84], role: 'train' },
    { name: 'Skåne Fields', bbox: [13.4, 55.7, 13.65, 55.85], role: 'train' },
    { name: 'Trøndelag Farmland', bbox: [10.3, 63.35, 10.55, 63.5], role: 'train' },
    { name: 'Latvian Farmland', bbox: [24.0, 56.85, 24.25, 57.0], role: 'train' },
    { name: 'Lithuanian Lowland', bbox: [23.8, 55.6, 24.05, 55.75], role: 'train' },
    { name: 'Finnish Coastal Farm', bbox: [24.0, 60.4, 24.25, 60.55], role: 'train' },
    { name: 'Lapland Tundra', bbox: [27.07, 68.28, 27.32, 68.42], role: 'train' },
    { name: 'Galicia Pastures', bbox: [-8.6, 42.8, -8.35, 42.95], role: 'train' },
    { name: 'Brittany Bocage', bbox: [-3.4, 48.1, -3.15, 48.25], role: 'train' },
    { name: 'Wales Upland', bbox: [-3.9, 52.0, -3.65, 52.15], role: 'train' },
    { name: 'Les Landes Forest', bbox: [-0.93, 44.08, -0.68, 44.22], role: 'train' },
    { name: 'Hortobágy Puszta', bbox: [21.02, 47.53, 21.27, 47.67], role: 'train' },
    { name: 'Wallachian Steppe', bbox: [25.5, 44.2, 25.75, 44.35], role: 'train' },
    { name: 'Thracian Farmland', bbox: [25.0, 42.1, 25.25, 42.25], role: 'train' },
    { name: 'Camargue Wetland', bbox: [4.4, 43.4, 4.65, 43.55], role: 'train' },
    { name: 'Wadden Tidal', bbox: [8.0, 53.55, 8.25, 53.7], role: 'train' },
    { name: 'Danube Delta', bbox: [29.32, 44.98, 29.57, 45.12], role: 'train' },
    { name: 'Pyrenees Meadows', bbox: [0.4, 42.6, 0.65, 42.75], role: 'train' },
    { name: 'Norwegian Fjord', bbox: [7.0, 61.5, 7.25, 61.65], role: 'train' },
    { name: 'Carpathian Alpine', bbox: [24.5, 47.5, 24.75, 47.65], role: 'train' },
    { name: 'Swiss Alps High', bbox: [7.62, 45.93, 7.88, 46.07], role: 'train' },
    { name: 'Foggia Wheat', bbox: [15.43, 41.38, 15.68, 41.52], role: 'train' },
    { name: 'Jutland Farmland', bbox: [8.82, 56.18, 9.07, 56.32], role: 'train' },
    { name: 'Doñana Marshes', bbox: [-6.5, 36.9, -6.25, 37.05], role: 'train' },
    { name: 'Finnish Bog', bbox: [26.0, 63.5, 26.25, 63.65], role: 'train' },
    { name: 'Mecklenburg Lakes', bbox: [12.6, 53.4, 12.85, 53.55], role: 'train' },
    { name: 'Danube Floodplain', bbox: [18.8, 47.8, 19.05, 47.95], role: 'train' },
    { name: 'Cretan Coast', bbox: [25.1, 35.3, 25.35, 35.45], role: 'train' },
    { name: 'Cyprus Troodos', bbox: [32.8, 34.85, 33.05, 35.0], role: 'train' },
    { name: 'Dalmatian Coast', bbox: [15.4, 43.85, 15.65, 44.0], role: 'train' },
    { name: 'Greek Maquis', bbox: [22.1, 37.5, 22.35, 37.65], role: 'train' },
    { name: 'Schwarzwald Edge', bbox: [7.8, 47.9, 8.05, 48.05], role: 'train' },
    { name: 'Algarve Coast', bbox: [-8.2, 37.0, -7.95, 37.15], role: 'train' },
    { name: 'Central Finland Bog', bbox: [25.3, 62.4, 25.55, 62.55], role: 'train' },
    { name: 'SW Ireland Heath', bbox: [-10.0, 51.7, -9.75, 51.85], role: 'train' },
    { name: 'Andalusia Sierra', bbox: [-3.6, 36.9, -3.35, 37.05], role: 'train' },

    // === VALIDATION CITIES (23) ===
    { name: 'Rostock', bbox: [12.0, 54.05, 12.2, 54.18], role: 'val' },
    { name: 'Paris South', bbox: [2.25, 48.75, 2.5, 48.89], role: 'val' },
    { name: 'Berlin', bbox: [13.3, 52.45, 13.55, 52.58], role: 'val' },
    { name: 'Helsinki', bbox: [24.85, 60.13, 25.1, 60.27], role: 'val' },
    { name: 'Madrid', bbox: [-3.8, 40.35, -3.55, 40.48], role: 'val' },
    { name: 'Alentejo Portugal', bbox: [-7.9, 38.1, -7.65, 38.25], role: 'val' },
    { name: 'Peloponnese Rural', bbox: [22.0, 37.4, 22.25, 37.55], role: 'val' },
    { name: 'Po Valley Rural', bbox: [10.8, 44.9, 11.05, 45.05], role: 'val' },
    { name: 'Dutch Polders', bbox: [5.1, 52.55, 5.35, 52.7], role: 'val' },
    { name: 'Marseille', bbox: [5.3, 43.25, 5.5, 43.38], role: 'val' },
    { name: 'Bordeaux', bbox: [-0.65, 44.8, -0.45, 44.93], role: 'val' },
    { name: 'Corsica Interior', bbox: [9.1, 41.85, 9.35, 42.0], role: 'val' },
    { name: 'Estonian Plains', bbox: [25.5, 58.5, 25.75, 58.65], role: 'val' },
    { name: 'Iceland Highlands', bbox: [-19.48, 64.13, -19.23, 64.27], role: 'val' },
    { name: 'Ireland Bog Pasture', bbox: [-7.8, 53.2, -7.55, 53.35], role: 'val' },
    { name: 'Vojvodina Cropland', bbox: [20.2, 45.3, 20.45, 45.45], role: 'val' },
    { name: 'Jaén Olives', bbox: [-3.92, 37.78, -3.67, 37.92], role: 'val' },
    { name: 'Ebro Delta', bbox: [0.65, 40.6, 0.9, 40.75], role: 'val' },
    { name: 'Andalusia Olives', bbox: [-4.2, 37.5, -3.95, 37.65], role: 'val' },
    { name: 'Central Spain Plateau', bbox: [-3.5, 40.6, -3.25, 40.75], role: 'val' },
    { name: 'Uppland Farmland', bbox: [17.6, 59.6, 17.85, 59.75], role: 'val' },
    { name: 'Northern Sweden', bbox: [19.4, 68.3, 19.65, 68.45], role: 'val' },
    { name: 'Dresden', bbox: [13.65, 50.98, 13.9, 51.12], role: 'val' },

    // === TEST CITIES (11) ===
    { name: 'Munich', bbox: [11.45, 48.08, 11.7, 48.22], role: 'test' },
    { name: 'Nuremberg', bbox: [10.95, 49.38, 11.2, 49.52], role: 'test' },
    { name: 'Warsaw', bbox: [20.9, 52.15, 21.15, 52.3], role: 'test' },
    { name: 'Prague', bbox: [14.35, 50.02, 14.55, 50.15], role: 'test' },
    { name: 'Seville', bbox: [-6.05, 37.32, -5.85, 37.45], role: 'test' },
    { name: 'Stockholm', bbox: [17.95, 59.28, 18.2, 59.42], role: 'test' },
    // Extra test cities (from eval_test / stress_test)
    { name: 'Ankara', bbox: [32.65, 39.85, 32.95, 40.05], role: 'test' },
    { name: 'Sofia', bbox: [23.20, 42.62, 23.45, 42.77], role: 'test' },
    { name: 'Riga', bbox: [23.95, 56.88, 24.25, 57.05], role: 'test' },
    { name: 'Edinburgh', bbox: [-3.35, 55.88, -3.05, 56.02], role: 'test' },
    { name: 'Palermo', bbox: [13.25, 38.05, 13.45, 38.20], role: 'test' },
];

/** Colour scheme per role — [R, G, B] */
export const REGION_COLORS = {
    train: { fill: [66, 135, 245, 35], stroke: [66, 135, 245, 180], label: 'Training', hex: '#4287f5' },
    val:   { fill: [245, 166, 35, 35], stroke: [245, 166, 35, 180], label: 'Validation', hex: '#f5a623' },
    test:  { fill: [168, 85, 247, 35], stroke: [168, 85, 247, 180], label: 'Test', hex: '#a855f7' },
};

/** Convert a bbox [west, south, east, north] to a GeoJSON Polygon feature */
function bboxToFeature(region) {
    const [west, south, east, north] = region.bbox;
    return {
        type: 'Feature',
        geometry: {
            type: 'Polygon',
            coordinates: [[
                [west, south], [east, south],
                [east, north], [west, north],
                [west, south],
            ]],
        },
        properties: { name: region.name, role: region.role },
    };
}

/** Pre-built GeoJSON FeatureCollection for all training regions */
export const REGIONS_GEOJSON = {
    type: 'FeatureCollection',
    features: TRAINING_REGIONS.map(bboxToFeature),
};
