/// Configuration constants matching the Python pipeline.
pub const N_CLASSES: usize = 6;
pub const N_FOLDS: usize = 5;
pub const GRID_PITCH: usize = 10; // pixels per cell side

pub const CLASS_NAMES: [&str; N_CLASSES] = [
    "tree", "shrubland", "grassland",
    "cropland", "built_up", "bare_sparse_vegetation",
];

/// Seasons used for compositing.
pub const SEASONS: [&str; 3] = ["spring", "summer", "autumn"];
