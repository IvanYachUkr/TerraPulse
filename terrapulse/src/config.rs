/// Configuration constants matching the Python pipeline.
pub const N_CLASSES: usize = 6;
pub const N_FOLDS: usize = 5;

pub const CLASS_NAMES: [&str; N_CLASSES] = [
    "tree", "shrubland", "grassland",
    "cropland", "built_up", "bare_sparse_vegetation",
];
