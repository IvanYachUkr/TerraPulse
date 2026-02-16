use anyhow::{Context, Result};
use std::path::Path;

/// Read a feature parquet file and return (column_names, data_matrix).
/// data_matrix is row-major: [n_cells, n_features].
pub fn read_feature_parquet(path: &Path) -> Result<(Vec<String>, Vec<Vec<f32>>)> {
    use parquet::file::reader::FileReader;
    use parquet::file::serialized_reader::SerializedFileReader;
    use parquet::record::RowAccessor;

    let file = std::fs::File::open(path)
        .with_context(|| format!("Cannot open parquet: {}", path.display()))?;
    let reader = SerializedFileReader::new(file)?;

    let metadata = reader.metadata();
    let schema = metadata.file_metadata().schema();
    let n_cols = schema.get_fields().len();

    // Get column names
    let col_names: Vec<String> = schema
        .get_fields()
        .iter()
        .map(|f| f.name().to_string())
        .collect();

    // Read all rows
    let mut rows: Vec<Vec<f32>> = Vec::new();
    let iter = reader.get_row_iter(None)?;
    for record in iter {
        let record = record?;
        let mut row = Vec::with_capacity(n_cols);
        for i in 0..n_cols {
            let val = match record.get_double(i) {
                Ok(v) => v as f32,
                Err(_) => match record.get_float(i) {
                    Ok(v) => v,
                    Err(_) => match record.get_long(i) {
                        Ok(v) => v as f32,
                        Err(_) => match record.get_int(i) {
                            Ok(v) => v as f32,
                            Err(_) => f32::NAN,
                        },
                    },
                },
            };
            row.push(val);
        }
        rows.push(row);
    }

    Ok((col_names, rows))
}

/// Write predictions to a parquet file.
/// predictions: [n_cells, n_classes] row-major.
pub fn write_predictions_parquet(
    path: &Path,
    class_names: &[&str],
    predictions: &[Vec<f32>],
    model_name: &str,
) -> Result<()> {
    use arrow::array::Float32Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let n_cells = predictions.len();
    let n_classes = class_names.len();

    // Build schema: cell_id + class columns
    let mut fields = vec![Field::new("cell_id", DataType::Int32, false)];
    for cn in class_names {
        fields.push(Field::new(
            format!("{}_{}", cn, model_name),
            DataType::Float32,
            false,
        ));
    }
    let schema = Arc::new(Schema::new(fields));

    // Build arrays
    let cell_ids: Vec<i32> = (0..n_cells as i32).collect();
    let cell_id_array = Arc::new(arrow::array::Int32Array::from(cell_ids));

    let mut columns: Vec<Arc<dyn arrow::array::Array>> = vec![cell_id_array];
    for ci in 0..n_classes {
        let vals: Vec<f32> = predictions.iter().map(|row| row[ci]).collect();
        columns.push(Arc::new(Float32Array::from(vals)));
    }

    let batch = RecordBatch::try_new(schema.clone(), columns)?;

    let file = std::fs::File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;

    Ok(())
}

/// Write features to a parquet file.
/// extra_cols/extra_data: metadata columns (cell_id, valid_fraction).
/// feature_cols: feature column names.
/// rows: [n_cells][n_features] feature data.
pub fn write_feature_parquet(
    path: &Path,
    extra_cols: &[String],
    extra_data: &[Vec<f32>],
    feature_cols: &[String],
    rows: &[Vec<f32>],
) -> Result<()> {
    use arrow::array::Float32Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;

    let n_cells = rows.len();
    let mut fields = Vec::new();
    let mut arrays: Vec<Arc<dyn arrow::array::Array>> = Vec::new();

    // Extra columns first
    for (i, name) in extra_cols.iter().enumerate() {
        fields.push(Field::new(name, DataType::Float32, false));
        arrays.push(Arc::new(Float32Array::from(extra_data[i].clone())));
    }

    // Feature columns
    for (ci, name) in feature_cols.iter().enumerate() {
        fields.push(Field::new(name, DataType::Float32, true));
        let vals: Vec<f32> = (0..n_cells).map(|ri| rows[ri][ci]).collect();
        arrays.push(Arc::new(Float32Array::from(vals)));
    }

    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), arrays)?;

    let file = std::fs::File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;

    Ok(())
}
