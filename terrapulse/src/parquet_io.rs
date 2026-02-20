use anyhow::{Context, Result};
use std::path::Path;

/// Read a feature parquet file and return (column_names, data_matrix).
/// data_matrix is row-major: [n_cells, n_features].
///
/// Uses Arrow RecordBatch reader for columnar access (much faster than row iteration).
pub fn read_feature_parquet(path: &Path) -> Result<(Vec<String>, Vec<Vec<f32>>)> {
    use arrow::array::{Array, AsArray};
    use arrow::datatypes::*;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let file = std::fs::File::open(path)
        .with_context(|| format!("Cannot open parquet: {}", path.display()))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema().clone();
    let n_cols = schema.fields().len();
    let col_names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();

    let reader = builder.build()?;

    // Read all batches and accumulate rows
    let mut rows: Vec<Vec<f32>> = Vec::new();

    for batch_result in reader {
        let batch = batch_result?;
        let n_rows = batch.num_rows();
        let batch_start = rows.len();

        // Extend rows for this batch
        rows.resize_with(batch_start + n_rows, || vec![0.0f32; n_cols]);

        // Read each column and fill into rows (columnar -> row-major)
        for col_idx in 0..n_cols {
            let col = batch.column(col_idx);

            match col.data_type() {
                DataType::Float32 => {
                    let arr = col.as_primitive::<Float32Type>();
                    for r in 0..n_rows {
                        rows[batch_start + r][col_idx] = if arr.is_null(r) {
                            f32::NAN
                        } else {
                            arr.value(r)
                        };
                    }
                }
                DataType::Float64 => {
                    let arr = col.as_primitive::<Float64Type>();
                    for r in 0..n_rows {
                        rows[batch_start + r][col_idx] = if arr.is_null(r) {
                            f32::NAN
                        } else {
                            arr.value(r) as f32
                        };
                    }
                }
                DataType::Int64 => {
                    let arr = col.as_primitive::<Int64Type>();
                    for r in 0..n_rows {
                        rows[batch_start + r][col_idx] = if arr.is_null(r) {
                            f32::NAN
                        } else {
                            arr.value(r) as f32
                        };
                    }
                }
                DataType::Int32 => {
                    let arr = col.as_primitive::<Int32Type>();
                    for r in 0..n_rows {
                        rows[batch_start + r][col_idx] = if arr.is_null(r) {
                            f32::NAN
                        } else {
                            arr.value(r) as f32
                        };
                    }
                }
                _ => {
                    // Unknown type, fill with NaN
                    for r in 0..n_rows {
                        rows[batch_start + r][col_idx] = f32::NAN;
                    }
                }
            }
        }
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
