"""
Output formatters for corpus descriptive analysis.

Produces JSON (per-analyzer + combined), CSV (flat, R-friendly), and
Parquet (sentence-level annotations) output.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq


def save_results(
    results: Dict[str, Dict[str, Any]],
    metadata: Dict[str, Any],
    output_dir: Path,
    split_name: str,
) -> None:
    """
    Save all analyzer results as JSON and CSV.

    Args:
        results: {analyzer_name: analyzer.get_results()} mapping
        metadata: Processing metadata dict
        output_dir: Output directory
        split_name: Split identifier (added as column in CSVs)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-analyzer JSON
    for name, data in results.items():
        json_path = output_dir / f"{name}.json"
        json_path.write_text(json.dumps(data, indent=2, default=str))

    # Combined results JSON
    combined_path = output_dir / "results.json"
    combined_path.write_text(json.dumps(results, indent=2, default=str))

    # Metadata JSON
    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))

    # Per-analyzer CSV
    for name, data in results.items():
        csv_path = output_dir / f"{name}.csv"
        _write_analyzer_csv(csv_path, name, data, split_name)


def _write_analyzer_csv(
    path: Path,
    analyzer_name: str,
    data: Dict[str, Any],
    split_name: str,
) -> None:
    """Write a single analyzer's results as a flat CSV with genre column."""
    rows = _flatten_results(analyzer_name, data, split_name)
    if not rows:
        return

    # Collect all unique fieldnames across all rows (heterogeneous dicts)
    seen = set()
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _flatten_results(
    analyzer_name: str,
    data: Dict[str, Any],
    split_name: str,
) -> List[Dict[str, Any]]:
    """
    Flatten nested analyzer results into flat rows suitable for CSV.

    Each row gets a 'split' and 'genre' column.
    """
    rows: List[Dict[str, Any]] = []

    # Overall
    overall = data.get("overall", {})
    rows.extend(_flatten_section(overall, "overall", split_name, analyzer_name))

    # By genre
    by_genre = data.get("by_genre", {})
    for genre, genre_data in by_genre.items():
        rows.extend(_flatten_section(genre_data, genre, split_name, analyzer_name))

    return rows


def _flatten_section(
    section: Dict[str, Any],
    genre: str,
    split_name: str,
    analyzer_name: str,
) -> List[Dict[str, Any]]:
    """Flatten one section (overall or a specific genre) into rows."""
    rows: List[Dict[str, Any]] = []

    for key, value in section.items():
        if isinstance(value, list):
            # List of dicts — each becomes a row
            for item in value:
                if isinstance(item, dict):
                    row = {"split": split_name, "genre": genre}
                    row.update(item)
                    rows.append(row)
        elif isinstance(value, dict):
            # Single dict — one row
            row = {"split": split_name, "genre": genre, "metric": key}
            row.update(value)
            rows.append(row)
        else:
            # Scalar — one row
            rows.append({
                "split": split_name,
                "genre": genre,
                "metric": key,
                "value": value,
            })

    return rows


# === Parquet Annotation Writer ===


class AnnotationWriter:
    """
    Streaming Parquet writer for sentence-level annotations.

    Buffers annotations in memory and writes in batches to minimize I/O.
    Supports both single-file and partitioned output.
    """

    def __init__(
        self,
        output_path: Path,
        schema: pa.Schema,
        batch_size: int = 10000,
        partition_cols: Optional[List[str]] = None,
    ):
        """
        Initialize the annotation writer.

        Args:
            output_path: Path to output Parquet file or directory (if partitioned)
            schema: PyArrow schema for annotations
            batch_size: Number of annotations to buffer before writing
            partition_cols: Columns to partition by (e.g., ["split", "genre"])
        """
        self.output_path = Path(output_path)
        self.schema = schema
        self.batch_size = batch_size
        self.partition_cols = partition_cols
        self._buffer: List[Dict[str, Any]] = []
        self._writer: Optional[pq.ParquetWriter] = None
        self._total_written = 0
        self.logger = logging.getLogger("corpus_descriptives.output")

        # Create output directory if partitioned
        if partition_cols:
            self.output_path.mkdir(parents=True, exist_ok=True)
        else:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def write_annotation(self, annotation: Dict[str, Any]) -> None:
        """
        Add a single annotation to the buffer.

        Automatically flushes when buffer reaches batch_size.
        """
        self._buffer.append(annotation)
        if len(self._buffer) >= self.batch_size:
            self._flush()

    def write_batch(self, annotations: List[Dict[str, Any]]) -> None:
        """
        Add multiple annotations to the buffer.

        More efficient than calling write_annotation in a loop.
        """
        self._buffer.extend(annotations)
        if len(self._buffer) >= self.batch_size:
            self._flush()

    def _flush(self) -> None:
        """Write buffered annotations to Parquet."""
        if not self._buffer:
            return

        # Convert to PyArrow table
        try:
            table = pa.Table.from_pylist(self._buffer, schema=self.schema)
        except (pa.ArrowInvalid, pa.ArrowTypeError) as e:
            # Try with permissive schema inference if strict fails
            self.logger.warning(f"Schema mismatch, using inferred schema: {e}")
            table = pa.Table.from_pylist(self._buffer)

        if self.partition_cols:
            # Write to partitioned dataset
            pq.write_to_dataset(
                table,
                root_path=self.output_path,
                partition_cols=self.partition_cols,
                existing_data_behavior="overwrite_or_ignore",
            )
        else:
            # Write to single file
            if self._writer is None:
                self._writer = pq.ParquetWriter(
                    self.output_path,
                    self.schema,
                    compression="snappy",
                )
            self._writer.write_table(table)

        self._total_written += len(self._buffer)
        self.logger.debug(f"Wrote {len(self._buffer)} annotations (total: {self._total_written})")
        self._buffer = []

    def close(self) -> None:
        """Flush remaining buffer and close the writer."""
        self._flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        self.logger.info(f"Annotation writer closed. Total annotations: {self._total_written}")

    def __enter__(self) -> "AnnotationWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            # Exception in flight — discard buffer and close writer without
            # flushing to avoid producing a valid parquet with partial data.
            self._buffer = []
            if self._writer is not None:
                self._writer.close()
                self._writer = None
                # Remove the incomplete file so it's obvious re-annotation is needed
                if self.output_path.exists():
                    self.output_path.unlink()
            self.logger.warning(
                f"Annotation writer aborted due to {exc_type.__name__}, "
                f"removed incomplete output at {self.output_path}"
            )
        else:
            self.close()

    @property
    def total_written(self) -> int:
        """Return total number of annotations written (including buffer)."""
        return self._total_written + len(self._buffer)


class LayeredAnnotationWriter:
    """
    Writer for incremental layer storage architecture.

    Writes base annotations and separate layer files that can be
    joined on sentence_id for queries.
    """

    def __init__(
        self,
        output_dir: Path,
        split_name: str,
        batch_size: int = 10000,
        file_suffix: Optional[str] = None,
    ):
        """
        Initialize layered writer.

        Args:
            output_dir: Root directory for annotated corpus
            split_name: Split identifier (e.g., "train_90M")
            batch_size: Batch size for each layer writer
            file_suffix: If set, output files become {split_name}_{file_suffix}.parquet
        """
        self.output_dir = Path(output_dir)
        self.split_name = split_name
        self.batch_size = batch_size
        self.file_suffix = file_suffix
        self._layer_writers: Dict[str, AnnotationWriter] = {}
        self._layer_buffers: Dict[str, List[Dict[str, Any]]] = {}
        self.logger = logging.getLogger("corpus_descriptives.output")

        # Create directory structure
        (self.output_dir / "base").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "layers").mkdir(parents=True, exist_ok=True)

    def _get_layer_writer(self, layer_name: str, schema: pa.Schema) -> AnnotationWriter:
        """Get or create writer for a specific layer."""
        if layer_name not in self._layer_writers:
            if self.file_suffix:
                fname = f"{self.split_name}_{self.file_suffix}.parquet"
            else:
                fname = f"{self.split_name}.parquet"

            if layer_name == "base":
                path = self.output_dir / "base" / fname
            else:
                layer_dir = self.output_dir / "layers" / layer_name
                layer_dir.mkdir(parents=True, exist_ok=True)
                path = layer_dir / fname

            self._layer_writers[layer_name] = AnnotationWriter(
                output_path=path,
                schema=schema,
                batch_size=self.batch_size,
            )
        return self._layer_writers[layer_name]

    def write_base(
        self,
        annotation: Dict[str, Any],
        base_schema: pa.Schema,
    ) -> None:
        """Write base annotation (identifiers, text, tokens)."""
        writer = self._get_layer_writer("base", base_schema)
        writer.write_annotation(annotation)

    def write_layer(
        self,
        layer_name: str,
        annotation: Dict[str, Any],
        layer_schema: pa.Schema,
    ) -> None:
        """Write annotation to a specific layer."""
        writer = self._get_layer_writer(layer_name, layer_schema)
        writer.write_annotation(annotation)

    def close(self) -> None:
        """Close all layer writers."""
        for name, writer in self._layer_writers.items():
            writer.close()
            self.logger.info(f"Layer '{name}' written: {writer.total_written} annotations")
        self._layer_writers = {}

    def __enter__(self) -> "LayeredAnnotationWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            # Exception in flight — close each layer writer without flushing
            # and remove incomplete files to avoid corrupt parquet output.
            for name, writer in self._layer_writers.items():
                writer._buffer = []
                if writer._writer is not None:
                    writer._writer.close()
                    writer._writer = None
                if writer.output_path.exists():
                    writer.output_path.unlink()
            self._layer_writers = {}
            self.logger.warning(
                f"Layered writer aborted due to {exc_type.__name__}, "
                f"removed incomplete outputs under {self.output_dir}"
            )
        else:
            self.close()

    def write_metadata(self, metadata: Dict[str, Any]) -> None:
        """Write metadata.json to the output directory."""
        meta_path = self.output_dir / "metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2, default=str))
