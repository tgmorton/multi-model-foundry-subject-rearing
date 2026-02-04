"""
Query interface for annotated corpus data.

Provides a high-level API for filtering and extracting sentences
based on linguistic annotations.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import polars as pl


class AnnotatedCorpus:
    """
    Query interface for sentence-level annotations stored in Parquet.

    Supports filtering by linguistic properties, extracting specific
    sentence types, and computing statistics.

    Example usage:
        corpus = AnnotatedCorpus("annotated_corpus/train_90M.parquet")

        # Get all that-trace violations
        violations = corpus.get_that_trace_violations()

        # Get null subject sentences from CHILDES
        null_subj = corpus.filter(
            has_null_subject=True,
            genre="CHILDES_child"
        )

        # Get sentences with specific properties
        sentences = corpus.get_sentences_with(
            has_expletive=True,
            has_negation=True
        )
    """

    def __init__(
        self,
        path: Union[str, Path],
        lazy: bool = True,
    ):
        """
        Initialize corpus query interface.

        Args:
            path: Path to Parquet file or directory
            lazy: If True, use lazy evaluation (recommended for large corpora)
        """
        self.path = Path(path)
        self.lazy = lazy

        if self.path.is_file():
            self._df = pl.scan_parquet(self.path)
        elif self.path.is_dir():
            # Scan directory for parquet files
            pattern = str(self.path / "**" / "*.parquet")
            self._df = pl.scan_parquet(pattern)
        else:
            raise FileNotFoundError(f"Path does not exist: {self.path}")

    @property
    def df(self) -> pl.LazyFrame:
        """Return the underlying LazyFrame."""
        return self._df

    def _collect(self, lf: pl.LazyFrame) -> pl.DataFrame:
        """Collect a LazyFrame, respecting the lazy setting."""
        return lf.collect()

    # === High-level query methods ===

    def get_that_trace_violations(
        self,
        genre: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Get all sentences containing that-trace violations.

        Returns sentences where a complementizer "that" appears with
        subject extraction from the embedded clause.
        """
        query = self._df.filter(pl.col("has_that_trace_violation") == True)
        if genre:
            query = query.filter(pl.col("genre") == genre)
        return self._collect(query)

    def get_null_subject_sentences(
        self,
        genre: Optional[str] = None,
        finite_only: bool = True,
    ) -> pl.DataFrame:
        """
        Get sentences with null subjects.

        Args:
            genre: Optional genre filter
            finite_only: If True, only return finite clauses with null subjects
        """
        query = self._df.filter(pl.col("has_null_subject") == True)
        if genre:
            query = query.filter(pl.col("genre") == genre)
        return self._collect(query)

    def get_expletive_sentences(
        self,
        expletive_class: Optional[str] = None,
        genre: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Get sentences containing expletives.

        Args:
            expletive_class: Filter by class ("weather", "existential", "raising")
            genre: Optional genre filter
        """
        query = self._df.filter(pl.col("has_expletive") == True)
        if genre:
            query = query.filter(pl.col("genre") == genre)

        result = self._collect(query)

        if expletive_class:
            # Filter by expletive class within the nested structure
            result = result.filter(
                pl.col("expletives").list.eval(
                    pl.element().struct.field("expletive_class") == expletive_class
                ).list.any()
            )

        return result

    def get_wh_questions(
        self,
        extraction_type: Optional[str] = None,
        clause_level: Optional[str] = None,
        genre: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Get sentences with wh-extractions.

        Args:
            extraction_type: Filter by type ("subject", "object", "adjunct")
            clause_level: Filter by level ("root", "embedded")
            genre: Optional genre filter
        """
        query = self._df.filter(pl.col("has_wh_extraction") == True)
        if genre:
            query = query.filter(pl.col("genre") == genre)
        return self._collect(query)

    def get_relative_clauses(
        self,
        rel_type: Optional[str] = None,
        genre: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Get sentences containing relative clauses.

        Args:
            rel_type: Filter by type ("subject", "object")
            genre: Optional genre filter
        """
        query = self._df.filter(pl.col("has_relative_clause") == True)
        if genre:
            query = query.filter(pl.col("genre") == genre)
        return self._collect(query)

    def get_fragments(
        self,
        genre: Optional[str] = None,
    ) -> pl.DataFrame:
        """Get sentence fragments (no finite verb)."""
        query = self._df.filter(pl.col("is_fragment") == True)
        if genre:
            query = query.filter(pl.col("genre") == genre)
        return self._collect(query)

    def get_imperatives(
        self,
        genre: Optional[str] = None,
    ) -> pl.DataFrame:
        """Get imperative sentences."""
        query = self._df.filter(pl.col("is_imperative") == True)
        if genre:
            query = query.filter(pl.col("genre") == genre)
        return self._collect(query)

    # === Flexible filtering ===

    def filter(
        self,
        **conditions: Any,
    ) -> pl.DataFrame:
        """
        Filter sentences by arbitrary conditions.

        Supports boolean flags as keyword arguments:
            corpus.filter(has_null_subject=True, has_negation=True)

        Also supports genre and split:
            corpus.filter(genre="CHILDES_child", split="train_90M")
        """
        query = self._df

        for key, value in conditions.items():
            if value is not None:
                query = query.filter(pl.col(key) == value)

        return self._collect(query)

    def get_sentences_with(
        self,
        **conditions: bool,
    ) -> pl.DataFrame:
        """
        Get sentences matching all specified boolean conditions.

        Example:
            corpus.get_sentences_with(
                has_expletive=True,
                has_negation=True,
                is_fragment=False
            )
        """
        return self.filter(**conditions)

    # === Co-occurrence queries ===

    def get_cooccurrence(
        self,
        property1: str,
        property2: str,
        genre: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        Count co-occurrence of two boolean properties.

        Returns counts for all four combinations:
        - both True
        - first True, second False
        - first False, second True
        - both False
        """
        query = self._df
        if genre:
            query = query.filter(pl.col("genre") == genre)

        result = (
            query
            .group_by([property1, property2])
            .agg(pl.count().alias("count"))
            .collect()
            .to_dicts()
        )

        return {
            f"{property1}={r[property1]},{property2}={r[property2]}": r["count"]
            for r in result
        }

    # === Statistics ===

    def count_by_genre(
        self,
        property_name: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Count sentences (or property occurrences) by genre.

        Args:
            property_name: Optional boolean property to count (e.g., "has_null_subject")
        """
        if property_name:
            query = (
                self._df
                .filter(pl.col(property_name) == True)
                .group_by("genre")
                .agg(pl.count().alias("count"))
            )
        else:
            query = (
                self._df
                .group_by("genre")
                .agg(pl.count().alias("count"))
            )
        return self._collect(query.sort("count", descending=True))

    def count_by_speaker(
        self,
        property_name: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Count sentences (or property occurrences) by speaker.
        """
        if property_name:
            query = (
                self._df
                .filter(pl.col(property_name) == True)
                .group_by("speaker")
                .agg(pl.count().alias("count"))
            )
        else:
            query = (
                self._df
                .group_by("speaker")
                .agg(pl.count().alias("count"))
            )
        return self._collect(query.sort("count", descending=True))

    def summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for the corpus.
        """
        stats = (
            self._df
            .select([
                pl.count().alias("total_sentences"),
                pl.col("n_tokens").sum().alias("total_tokens"),
                pl.col("genre").n_unique().alias("n_genres"),
            ])
            .collect()
            .to_dicts()[0]
        )

        # Count boolean properties
        bool_cols = [
            "has_null_subject", "has_expletive", "has_that_trace_violation",
            "has_wh_extraction", "has_relative_clause", "has_negation",
            "is_fragment", "is_imperative",
        ]

        bool_counts = {}
        for col in bool_cols:
            try:
                count = (
                    self._df
                    .filter(pl.col(col) == True)
                    .select(pl.count())
                    .collect()
                    .item()
                )
                bool_counts[col] = count
            except Exception:
                bool_counts[col] = None

        return {
            **stats,
            "property_counts": bool_counts,
        }

    # === Sentence extraction ===

    def get_sentence_by_id(
        self,
        sentence_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get a single sentence by its ID.
        """
        result = (
            self._df
            .filter(pl.col("sentence_id") == sentence_id)
            .collect()
        )
        if result.is_empty():
            return None
        return result.to_dicts()[0]

    def get_sentences_by_ids(
        self,
        sentence_ids: List[str],
    ) -> pl.DataFrame:
        """
        Get multiple sentences by their IDs.
        """
        return self._collect(
            self._df.filter(pl.col("sentence_id").is_in(sentence_ids))
        )

    def sample(
        self,
        n: int = 100,
        seed: Optional[int] = None,
        **conditions: Any,
    ) -> pl.DataFrame:
        """
        Get a random sample of sentences.

        Args:
            n: Number of sentences to sample
            seed: Random seed for reproducibility
            conditions: Optional filter conditions
        """
        query = self._df
        for key, value in conditions.items():
            if value is not None:
                query = query.filter(pl.col(key) == value)

        # Collect first since sample doesn't work on lazy frames
        df = query.collect()
        if df.height <= n:
            return df
        return df.sample(n=n, seed=seed)


def load_layered_corpus(
    base_path: Union[str, Path],
    layers: Optional[List[str]] = None,
) -> pl.LazyFrame:
    """
    Load a layered corpus, joining base annotations with specified layers.

    Args:
        base_path: Path to annotated_corpus directory
        layers: List of layer names to join (default: all available)

    Returns:
        LazyFrame with all layers joined on sentence_id
    """
    base_path = Path(base_path)

    # Load base
    base_file = list((base_path / "base").glob("*.parquet"))
    if not base_file:
        raise FileNotFoundError(f"No base parquet files in {base_path / 'base'}")

    df = pl.scan_parquet(base_file)

    # Discover available layers
    layers_dir = base_path / "layers"
    if layers_dir.exists():
        available_layers = [d.name for d in layers_dir.iterdir() if d.is_dir()]

        if layers is None:
            layers = available_layers

        for layer in layers:
            layer_dir = layers_dir / layer
            if layer_dir.exists():
                layer_files = list(layer_dir.glob("*.parquet"))
                if layer_files:
                    layer_df = pl.scan_parquet(layer_files)
                    df = df.join(layer_df, on="sentence_id", how="left")

    return df
