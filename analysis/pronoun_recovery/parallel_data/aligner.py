"""
awesome-align word alignment via direct Python API.

Loads the awesome-align model once and runs alignment on tokenised
EN ||| IT pairs in batches.  Bypasses the CLI's deprecated HuggingFace
config loader by using transformers + awesome_align.modeling directly.
"""

import itertools
import logging
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


class AwesomeAligner:
    """Wraps awesome-align model for batch word alignment.

    Loads the model once on construction; call :meth:`align_batch`
    repeatedly for each chunk of sentence pairs.

    Args:
        model_name: HuggingFace model ID (e.g. "aneuraz/awesome-align-with-co").
        align_layer: BERT layer to extract alignments from.
        device: torch device string; auto-detected if None.
    """

    def __init__(
        self,
        model_name: str = "aneuraz/awesome-align-with-co",
        align_layer: int = 8,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.align_layer = align_layer

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self._load_model()

    def _load_model(self) -> None:
        """Load tokenizer and model, setting global token IDs.

        Uses huggingface_hub to download/cache the model locally, then
        passes the local path to awesome-align's BertForMaskedLM
        (bypassing its deprecated S3 URL resolution).
        """
        from huggingface_hub import snapshot_download
        from awesome_align import modeling
        from awesome_align.modeling import BertForMaskedLM
        from awesome_align.tokenization_bert import BertTokenizer

        logger.info(
            "Loading awesome-align model: %s (device=%s)",
            self.model_name, self.device,
        )

        # Download model to local HF cache (or use cached version).
        local_dir = snapshot_download(self.model_name)
        logger.info("Model cached at: %s", local_dir)

        # Load via awesome-align's own classes from the local directory.
        self.tokenizer = BertTokenizer.from_pretrained(local_dir)
        self.model = BertForMaskedLM.from_pretrained(local_dir)
        self.model.to(self.device)
        self.model.eval()

        # Set global token IDs that awesome-align's modeling module expects.
        modeling.PAD_ID = self.tokenizer.pad_token_id
        modeling.CLS_ID = self.tokenizer.cls_token_id
        modeling.SEP_ID = self.tokenizer.sep_token_id

    def _tokenize_pair(
        self,
        src_words: List[str],
        tgt_words: List[str],
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]]:
        """Tokenize a single word-tokenized pair into BPE IDs + word maps.

        Returns None if either side produces only [CLS] [SEP] (empty).
        """
        token_src = [self.tokenizer.tokenize(w) for w in src_words]
        token_tgt = [self.tokenizer.tokenize(w) for w in tgt_words]

        wid_src = [self.tokenizer.convert_tokens_to_ids(x) for x in token_src]
        wid_tgt = [self.tokenizer.convert_tokens_to_ids(x) for x in token_tgt]

        max_len = getattr(self.tokenizer, "model_max_length", None) or getattr(self.tokenizer, "max_len", 512)
        ids_src = self.tokenizer.prepare_for_model(
            list(itertools.chain(*wid_src)),
            return_tensors="pt",
            max_length=max_len,
        )["input_ids"][0]
        ids_tgt = self.tokenizer.prepare_for_model(
            list(itertools.chain(*wid_tgt)),
            return_tensors="pt",
            max_length=max_len,
        )["input_ids"][0]

        # Skip if either side is just [CLS] [SEP].
        if len(ids_src) <= 2 or len(ids_tgt) <= 2:
            return None

        bpe2word_src = []
        for i, word_list in enumerate(token_src):
            bpe2word_src.extend([i] * len(word_list))

        bpe2word_tgt = []
        for i, word_list in enumerate(token_tgt):
            bpe2word_tgt.extend([i] * len(word_list))

        return ids_src, ids_tgt, bpe2word_src, bpe2word_tgt

    def align_batch(
        self,
        en_tokens_list: List[List[str]],
        it_tokens_list: List[List[str]],
        batch_size: int = 32,
    ) -> List[List[Tuple[int, int]]]:
        """Align a batch of tokenised EN-IT sentence pairs.

        Args:
            en_tokens_list: List of English word-token lists.
            it_tokens_list: List of Italian word-token lists (same length).
            batch_size: Sub-batch size for model inference.

        Returns:
            List of alignment lists, one per pair.  Each alignment is a
            list of ``(en_idx, it_idx)`` word-index tuples.
            Pairs that fail tokenization return empty lists.
        """
        assert len(en_tokens_list) == len(it_tokens_list)
        if not en_tokens_list:
            return []

        # Tokenize all pairs.
        tokenized = []
        valid_indices = []
        for i, (en_toks, it_toks) in enumerate(zip(en_tokens_list, it_tokens_list)):
            result = self._tokenize_pair(en_toks, it_toks)
            if result is not None:
                tokenized.append(result)
                valid_indices.append(i)

        # Initialize output with empty alignments for all pairs.
        all_alignments: List[List[Tuple[int, int]]] = [[] for _ in range(len(en_tokens_list))]

        # Process in sub-batches.
        for batch_start in range(0, len(tokenized), batch_size):
            batch = tokenized[batch_start : batch_start + batch_size]
            batch_valid = valid_indices[batch_start : batch_start + batch_size]

            ids_src_list = [t[0] for t in batch]
            ids_tgt_list = [t[1] for t in batch]
            bpe2word_src_list = [t[2] for t in batch]
            bpe2word_tgt_list = [t[3] for t in batch]

            ids_src_padded = pad_sequence(
                ids_src_list, batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )
            ids_tgt_padded = pad_sequence(
                ids_tgt_list, batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )

            with torch.no_grad():
                word_aligns_list = self.model.get_aligned_word(
                    ids_src_padded,
                    ids_tgt_padded,
                    bpe2word_src_list,
                    bpe2word_tgt_list,
                    self.device,
                    0, 0,
                    align_layer=self.align_layer,
                    extraction="softmax",
                    softmax_threshold=0.001,
                    test=True,
                )

            for j, (orig_idx, word_aligns) in enumerate(zip(batch_valid, word_aligns_list)):
                pairs = []
                for pair in word_aligns:
                    if pair[0] != -1:
                        pairs.append((pair[0], pair[1]))
                all_alignments[orig_idx] = pairs

        logger.info(
            "Aligned %d/%d pairs (%d skipped tokenization)",
            len(valid_indices), len(en_tokens_list),
            len(en_tokens_list) - len(valid_indices),
        )
        return all_alignments


# ── Standalone utilities (used by tests and resolution) ──────────────────


def _parse_alignment_line(line: str) -> List[Tuple[int, int]]:
    """Parse a single awesome-align output line into (en_idx, it_idx) pairs.

    awesome-align outputs space-separated ``i-j`` pairs per line.
    """
    pairs: List[Tuple[int, int]] = []
    for pair_str in line.strip().split():
        parts = pair_str.split("-")
        if len(parts) == 2:
            try:
                pairs.append((int(parts[0]), int(parts[1])))
            except ValueError:
                continue
    return pairs


def build_alignment_dicts(
    alignments: List[Tuple[int, int]],
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """Convert a list of (en_idx, it_idx) pairs into bidirectional dicts.

    Returns:
        (en_to_it, it_to_en) where each maps a token index to a list
        of aligned indices on the other side.
    """
    en_to_it: Dict[int, List[int]] = {}
    it_to_en: Dict[int, List[int]] = {}

    for en_idx, it_idx in alignments:
        en_to_it.setdefault(en_idx, []).append(it_idx)
        it_to_en.setdefault(it_idx, []).append(en_idx)

    return en_to_it, it_to_en
