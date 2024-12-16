import json
import logging
import re
from collections.abc import Callable
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

""""desc: 1. split a paragraph into a list of chunks recursively by a list of separators"""
"""       2. optimize the chunks by the defined chunk size and chunk overlapping """
"""       3. the chunk length is usually measured LLM tokens"""


class RecursiveCharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        separators: Optional[list[str]] = None,
        length_function: Callable[[str], int] = len,
        keep_separator: bool = False,
        add_start_index: bool = False,
    ) -> None:
        if chunk_overlap > chunk_size:
            raise ValueError(
                f'Got a larger chunk overlap ({chunk_overlap}) than chunk size ' f'({chunk_size}), should be smaller.'
            )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function
        self._keep_separator = keep_separator
        self._add_start_index = add_start_index
        self._separators = separators or [
            '\n\n',
            '\n',
            ' ',
            '',
        ]

    def _split_text(
        self,
        text: str,
        separators: list[str],
    ) -> list[str]:
        """Split incoming text and return chunks."""

        final_chunks = []
        # Get appropriate separator to use
        separator = separators[-1]
        new_separators = []
        for (
            i,
            _s,
        ) in enumerate(separators):
            if _s == '':
                separator = _s
                break
            if re.search(
                _s,
                text,
            ):
                separator = _s
                new_separators = separators[i + 1 :]
                break

        splits = _split_text_with_regex(
            text,
            separator,
            self._keep_separator,
        )

        # Now go merging things, recursively splitting longer texts.
        _good_splits = []
        _separator = '' if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(
                        _good_splits,
                        _separator,
                    )
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(
                        s,
                        new_separators,
                    )
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(
                _good_splits,
                _separator,
            )
            final_chunks.extend(merged_text)
        return final_chunks

    def split_text(
        self,
        text: str,
    ) -> list[str]:
        return self._split_text(
            text,
            self._separators,
        )

    def _merge_splits(self, splits: Iterable[str], separator: str) -> list[str]:
        """merge splits by separator"""
        """return an optimized chunks by chunk size and overlapping"""

        # We now want to combine these smaller pieces into medium size
        # chunks to send to the LLM.
        separator_len = self._length_function(separator)

        docs = []
        current_doc: list[str] = []
        total = 0
        for d in splits:
            _len = self._length_function(d)
            if total + _len + (separator_len if len(current_doc) > 0 else 0) > self._chunk_size:
                if total > self._chunk_size:
                    logger.warning(
                        f'Created a chunk of size {total}, ' f'which is longer than the specified {self._chunk_size}'
                    )
                if len(current_doc) > 0:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    # Keep on popping if:
                    # - we have a larger chunk than in the chunk overlap
                    # - or if we still have any chunks and the length is long
                    while total > self._chunk_overlap or (
                        total + _len + (separator_len if len(current_doc) > 0 else 0) > self._chunk_size and total > 0
                    ):
                        total -= self._length_function(current_doc[0]) + (separator_len if len(current_doc) > 1 else 0)
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += _len + (separator_len if len(current_doc) > 1 else 0)
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)

        return docs

    def _join_docs(self, docs: list[str], separator: str) -> Optional[str]:
        text = separator.join(docs)
        text = text.strip()
        if text == '':
            return None
        else:
            return text


def _split_text_with_regex(text: str, separator: str, keep_separator: bool) -> list[str]:
    # Now that we have the separator, split the text
    if separator:
        if keep_separator:
            # The parentheses in the pattern keep the delimiters in the result.
            _splits = re.split(f'({re.escape(separator)})', text)

            splits = [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]

            if len(_splits) % 2 == 0:
                splits += _splits[-1:]
            splits = [_splits[0]] + splits
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != '']
