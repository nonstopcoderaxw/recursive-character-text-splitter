import json

import pytest

from recursive_character_text_splitter.recursive_character_text_splitter import RecursiveCharacterTextSplitter


# @pytest.mark.skip
def test_RecursiveCharacterTextSplitter():
    chunk_size = 1000
    chunk_overlap = 20
    separators = [
        '\n\n',
        '\n',
        ' ',
        '',
    ]
    length_function = len  ### change to llm token count
    keep_separator = True
    add_start_index = True

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=length_function,
        keep_separator=keep_separator,
        add_start_index=add_start_index,
    )

    with open('tests/data/QA_2.txt', 'r') as file:
        content = file.read()
        print('before split', content)

        chunks = splitter.split_text(content)
        print('==============')
        print('after split', json.dumps(chunks, indent=4))
        print('after split: number of chunks', len(chunks))
