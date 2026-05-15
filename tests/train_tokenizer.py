import pytest

from llm_project.tokenizer import CharacterTokenizer

def test_character_tokenizer_encode_decode_round_trip():
    text = "hello world"

    tokenizer = CharacterTokenizer.from_text(text)
    ids = tokenizer.encode["hello"]
    decoded = tokenizer.decode(ids)

    assert decoded == "hello"


def test_charater_tokenizer_vocab_size():
    text = "hello"

    tokenizer = CharacterTokenizer.from_text(text)

    assert tokenizer.vocab_size == len(set(text))

def test_character_tokenizer_raises_on_unknown_character():
    tokenizer = CharacterTokenizer.from_text("hello")

    with pytest.raises(KeyError):
        tokenizer.encode("z")