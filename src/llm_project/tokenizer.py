class CharacterTokenizer:
    """문자 단위 tokenizer.
    예 :
        "hello" -> [2, 1, 3, 3, 4]
        [2, 1, 3, 3, 4] -> "hello"
    """
    def __init__(self, chars: list[str]) -> None:
        self.chars = sorted(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharacterTokenizer":
        chars = sorted(list(set(text)))
        return cls(chars)


    @property
    # 모델의 입력 가능한 문자 종류 개수
    def vocab_size(self) -> int:
        return len(self.chars)
    
    # JSON 요청을 DB 저장용 ID로 바꾸는 seializer
    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]
    
    # DB ID를 다시 사용자에게 보여줄 문자열로 바꾸는 deserializer
    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)