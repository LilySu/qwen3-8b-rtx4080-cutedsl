from pathlib import Path

from fastokens import Tokenizer as _Tokenizer


class Qwen3Tokenizer:
    # Qwen3 special token IDs
    EOS_ID = 151643
    IM_START_ID = 151644
    IM_END_ID = 151645

    def __init__(self, tokenizer_json_path: str | Path):
        self._tok = _Tokenizer.from_file(str(tokenizer_json_path))

    @classmethod
    def from_dir(cls, weights_dir: str | Path) -> "Qwen3Tokenizer":
        return cls(Path(weights_dir) / "tokenizer.json")

    def encode(self, text: str, add_bos: bool = False) -> list[int]:
        ids = self._tok.encode(text, add_special_tokens=False).ids
        if add_bos:
            ids = [self.IM_START_ID] + ids
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special)

    def apply_chat_template(self, messages: list[dict]) -> list[int]:
        tokens: list[int] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            header = f"<|im_start|>{role}\n"
            tokens += self._tok.encode(header, add_special_tokens=False).ids
            tokens += self._tok.encode(content, add_special_tokens=False).ids
            tokens += self._tok.encode("<|im_end|>\n", add_special_tokens=False).ids
        tokens += self._tok.encode("<|im_start|>assistant\n", add_special_tokens=False).ids
        return tokens

    @property
    def vocab_size(self) -> int:
        return self._tok.vocab_size
