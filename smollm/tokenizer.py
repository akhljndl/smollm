import re


class SmilesTokenizer:
    PATTERN = r'Cl|Br|@@|\[.*?\]|[CNOSPFIcnosp()=\#\-\+\.\/\\@0-9]'
    SPECIAL = ["<PAD>", "<BOS>", "<EOS>"]

    def __init__(self) -> None:
        self.token2id: dict[str, int] = {}
        self.id2token: dict[int, str] = {}

    def fit(self, smiles_list: list[str]) -> "SmilesTokenizer":
        tokens: set[str] = set()
        for smi in smiles_list:
            tokens.update(re.findall(self.PATTERN, smi))
        all_tokens = self.SPECIAL + sorted(tokens)
        self.token2id = {t: i for i, t in enumerate(all_tokens)}
        self.id2token = {i: t for t, i in self.token2id.items()}
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    @property
    def pad_id(self) -> int:
        return self.token2id["<PAD>"]

    @property
    def bos_id(self) -> int:
        return self.token2id["<BOS>"]

    @property
    def eos_id(self) -> int:
        return self.token2id["<EOS>"]

    def encode(self, smi: str) -> list[int]:
        tokens = re.findall(self.PATTERN, smi)
        return [self.bos_id] + [self.token2id[t] for t in tokens] + [self.eos_id]

    def decode(self, ids: list[int]) -> str:
        return "".join(
            self.id2token.get(i, "")
            for i in ids
            if self.id2token.get(i, "") not in self.SPECIAL
        )

    def state_dict(self) -> dict:
        return {"token2id": self.token2id, "id2token": self.id2token}

    @classmethod
    def from_state_dict(cls, state: dict) -> "SmilesTokenizer":
        tok = cls()
        tok.token2id = state["token2id"]
        tok.id2token = {int(k): v for k, v in state["id2token"].items()}
        return tok
