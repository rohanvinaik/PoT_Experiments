from pot.lm.verifier import LMVerifier


class DummyTokenizer:
    def __init__(self):
        self.vocab = {}
        self.pad_token_id = 0
        self.bos_token_id = 0
        self.eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        tokens = text.lower().replace('.', '').split()
        ids = []
        for tok in tokens:
            if tok not in self.vocab:
                self.vocab[tok] = len(self.vocab) + 1
            ids.append(self.vocab[tok])
        return ids


class DummyLM:
    def __init__(self):
        self.tok = DummyTokenizer()

    def generate(self, prompt, max_new_tokens=64):
        # Echo the prompt for simplicity
        return prompt


def test_paraphrase_vs_unrelated_distances():
    lm = DummyLM()
    verifier = LMVerifier(lm, use_sequential=False)

    paraphrase_a = "the cat sat on the mat"
    paraphrase_b = "on the mat sat the cat"
    unrelated = "i love eating pizza"

    for method in ["edit", "embedding", "fuzzy"]:
        d_para = verifier.compute_output_distance(paraphrase_a, paraphrase_b, method=method)
        d_unrel = verifier.compute_output_distance(paraphrase_a, unrelated, method=method)
        assert d_para < d_unrel, f"{method} distance should be smaller for paraphrased outputs"
