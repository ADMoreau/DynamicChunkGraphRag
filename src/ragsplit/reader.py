"""LLM reader backends. The reader's own tokenizer is used for all token counts."""

import time

SYSTEM_PROMPT = (
    "You answer questions using only the given context. Reply with the shortest span "
    "copied exactly from the context that answers the question. No explanation, no full sentence."
)


def user_prompt(question: str, context: str) -> str:
    return f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"


def clean_answer(text: str) -> str:
    text = text.strip().split("\n")[0].strip()
    if text.lower().startswith("answer:"):
        text = text[len("answer:"):].strip()
    return text.strip('"').strip()


class LlamaCppReader:
    def __init__(self, cfg: dict):
        from llama_cpp import Llama

        self.cfg = cfg
        self.name = cfg["model_path"].rsplit("/", 1)[-1]
        self.llm = Llama(
            model_path=cfg["model_path"],
            vocab_only=cfg.get("vocab_only", False),  # tokenizer only, no weights
            n_ctx=cfg.get("n_ctx", 2048),
            n_threads=cfg.get("n_threads", 8),
            n_batch=cfg.get("n_batch", 512),
            seed=cfg.get("seed", 0),
            verbose=False,
        )

    def close(self) -> None:
        """Free llama.cpp explicitly; leaving it to interpreter shutdown can segfault
        when torch is also loaded (seen in m2_pilot, exit code 139)."""
        self.llm.close()

    def tokenize(self, text: str) -> list[int]:
        return self.llm.tokenize(text.encode("utf-8"), add_bos=False, special=False)

    def count(self, text: str) -> int:
        return len(self.tokenize(text))

    def truncate(self, text: str, n_tokens: int) -> str:
        toks = self.tokenize(text)
        if len(toks) <= n_tokens:
            return text
        return self.llm.detokenize(toks[:n_tokens]).decode("utf-8", errors="ignore")

    def answer(self, question: str, context: str) -> dict:
        t0 = time.perf_counter()
        out = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(question, context)},
            ],
            temperature=0.0,
            max_tokens=self.cfg.get("max_tokens", 32),
        )
        return {
            "answer": clean_answer(out["choices"][0]["message"]["content"] or ""),
            "raw_answer": out["choices"][0]["message"]["content"],
            "input_tokens": out["usage"]["prompt_tokens"],
            "output_tokens": out["usage"]["completion_tokens"],
            "reader_latency_s": time.perf_counter() - t0,
        }


def load_reader(cfg: dict):
    if cfg["backend"] == "llama_cpp":
        return LlamaCppReader(cfg)
    raise ValueError(f"unknown reader backend {cfg['backend']!r}")
