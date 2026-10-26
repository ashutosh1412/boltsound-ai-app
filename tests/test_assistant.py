from app.assistant import Assistant, build_prompt
from app.config import Settings
from app.providers.fake import FakeProvider
from app.retrieval import Chunk

SETTINGS = Settings("", "", "fake", price_in_per_m=1.0, price_out_per_m=1.0)
RETURNS = Chunk("returns.md", "Returns within 30 days.")


class FakeRetriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks

    def search(self, question: str) -> list[Chunk]:
        return self.chunks


def test_answer_uses_retrieved_extracts():
    provider = FakeProvider("Yes, within 30 days. [returns.md]")
    reply = Assistant(provider, SETTINGS, FakeRetriever([RETURNS])).answer("Can I return it?")
    assert "30 days" in provider.calls[0][1]
    assert reply.text.endswith("[returns.md]")
    assert [step.name for step in reply.trace] == ["retrieval", "model"]


def test_build_prompt_labels_each_source():
    assert "[returns.md]" in build_prompt("hi", [RETURNS])
