from app.assistant import Assistant
from app.config import Settings
from app.providers.fake import FakeProvider

SETTINGS = Settings("", "", "fake", price_in_per_m=1.0, price_out_per_m=1.0)


def test_answer_returns_model_text_and_trace():
    assistant = Assistant(FakeProvider("Hello from Bolt"), SETTINGS)
    reply = assistant.answer("hi")
    assert reply.text == "Hello from Bolt"
    assert [step.name for step in reply.trace] == ["model"]
    assert reply.cost_usd > 0
