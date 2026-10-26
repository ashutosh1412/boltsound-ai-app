import pytest

from app.gateway import ModelGateway
from app.providers.base import ProviderError
from app.providers.fake import DownProvider, FakeProvider


def no_wait(seconds: float) -> None:
    pass


def test_falls_back_to_the_next_provider():
    down, backup = DownProvider(), FakeProvider("hello", name="backup")
    completion = ModelGateway([down, backup], sleep=no_wait).complete("sys", "hi")
    assert completion.text == "hello"
    assert down.calls == 2  # first try + one retry
    assert completion.failures == ["down: 503 service unavailable"] * 2


def test_raises_when_every_provider_is_down():
    with pytest.raises(ProviderError, match="all providers failed"):
        ModelGateway([DownProvider("a"), DownProvider("b")], sleep=no_wait).complete("sys", "hi")


def test_does_not_retry_errors_that_will_repeat():
    class NotFound(DownProvider):
        def complete(self, system: str, prompt: str):
            self.calls += 1
            raise ProviderError("down failed: 404 NOT_FOUND. model is not found")

    retired, backup = NotFound(), FakeProvider("hello", name="backup")
    ModelGateway([retired, backup], sleep=no_wait).complete("sys", "hi")
    assert retired.calls == 1
