from app.feedback import implicit_signal


def test_frustration():
    assert implicit_signal("Is the X1 waterproof?", "That's wrong, get me a human") == "frustrated"


def test_rephrased_question():
    assert implicit_signal("how long is the refund", "how long is the refund for my X2") == "rephrased"


def test_a_new_topic_is_not_feedback():
    assert implicit_signal("Is the X1 waterproof?", "Where is my order BS-2291?") is None
