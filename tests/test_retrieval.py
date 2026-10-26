from app.retrieval import load_chunks


def test_chunks_are_bullets_with_their_document_title(tmp_path):
    (tmp_path / "returns.md").write_text("# Returns\n\n- Within 30 days.\n- Original box.\n")
    chunks = load_chunks(tmp_path)
    assert [chunk.text for chunk in chunks] == ["Returns: Within 30 days.", "Returns: Original box."]
    assert {chunk.source for chunk in chunks} == {"returns.md"}
