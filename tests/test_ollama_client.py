from src.llm.ollama_client import extract_json


def test_extract_json_clean():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced_with_prose():
    text = (
        "Sure! Here is the breakdown:\n"
        '```json\n{"sub_questions": [{"id": "sq1", "text": "who invented X", '
        '"requires_graph": true}]}\n```\n'
        "Let me know if you need anything else!"
    )
    result = extract_json(text)
    assert result is not None
    assert result["sub_questions"][0]["id"] == "sq1"


def test_extract_json_no_fence_embedded_in_prose():
    text = 'The answer is {"requires_graph": true, "reasoning": "needs lookup"} as shown.'
    assert extract_json(text) == {"requires_graph": True, "reasoning": "needs lookup"}


def test_extract_json_returns_none_on_garbage():
    assert extract_json("I think the answer is yes, definitely.") is None
    assert extract_json("") is None
    assert extract_json("{broken json here") is None
