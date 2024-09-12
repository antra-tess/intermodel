from intermodel.callgpt import format_messages


def test_basic():
    string = "Hi"
    assert format_messages(string, "user") == [{"role": "user", "content": string}]


def test_turns():
    string = "Hi<|eot_id|>Hello!"
    assert format_messages(string, "user") == [
        {"role": "user", "content": "Hi"},
        {"role": "user", "content": "Hello!"},
    ]


def test_roles():
    string = (
        "<|im_start|>system<|im_sep|>You are a helpful assistant<|im_end|><|im_start|>user<|im_sep|>What is the "
        "color of the sky?<|im_end|><|im_start|>assistant<|im_sep|>"
    )
    assert format_messages(string, "user") == [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "What is the color of the sky?"},
    ]


def test_names():
    string = (
        "<|im_start|>system<|im_sep|>You are a helpful assistant<|im_end|><|name_start|>Alice<|name_end|><|im_start|>user<|im_sep|>What is the "
        "color of the sky?<|im_end|><|im_start|>assistant<|im_sep|>"
    )
    assert format_messages(string, "user") == [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "name": "Alice", "content": "What is the color of the sky?"},
    ]
