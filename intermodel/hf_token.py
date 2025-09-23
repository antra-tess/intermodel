import typing

if typing.TYPE_CHECKING:
    from tokenizers import Tokenizer

hf_tokenizers = {}


def get_hf_tokenizer(hf_name) -> "Tokenizer":
    if hf_name in hf_tokenizers:
        return hf_tokenizers[hf_name]
    else:
        from tokenizers import Tokenizer
        import warnings

        try:
            # The tokenizers library doesn't use auth_token parameter
            # Authentication is handled through huggingface-cli login
            hf_tokenizers[hf_name] = Tokenizer.from_pretrained(hf_name)
            return hf_tokenizers[hf_name]
        except Exception as e:
            # Fallback to GPT-2 tokenizer
            warnings.warn(
                f"Failed to load tokenizer for '{hf_name}': {str(e)}. "
                f"Falling back to GPT-2 tokenizer. This may affect tokenization accuracy."
            )
            # Try to load GPT-2 tokenizer
            try:
                gpt2_tokenizer = Tokenizer.from_pretrained("gpt2")
                hf_tokenizers[hf_name] = gpt2_tokenizer
                return gpt2_tokenizer
            except Exception as gpt2_error:
                # If even GPT-2 fails, re-raise the original error
                raise e


def get_hf_auth_token():
    import huggingface_hub

    try:
        get_token = huggingface_hub.get_token
    except AttributeError:
        from intermodel.hf_auth_token import get_token
    return get_token()
