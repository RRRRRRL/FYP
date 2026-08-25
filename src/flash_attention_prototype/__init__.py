from importlib import import_module
from typing import Any


def flash_attention_forward(*args, **kwargs):
	from .kernel import flash_attention_forward as triton_attention

	return triton_attention(*args, **kwargs)


def __getattr__(name: str) -> Any:
	if name in {
		"reference_attention",
		"reference_attention_backward",
		"reference_attention_delta",
		"reference_attention_forward",
	}:
		return getattr(import_module(".reference", __name__), name)
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
	"flash_attention_forward",
	"reference_attention",
	"reference_attention_backward",
	"reference_attention_delta",
	"reference_attention_forward",
]
