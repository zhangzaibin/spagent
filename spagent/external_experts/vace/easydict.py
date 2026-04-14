"""Lightweight fallback implementation of easydict.EasyDict.

This local module is used when pip installation is unavailable in offline
environments. It supports attribute-style access for dict keys.
"""


class EasyDict(dict):
    """Dictionary with attribute-style access."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.update(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = self._convert(value)

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setitem__(self, key, value):
        super().__setitem__(key, self._convert(value))

    def update(self, *args, **kwargs):
        other = dict(*args, **kwargs)
        for key, value in other.items():
            self[key] = value

    @classmethod
    def _convert(cls, value):
        if isinstance(value, dict) and not isinstance(value, EasyDict):
            return cls(value)
        if isinstance(value, list):
            return [cls._convert(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._convert(item) for item in value)
        return value
