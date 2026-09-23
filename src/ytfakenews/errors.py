"""Exception types raised by ytfakenews.

Every error that is caused by user input or the environment (a missing file, a
malformed dataset, an invalid model directory, ...) derives from
:class:`YTFakeNewsError`. The CLI prints these as a one-line message instead of a
traceback.
"""


class YTFakeNewsError(Exception):
    """Base class for expected, user-facing errors."""


class DatasetError(YTFakeNewsError):
    """The training dataset is missing or malformed."""


class ModelLoadError(YTFakeNewsError):
    """A model directory is missing, incomplete or of an unknown type."""


class TranscriptionError(YTFakeNewsError):
    """Downloading a video or transcribing its audio failed."""


class MissingDependencyError(YTFakeNewsError, ImportError):
    """A feature needs an optional extra (``asr`` or ``transformer``) that is not installed."""

    def __init__(self, *, module: str, extra: str, reason: str = "") -> None:
        self.module = module
        self.extra = extra
        detail = f": {reason}" if reason else ""
        super().__init__(
            f"this feature needs the optional '{extra}' dependencies "
            f"({module} could not be imported{detail}). Install them with\n"
            f'  pip install "ytfakenews[{extra}] @ '
            'git+https://github.com/jumincho/yt-fakenews-classifier"\n'
            f'or, inside a clone of the repository: pip install -e ".[{extra}]"'
        )
