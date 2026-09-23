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
