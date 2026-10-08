"""Command-line interface: ``ytfakenews COMMAND [OPTIONS]``.

- :mod:`ytfakenews.cli.parser` defines the commands and their options;
- :mod:`ytfakenews.cli.commands` carries out a parsed command;
- :mod:`ytfakenews.cli.output` formats what the commands print.

Exit codes: 0 on success, 1 for an error the user can fix (reported in one line on
stderr), 2 for invalid arguments and 130 when interrupted.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Sequence

from ytfakenews.cli.parser import build_parser
from ytfakenews.errors import YTFakeNewsError

__all__ = ["build_parser", "main"]

_package_logger = logging.getLogger("ytfakenews")


class _LogFormatter(logging.Formatter):
    """Plain messages for progress; ``warning: ...``/``error: ...`` prefixes otherwise."""

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        if record.levelno >= logging.WARNING:
            return f"{record.levelname.lower()}: {message}"
        return message


def _configure_logging(*, verbose: bool, quiet: bool) -> None:
    """Send the package's log records to the current ``sys.stderr``, one line each."""
    # main() can run several times in one process (the tests do): replace the handler
    # of an earlier run, which writes to the sys.stderr of that run.
    for handler in list(_package_logger.handlers):
        if isinstance(handler.formatter, _LogFormatter):
            _package_logger.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_LogFormatter("%(message)s"))
    _package_logger.addHandler(handler)
    _package_logger.setLevel(
        logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    )
    _package_logger.propagate = False


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``ytfakenews`` console script; returns the exit code."""
    args = build_parser().parse_args(argv)
    _configure_logging(verbose=args.verbose, quiet=args.quiet)
    try:
        return int(args.handler(args))
    except YTFakeNewsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
