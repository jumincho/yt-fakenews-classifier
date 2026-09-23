"""Allow ``python -m ytfakenews`` as an alias for the ``ytfakenews`` command."""

from ytfakenews.cli import main

raise SystemExit(main())
