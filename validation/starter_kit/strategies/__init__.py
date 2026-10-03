"""Strategy modules for validation/starter_kit/run_in_starter.py, in the starter kit's convention.

Each module is loaded by its bare name (``--strategy <module>``) with the kit's own loader and must define a
module-level ``STRATEGY_CLASS`` (a ``bt.Strategy`` subclass). Optional module-level attributes, each a constant
or a callable taking the parsed ``--params`` dict:

* ``SYMBOLS``       default symbol list when ``--symbols`` is not given
* ``EXEC``          "next_open", "next_close", "mixed" or "coc" (cheat-on-close: the harness enables it)
* ``NEEDS_MARGIN``  True to run on the margin account (leverage-10 commission schemes) without ``--margin``
* ``CASH``          starting cash (default 1e9 for modules in this folder)
* ``IS_START``      first day of the in-sample metrics window (default: first day of the run)

Modules here shadow same-named examples of the kit.
"""
