"""
Logger configuration for the app.

- Provides a standard logging setup for workflows, scripts, and tests.
- Initialize once at the start of the application using "setup_logging()".
- Log initialize with command "log = structlog.get_logger(__name__)" in each file required logging.
"""

import logging
import logging.handlers
import sys
from pathlib import Path

import structlog


def _get_shared_processors():
    """
    Processors that enrich the log event dict before it hits the renderer.
    """
    return [
        structlog.contextvars.merge_contextvars,  # Thread-local/async context variables
        structlog.stdlib.add_logger_name,        # Logger name
        structlog.stdlib.add_log_level,          # "info", "warning" etc.
        structlog.stdlib.ExtraAdder(),           # Integrates stdlib extra= dict
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),  # Cleaner time format for local dev
        structlog.processors.CallsiteParameterAdder(
            [
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
                structlog.processors.CallsiteParameter.FUNC_NAME,
            ]
        ),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        structlog.processors.format_exc_info,   # Properly formats tracebacks
    ]

def _configure_stdlib_logging(
    level: int,
    shared_processors: list,
    log_file_name: str,
    root_path: Path,
) -> None:
    """
    Bridges structlog into the stdlib logging system using local-friendly formatters.
    """
    handlers: list[logging.Handler] = []

    # 1. Terminal Handler (sys.stdout) - Sa uključenim bojama
    stream_formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(colors=True),  # Obojen ispis na ekranu
        ],
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(stream_formatter)
    handlers.append(stream_handler)

    # 2. File Handler - Isti čitljiv format, ali BEZ ANSI kodova za boje
    log_dir = root_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    file_formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(colors=False, exception_formatter=structlog.dev.plain_traceback),
        ],
    )
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_dir / log_file_name,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(file_formatter)
    handlers.append(file_handler)

    # Primena konfiguracije
    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True,
    )

    _suppress_noisy_loggers()

def _suppress_noisy_loggers() -> None:
    """
    Mute overly talkative libraries during local development.
    """
    noisy: dict[str, int] = {
        "httpx": logging.WARNING,
        "httpcore": logging.WARNING,
        "temporalio": logging.WARNING,
        "asyncio": logging.WARNING,
        "alembic.runtime.migration": logging.ERROR,
    }
    for name, lvl in noisy.items():
        logging.getLogger(name).setLevel(lvl)


def setup_logging(
    log_file_name: str = "app.log",
    log_level: int = logging.INFO,
) -> None:
    """
    Initialize the local-only structlog + stdlib pipeline.
    Call once at the entry point of your app.
    """
    # Više ne uvozimo Settings jer nam ne treba provera okruženja
    from retrieval_lab.config.settings import ROOT_DIR

    shared_processors = _get_shared_processors()

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _configure_stdlib_logging(
        level=log_level,
        shared_processors=shared_processors,
        log_file_name=log_file_name,
        root_path=ROOT_DIR,
    )