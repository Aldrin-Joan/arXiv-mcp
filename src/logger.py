"""
Centralized structured logging for arxiv-mcp.
All modules import `get_logger` from here.
"""
import logging
import sys
import structlog


def configure_logging(level: str = "INFO") -> None:
    """Call once at startup to configure structlog + stdlib logging."""
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=getattr(logging, level.upper(), logging.INFO),
    )

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "arxiv-mcp") -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
