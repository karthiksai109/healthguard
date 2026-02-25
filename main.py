"""HealthGuard — Entry Point

Starts:
  1. FastAPI gateway server (HTTP requests, UI serving)
  2. Agent autonomous loop (60s cycle, never stops)
  3. Cleanup worker (deletes raw files after 60s TTL)

All processes run inside one container on Akash.
"""
import structlog

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(0),
)

import uvicorn
from app.core.config import get_config


def main():
    config = get_config()
    # FastAPI app handles startup (agent init, demo load) via on_event("startup")
    uvicorn.run(
        "app.gateway:app",
        host=config.host,
        port=config.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
