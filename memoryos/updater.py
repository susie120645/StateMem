try:
    from .services.ingestion_service import Updater, mem_log
except ImportError:
    from services.ingestion_service import Updater, mem_log


__all__ = ["Updater", "mem_log"]
