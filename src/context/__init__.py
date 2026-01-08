"""Context management operations"""

from .manager import (
    get_context,
    save_context,
    search_similar_contexts,
    delete_context,
    persist_schema_to_chroma,
    set_context,
    add_table_to_context,
    remove_table_from_context,
    clear_context,
)

from .persistence import (
    persist_current_context_marker,
    load_persisted_context_to_session,
)

__all__ = [
    "get_context",
    "save_context",
    "search_similar_contexts",
    "delete_context",
    "persist_schema_to_chroma",
    "set_context",
    "add_table_to_context",
    "remove_table_from_context",
    "clear_context",
    "persist_current_context_marker",
    "load_persisted_context_to_session",
]
