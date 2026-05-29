"""
Shared file-type helpers.

Centralises MIME-type mapping and extension categorisation so that
local_tools.py, upload handlers, and any future modules all use the
same single source of truth.
"""

MIME_TYPES: dict[str, str] = {
    ".pdf":  "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc":  "application/msword",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls":  "application/vnd.ms-excel",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".png":  "image/png",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif":  "image/gif",
    ".txt":  "text/plain",
    ".md":   "text/markdown",
    ".csv":  "text/csv",
    ".json": "application/json",
    ".html": "text/html",
    ".xml":  "application/xml",
    ".zip":  "application/zip",
}

# Extensions that are plain text and can be read / displayed directly.
TEXT_EXTENSIONS: frozenset[str] = frozenset({
    ".txt", ".md", ".json", ".csv", ".py", ".html",
    ".css", ".js", ".xml", ".yaml", ".yml",
})

# Extensions that must be transmitted as binary (base64) to external services.
BINARY_EXTENSIONS: frozenset[str] = frozenset({
    ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx",
    ".png", ".jpg", ".jpeg", ".gif", ".zip",
})


def get_mime_type(ext: str) -> str:
    """Return the MIME type for *ext* (e.g. '.pdf'), defaulting to octet-stream."""
    return MIME_TYPES.get(ext.lower(), "application/octet-stream")


def is_text(ext: str) -> bool:
    return ext.lower() in TEXT_EXTENSIONS


def is_binary(ext: str) -> bool:
    return ext.lower() in BINARY_EXTENSIONS
