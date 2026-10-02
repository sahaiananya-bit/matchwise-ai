"""Bounded, in-memory resume extraction without upload files on disk."""

from __future__ import annotations

import importlib
import io
import os
from pathlib import Path
from typing import BinaryIO
import xml.etree.ElementTree as ET
import zipfile
import zlib

from config import MAX_UPLOAD_SIZE as MAX_FILE_BYTES, MAX_PDF_PAGES, MAX_TEXT_CHARS
MAX_DOCX_ENTRIES = 1_000
MAX_DOCX_UNCOMPRESSED_BYTES = 20 * 1024 * 1024
MAX_DOCX_XML_BYTES = 5 * 1024 * 1024
MAX_DOCX_COMPRESSION_RATIO = 200
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}
_WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


class InputError(ValueError):
    """An uploaded document cannot be safely read or contains no usable text."""


def _read_source(source: bytes | os.PathLike | BinaryIO, filename: str | None):
    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        filename = filename or path.name
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                raise InputError("The file exceeds the 5 MB limit. Upload a smaller file.")
            with path.open("rb") as stream:
                data = stream.read(MAX_FILE_BYTES + 1)
        except InputError:
            raise
        except (OSError, ValueError) as exc:
            raise InputError("The document could not be opened. Choose a readable file.") from exc
    elif isinstance(source, (bytes, bytearray, memoryview)):
        if len(source) > MAX_FILE_BYTES:
            raise InputError("The file exceeds the 5 MB limit. Upload a smaller file.")
        data = bytes(source)
    elif hasattr(source, "read"):
        filename = filename or getattr(source, "name", None)
        original_position = None
        try:
            try:
                original_position = source.tell()
                source.seek(0)
            except (AttributeError, OSError, io.UnsupportedOperation):
                pass
            data = source.read(MAX_FILE_BYTES + 1)
        except (OSError, ValueError) as exc:
            raise InputError("The document could not be read. Upload the file again.") from exc
        finally:
            if original_position is not None:
                try:
                    source.seek(original_position)
                except (AttributeError, OSError, ValueError, io.UnsupportedOperation):
                    pass
    else:
        raise InputError("Choose a PDF, DOCX, or UTF-8 TXT file.")
    if not isinstance(data, bytes):
        raise InputError("Read the document as bytes, or upload a PDF, DOCX, or TXT file.")
    if len(data) > MAX_FILE_BYTES:
        raise InputError("The file exceeds the 5 MB limit. Upload a smaller file.")
    if not data:
        raise InputError("The uploaded file is empty. Choose a document with resume text.")
    return data, filename


def _bounded_text(text: str) -> str:
    if len(text) > MAX_TEXT_CHARS:
        raise InputError("The document exceeds the 50,000-character text limit. Shorten it and try again.")
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise InputError("No readable text was found. Upload a document containing text.")
    return text


def _extract_pdf(data: bytes) -> str:
    if not data.lstrip().startswith(b"%PDF-"):
        raise InputError("This file is not a valid PDF. Export it as a PDF again, or use DOCX or TXT.")
    try:
        pdfplumber = importlib.import_module("pdfplumber")
    except ImportError as exc:
        raise InputError("PDF support is unavailable. Install the project dependencies or upload DOCX or TXT.") from exc
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            if len(pdf.pages) > MAX_PDF_PAGES:
                raise InputError("The PDF exceeds the 30-page limit. Upload only the resume pages.")
            parts = []
            char_count = 0
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                char_count += len(page_text) + (1 if parts else 0)
                if char_count > MAX_TEXT_CHARS:
                    raise InputError("The document exceeds the 50,000-character text limit. Shorten it and try again.")
                parts.append(page_text)
    except InputError:
        raise
    except Exception as exc:
        error_kind = f"{type(exc).__name__} {exc}".lower()
        if "password" in error_kind or "encrypt" in error_kind:
            raise InputError("This PDF is password-protected. Upload an unlocked copy.") from exc
        raise InputError("The PDF could not be read. It may be damaged; export it again or use DOCX or TXT.") from exc
    text = "\n".join(parts)
    if not text.strip():
        raise InputError("No readable text was found in the PDF. It may be scanned; run OCR first or upload DOCX or TXT.")
    return _bounded_text(text)


class _SafeTreeBuilder(ET.TreeBuilder):
    def doctype(self, name, pubid, system):
        raise InputError("The DOCX contains an unsupported XML document type. Export a new DOCX or TXT copy.")


def _extract_docx(data: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            if len(entries) > MAX_DOCX_ENTRIES:
                raise InputError("The DOCX has too many embedded files. Export a simpler DOCX or TXT copy.")
            if sum(entry.file_size for entry in entries) > MAX_DOCX_UNCOMPRESSED_BYTES:
                raise InputError("The DOCX expands beyond the safe document size limit. Export a smaller DOCX or TXT copy.")
            names = [entry.filename for entry in entries]
            if len(set(names)) != len(names):
                raise InputError("The DOCX contains duplicate archive entries. Export a new DOCX or TXT copy.")
            for entry in entries:
                if entry.flag_bits & 1:
                    raise InputError("The DOCX is encrypted. Upload an unlocked copy.")
                if entry.file_size > MAX_DOCX_COMPRESSION_RATIO * max(entry.compress_size, 1):
                    raise InputError("The DOCX has an unsafe compression ratio. Export a simpler DOCX or TXT copy.")
            if "word/document.xml" not in names:
                raise InputError("This file is not a valid DOCX. Export it as DOCX or TXT and try again.")
            headers = sorted(n for n in names if n.startswith("word/header") and n.endswith(".xml") and n.count("/") == 1)
            footers = sorted(n for n in names if n.startswith("word/footer") and n.endswith(".xml") and n.count("/") == 1)
            parts = []
            char_count = 0
            for name in [*headers, "word/document.xml", *footers]:
                if archive.getinfo(name).file_size > MAX_DOCX_XML_BYTES:
                    raise InputError("A DOCX text section exceeds the safe size limit. Export a simpler document.")
                with archive.open(name) as member:
                    xml_data = member.read(MAX_DOCX_XML_BYTES + 1)
                if len(xml_data) > MAX_DOCX_XML_BYTES:
                    raise InputError("A DOCX text section exceeds the safe size limit. Export a simpler document.")
                root = ET.fromstring(xml_data, parser=ET.XMLParser(target=_SafeTreeBuilder()))
                # Paragraph iteration preserves body, table-cell, header, and footer order.
                for paragraph in root.iter(_WORD_NS + "p"):
                    text_nodes = []
                    for node in paragraph.iter():
                        if node.tag == _WORD_NS + "t" and node.text:
                            text_nodes.append(node.text)
                        elif node.tag == _WORD_NS + "tab":
                            text_nodes.append("\t")
                        elif node.tag in (_WORD_NS + "br", _WORD_NS + "cr"):
                            text_nodes.append("\n")
                    paragraph_text = "".join(text_nodes)
                    char_count += len(paragraph_text) + (1 if parts else 0)
                    if char_count > MAX_TEXT_CHARS:
                        raise InputError("The document exceeds the 50,000-character text limit. Shorten it and try again.")
                    parts.append(paragraph_text)
    except InputError:
        raise
    except (zipfile.BadZipFile, ET.ParseError, OSError, EOFError, RuntimeError, NotImplementedError, ValueError, zlib.error) as exc:
        raise InputError("The DOCX could not be read. It may be damaged; export it again or upload TXT.") from exc
    return _bounded_text("\n".join(parts))


def extract_text(source: bytes | os.PathLike | BinaryIO, filename: str | None = None) -> str:
    """Extract PDF, DOCX, or UTF-8 TXT from bytes, a path, or a binary stream.

    A supplied filename determines the format. Otherwise stream/path names or
    PDF/DOCX signatures are used. Seekable streams retain their original position.
    Uploads never leave memory; scanned PDFs require OCR before upload.
    """
    data, filename = _read_source(source, filename)
    extension = Path(str(filename)).suffix.lower() if filename else ""
    if not extension:
        extension = ".pdf" if data.lstrip().startswith(b"%PDF-") else ".docx" if data.startswith(b"PK\x03\x04") else ".txt"
    if extension not in SUPPORTED_EXTENSIONS:
        raise InputError("Unsupported file type. Upload a PDF, DOCX, or UTF-8 TXT file.")
    if extension == ".pdf":
        return _extract_pdf(data)
    if extension == ".docx":
        return _extract_docx(data)
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InputError("The TXT file is not UTF-8 text. Save it with UTF-8 encoding and try again.") from exc
    if any(ord(character) < 32 and character not in "\t\n\r" for character in text):
        raise InputError("The TXT file contains binary data. Upload a plain UTF-8 text document.")
    return _bounded_text(text)
