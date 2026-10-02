"""Boundary and error-path checks for untrusted resume uploads."""

import io
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch
import zipfile

from src.documents import InputError, MAX_FILE_BYTES, MAX_TEXT_CHARS, extract_text


def make_docx(document=None, extra=None, compression=zipfile.ZIP_STORED):
    if document is None:
        document = (
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            '<w:body><w:p><w:r><w:t>Python and PyTorch</w:t></w:r></w:p>'
            '<w:tbl><w:tr><w:tc><w:p><w:r><w:t>Model serving</w:t></w:r></w:p>'
            '</w:tc></w:tr></w:tbl></w:body></w:document>'
        )
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=compression) as archive:
        archive.writestr("word/document.xml", document)
        for name, content in (extra or {}).items():
            archive.writestr(name, content)
    return stream.getvalue()


def fake_pdf(*page_texts):
    pdf = MagicMock()
    pdf.pages = [types.SimpleNamespace(extract_text=lambda text=text: text) for text in page_texts]
    pdf.__enter__.return_value = pdf
    return types.SimpleNamespace(open=MagicMock(return_value=pdf))


class DocumentTests(unittest.TestCase):
    def test_utf8_bom_and_line_endings(self):
        self.assertEqual(extract_text(b"\xef\xbb\xbfPython\r\nSQL\rAI", "resume.TXT"), "Python\nSQL\nAI")

    def test_source_file_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.txt"
            path.write_text("Python engineer", encoding="utf-8")
            self.assertEqual(extract_text(path), "Python engineer")

    def test_large_path_reports_size_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.txt"
            path.write_bytes(b"x" * (MAX_FILE_BYTES + 1))
            with self.assertRaisesRegex(InputError, "5 MB"):
                extract_text(path)

    def test_binary_stream_is_read_from_start_and_position_restored(self):
        stream = io.BytesIO(b"Python engineer")
        stream.seek(7)
        self.assertEqual(extract_text(stream, "resume.txt"), "Python engineer")
        self.assertEqual(stream.tell(), 7)

    def test_unnamed_bytes_infer_format(self):
        self.assertIn("Python", extract_text(make_docx()))
        self.assertEqual(extract_text(b"Python engineer"), "Python engineer")

    def test_missing_path_reports_readable_error(self):
        with self.assertRaisesRegex(InputError, "could not be opened"):
            extract_text(Path("not-present-resume.txt"))

    def test_unsupported_and_empty_inputs(self):
        for source, name in ((b"text", "resume.exe"), (b"", "resume.txt"), (b" \n", "resume.txt")):
            with self.subTest(name=name, source=source), self.assertRaises(InputError):
                extract_text(source, name)

    def test_size_and_character_limits(self):
        with self.assertRaisesRegex(InputError, "5 MB"):
            extract_text(b"x" * (MAX_FILE_BYTES + 1), "resume.txt")
        with self.assertRaisesRegex(InputError, "50,000"):
            extract_text(b"x" * (MAX_TEXT_CHARS + 1), "resume.txt")
        self.assertEqual(len(extract_text(b"x" * MAX_TEXT_CHARS, "resume.txt")), MAX_TEXT_CHARS)

    def test_stream_reads_only_bounded_bytes(self):
        stream = MagicMock(spec=io.BytesIO)
        stream.read.return_value = b"x" * (MAX_FILE_BYTES + 1)
        with self.assertRaisesRegex(InputError, "5 MB"):
            extract_text(stream, "resume.txt")
        stream.read.assert_called_once_with(MAX_FILE_BYTES + 1)

    def test_txt_rejects_binary_and_invalid_encoding(self):
        for data in (b"Python\x00", b"Python\xff"):
            with self.subTest(data=data), self.assertRaises(InputError):
                extract_text(data, "resume.txt")

    def test_text_stream_requires_binary_input(self):
        with self.assertRaisesRegex(InputError, "bytes"):
            extract_text(io.StringIO("Python"), "resume.txt")

    def test_docx_reads_tables_and_header(self):
        header = '<w:hdr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:p><w:r><w:t>Candidate</w:t></w:r></w:p></w:hdr>'
        text = extract_text(make_docx(extra={"word/header1.xml": header}), "resume.docx")
        self.assertEqual(text, "Candidate\nPython and PyTorch\nModel serving")

    def test_docx_malformed_zip_and_xml(self):
        for data in (b"not a zip", make_docx("<document>")):
            with self.subTest(data=data[:20]), self.assertRaisesRegex(InputError, "could not be read"):
                extract_text(data, "resume.docx")

    def test_docx_rejects_non_word_zip(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("other.xml", "<root/>")
        with self.assertRaisesRegex(InputError, "not a valid DOCX"):
            extract_text(stream.getvalue(), "resume.docx")

    def test_docx_rejects_dtd_even_in_utf16(self):
        xml = '<!DOCTYPE document [<!ENTITY secret "unsafe">]><document>&secret;</document>'
        for document in (xml, xml.encode("utf-16")):
            with self.subTest(encoding=type(document).__name__), self.assertRaisesRegex(InputError, "document type"):
                extract_text(make_docx(document), "resume.docx")

    def test_docx_rejects_excessive_compression_ratio(self):
        data = make_docx("A" * 100_000, compression=zipfile.ZIP_DEFLATED)
        with self.assertRaisesRegex(InputError, "compression ratio"):
            extract_text(data, "resume.docx")

    def test_docx_rejects_total_expanded_size(self):
        with patch("src.documents.MAX_DOCX_UNCOMPRESSED_BYTES", 50):
            with self.assertRaisesRegex(InputError, "expands"):
                extract_text(make_docx(), "resume.docx")

    def test_docx_rejects_too_many_entries(self):
        with patch("src.documents.MAX_DOCX_ENTRIES", 1):
            with self.assertRaisesRegex(InputError, "too many"):
                extract_text(make_docx(extra={"extra.bin": b"x"}), "resume.docx")

    def test_docx_rejects_large_xml_part(self):
        with patch("src.documents.MAX_DOCX_XML_BYTES", 50):
            with self.assertRaisesRegex(InputError, "text section"):
                extract_text(make_docx(), "resume.docx")

    def test_docx_rejects_duplicate_archive_names(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("word/document.xml", "<root/>")
            with self.assertWarns(UserWarning):
                archive.writestr("word/document.xml", "<root/>")
        with self.assertRaisesRegex(InputError, "duplicate"):
            extract_text(stream.getvalue(), "resume.docx")

    def test_docx_rejects_excess_text(self):
        text = 'x' * (MAX_TEXT_CHARS + 1)
        document = '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:p><w:r><w:t>' + text + '</w:t></w:r></w:p></w:document>'
        with self.assertRaisesRegex(InputError, "50,000"):
            extract_text(make_docx(document), "resume.docx")

    def test_pdf_blank_page_is_skipped_without_type_error(self):
        parser = fake_pdf(None, "Python engineer")
        with patch("src.documents.importlib.import_module", return_value=parser):
            self.assertEqual(extract_text(b"%PDF-1.4 fake", "resume.pdf"), "Python engineer")
        self.assertIsInstance(parser.open.call_args.args[0], io.BytesIO)

    def test_pdf_scanned_and_empty_pdf_offer_ocr_guidance(self):
        for parser in (fake_pdf(None, ""), fake_pdf()):
            with patch("src.documents.importlib.import_module", return_value=parser):
                with self.assertRaisesRegex(InputError, "OCR"):
                    extract_text(b"%PDF-1.4 fake", "resume.pdf")

    def test_pdf_page_and_text_limits(self):
        for parser, message in ((fake_pdf(*(["Python"] * 31)), "30-page"), (fake_pdf("x" * (MAX_TEXT_CHARS + 1)), "50,000")):
            with patch("src.documents.importlib.import_module", return_value=parser):
                with self.assertRaisesRegex(InputError, message):
                    extract_text(b"%PDF-1.4 fake", "resume.pdf")

    def test_pdf_password_and_corruption_errors_are_actionable(self):
        for error, message in ((ValueError("encrypted password"), "password-protected"), (ValueError("xref"), "damaged")):
            parser = types.SimpleNamespace(open=MagicMock(side_effect=error))
            with patch("src.documents.importlib.import_module", return_value=parser):
                with self.assertRaisesRegex(InputError, message):
                    extract_text(b"%PDF-1.4 fake", "resume.pdf")

    def test_pdf_dependency_import_is_lazy_and_failure_is_clear(self):
        with patch("src.documents.importlib.import_module", side_effect=ImportError):
            self.assertEqual(extract_text(b"Python", "resume.txt"), "Python")
            self.assertIn("PyTorch", extract_text(make_docx(), "resume.docx"))
            with self.assertRaisesRegex(InputError, "PDF support is unavailable"):
                extract_text(b"%PDF-1.4 fake", "resume.pdf")

    def test_false_pdf_extension_rejected_before_parser(self):
        with patch("src.documents.importlib.import_module") as imported:
            with self.assertRaisesRegex(InputError, "not a valid PDF"):
                extract_text(b"not a PDF", "resume.pdf")
            imported.assert_not_called()


if __name__ == "__main__":
    unittest.main()
