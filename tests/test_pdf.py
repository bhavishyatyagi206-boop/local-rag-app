"""
Tests for PDF ingestion via PyMuPDF:
- Multi-page extraction with accurate page numbers
- Citation verification carrying correct page number
- Password-protected PDFs raising IngestError
- PDFs with no text layer raising IngestError
- Corrupt PDFs raising IngestError
"""

import pytest
import fitz  # PyMuPDF
from ragcore import extract, IngestError, KnowledgeBase


def test_pdf_multipage_extraction_and_citation(tmp_path):
    pdf_path = tmp_path / "solar_system.pdf"
    doc = fitz.open()

    # Page 1
    p1 = doc.new_page()
    p1.insert_text((50, 50), "The Sun is the star at the center of the Solar System. It contains 99.8% of the system's mass.")

    # Page 2
    p2 = doc.new_page()
    p2.insert_text((50, 50), "Jupiter is the largest planet, with an equatorial diameter of roughly 142,984 kilometers.")

    # Page 3
    p3 = doc.new_page()
    p3.insert_text((50, 50), "Neptune is the eighth planet and the densest giant planet in the Solar System.")

    doc.save(str(pdf_path))
    doc.close()

    # 1. Extraction verification
    pages = extract(pdf_path)
    assert len(pages) == 3
    assert pages[0][0] == 1
    assert "99.8%" in pages[0][1]
    assert pages[1][0] == 2
    assert "142,984 kilometers" in pages[1][1]
    assert pages[2][0] == 3
    assert "Neptune" in pages[2][1]

    # 2. End-to-end KnowledgeBase citation verification
    kb = KnowledgeBase(name="Astronomy KB")
    kb.add_document("doc_solar", "solar_system.pdf", pages)

    res = kb.ask("What is the equatorial diameter of Jupiter?")
    assert res["status"] == "answered"
    assert "142,984 kilometers" in res["answer"]
    assert len(res["citations"]) >= 1

    top_citation = res["citations"][0]
    assert top_citation["doc_name"] == "solar_system.pdf"
    assert top_citation["page_number"] == 2, f"Expected page 2, got page {top_citation['page_number']}"
    assert "142,984 kilometers" in top_citation["excerpt"]


def test_pdf_password_protected(tmp_path):
    pdf_path = tmp_path / "protected.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Classified confidential intelligence briefing.")
    
    # Save with user password
    perm = fitz.PDF_PERM_ACCESSIBILITY
    doc.save(
        str(pdf_path),
        encryption=fitz.PDF_ENCRYPT_AES_256,
        user_pw="securepass123",
        owner_pw="masterpass123",
        permissions=perm,
    )
    doc.close()

    with pytest.raises(IngestError) as exc_info:
        extract(pdf_path)
    assert "password" in str(exc_info.value).lower()


def test_pdf_no_text_layer(tmp_path):
    pdf_path = tmp_path / "blank_scanned.pdf"
    doc = fitz.open()
    # Create blank pages without text
    doc.new_page()
    doc.new_page()
    doc.save(str(pdf_path))
    doc.close()

    with pytest.raises(IngestError) as exc_info:
        extract(pdf_path)
    assert "no text layer" in str(exc_info.value).lower()


def test_pdf_corrupt(tmp_path):
    pdf_path = tmp_path / "corrupt.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\ncorrupted header and broken xref table bytes \x00\xff")

    with pytest.raises(IngestError) as exc_info:
        extract(pdf_path)
    assert "corrupt" in str(exc_info.value).lower() or "unable" in str(exc_info.value).lower()
