# -*- coding: utf-8 -*-
"""
Smoke tests for the network-free pure functions in uploader.py/app.py:
sanitize_filename, parse_xml_to_markdown, chunk_document, group_results.

Run with: pytest tests/
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document

from uploader import sanitize_filename, parse_xml_to_markdown, chunk_document
from app import group_results


# -- sanitize_filename --------------------------------------------------

def test_sanitize_filename_strips_spaces_and_accents():
    assert sanitize_filename("My Paper (2020) - cafe.pdf") == "My-Paper-2020---cafe.pdf"


def test_sanitize_filename_falls_back_when_nothing_survives():
    assert sanitize_filename("!!!!!") == "document.pdf"


def test_sanitize_filename_passes_through_safe_name():
    assert sanitize_filename("simple_name.pdf") == "simple_name.pdf"


# -- parse_xml_to_markdown -----------------------------------------------

SAMPLE_TEI = """<?xml version="1.0"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt><title>A Test Paper</title></titleStmt>
      <sourceDesc>
        <biblStruct>
          <analytic>
            <author><persName><forename>Jane</forename><surname>Doe</surname></persName></author>
          </analytic>
          <idno type="DOI">10.1234/test</idno>
          <monogr><imprint><date when="2021-05-01">May 2021</date></imprint></monogr>
        </biblStruct>
      </sourceDesc>
    </fileDesc>
    <profileDesc>
      <abstract><p>This is the abstract.</p></abstract>
    </profileDesc>
  </teiHeader>
  <text>
    <body>
      <div><head n="1">Introduction</head><p>Intro text.</p></div>
      <div><head>References</head><p>Should be dropped.</p></div>
    </body>
  </text>
</TEI>"""


def test_parse_xml_to_markdown_extracts_metadata():
    title, authors, doi, year, body, download_filename = parse_xml_to_markdown(SAMPLE_TEI, "paper.pdf")
    assert title == "A Test Paper"
    assert authors == ["Jane Doe"]
    assert doi == "10.1234/test"
    assert year == "2021"
    assert download_filename == "paper.md"


def test_parse_xml_to_markdown_drops_references_section():
    _, _, _, _, body, _ = parse_xml_to_markdown(SAMPLE_TEI, "paper.pdf")
    assert "Introduction" in body
    assert "Intro text." in body
    assert "References" not in body
    assert "Should be dropped." not in body


def test_parse_xml_to_markdown_includes_abstract():
    _, _, _, _, body, _ = parse_xml_to_markdown(SAMPLE_TEI, "paper.pdf")
    assert "## Abstract" in body
    assert "This is the abstract." in body


# -- chunk_document --------------------------------------------------------

def test_chunk_document_splits_on_headers_and_applies_metadata():
    markdown = "## Section One\n\n" + ("word " * 10) + "\n\n## Section Two\n\n" + ("word " * 10)
    chunks = chunk_document(markdown, {"title": "Doc Title", "doc_id": "abc123"})
    assert len(chunks) >= 2
    for c in chunks:
        assert c.metadata["title"] == "Doc Title"
        assert c.metadata["doc_id"] == "abc123"


def test_chunk_document_respects_token_limit():
    from common import EMBEDDING_MAX_TOKENS, load_embedding_tokenizer

    long_section = "## Long Section\n\n" + ("word " * 2000)
    chunks = chunk_document(long_section, {"title": "Long Doc"})
    tokenizer = load_embedding_tokenizer()
    for c in chunks:
        token_len = len(tokenizer.encode(c.page_content))
        assert token_len <= EMBEDDING_MAX_TOKENS


# -- group_results --------------------------------------------------------

def test_group_results_groups_by_source_then_title():
    d1 = Document(page_content="a", metadata={"source": "x.pdf", "title": "X"})
    d2 = Document(page_content="b", metadata={"source": "x.pdf", "title": "X"})
    d3 = Document(page_content="c", metadata={"title": "Y only"})
    groups = group_results([d1, d2, d3])
    assert len(groups) == 2
    assert len(groups[0]["chunks"]) == 2
    assert len(groups[1]["chunks"]) == 1
    assert groups[1]["metadata"]["title"] == "Y only"


def test_group_results_empty_list():
    assert group_results([]) == []
