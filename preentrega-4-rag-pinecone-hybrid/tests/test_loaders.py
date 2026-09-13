"""
tests/test_loaders.py

Tests unitarios offline (sin red, sin API keys) para la carga
multi-formato de `loaders.py`, usando el dataset real de `./data`.
"""

from __future__ import annotations

from pathlib import Path

from loaders import cargar_documentos

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_carga_los_cinco_archivos_del_dataset_de_ejemplo():
    documentos = cargar_documentos(DATA_DIR)
    fuentes = {doc.metadata["source"] for doc in documentos}

    assert "01_fastapi_routing.md" in fuentes
    assert "02_fastapi_dependencias.md" in fuentes
    assert "03_fastapi_validacion_pydantic.md" in fuentes
    assert "04_faq_fastapi.json" in fuentes
    assert "05_notas_migracion_pydantic_v2.pdf" in fuentes


def test_markdown_tiene_categoria_y_page_none():
    documentos = cargar_documentos(DATA_DIR)
    doc_md = next(d for d in documentos if d.metadata["source"] == "01_fastapi_routing.md")

    assert doc_md.metadata["categoria"] == "routing"
    assert doc_md.metadata["page"] is None
    assert "Path parameters" in doc_md.page_content


def test_json_genera_un_documento_por_entrada_de_faq():
    documentos = cargar_documentos(DATA_DIR)
    docs_faq = [d for d in documentos if d.metadata["source"] == "04_faq_fastapi.json"]

    assert len(docs_faq) == 4  # 4 entradas en el JSON de ejemplo
    assert all(d.metadata["categoria"] for d in docs_faq)
    assert "Pregunta:" in docs_faq[0].page_content
    assert "Respuesta:" in docs_faq[0].page_content


def test_pdf_genera_un_documento_por_pagina_con_numero_de_pagina():
    documentos = cargar_documentos(DATA_DIR)
    docs_pdf = [
        d for d in documentos if d.metadata["source"] == "05_notas_migracion_pydantic_v2.pdf"
    ]

    assert len(docs_pdf) >= 1
    numeros_de_pagina = [d.metadata["page"] for d in docs_pdf]
    assert numeros_de_pagina == sorted(numeros_de_pagina)
    assert all(isinstance(p, int) and p >= 1 for p in numeros_de_pagina)
    assert docs_pdf[0].metadata["categoria"] == "migracion"


def test_carpeta_inexistente_lanza_excepcion(tmp_path: Path):
    import pytest

    with pytest.raises(FileNotFoundError):
        cargar_documentos(tmp_path / "no_existe")
