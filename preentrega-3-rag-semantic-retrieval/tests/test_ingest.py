"""
tests/test_ingest.py

Tests unitarios offline (sin descargar ningún modelo de embeddings, sin
LLM) para la lógica de carga y chunking de `ingest.py`. Usa una carpeta
temporal con documentos de prueba en vez de `./data`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ingest import _cargar_documentos, _fragmentar


@pytest.fixture
def carpeta_datos_temporal(tmp_path: Path) -> Path:
    carpeta = tmp_path / "data"
    carpeta.mkdir()

    # Documento largo (varios párrafos) para forzar más de un fragmento.
    contenido_largo = "\n\n".join(
        f"Párrafo número {i}: " + ("contenido de ejemplo sobre arquitectura de software. " * 40)
        for i in range(15)
    )
    (carpeta / "doc_largo.md").write_text(contenido_largo, encoding="utf-8")

    # Documento corto: debería quedar en un solo fragmento.
    (carpeta / "doc_corto.txt").write_text(
        "Este es un documento corto de prueba.", encoding="utf-8"
    )

    # Archivo que NO debe cargarse (extensión no soportada).
    (carpeta / "notas.pdf").write_text("contenido binario simulado", encoding="utf-8")

    return carpeta


def test_cargar_documentos_solo_lee_txt_y_md(carpeta_datos_temporal: Path):
    documentos = _cargar_documentos(carpeta_datos_temporal)
    nombres = {doc.metadata["source"] for doc in documentos}

    assert nombres == {"doc_largo.md", "doc_corto.txt"}
    assert "notas.pdf" not in nombres


def test_cargar_documentos_carpeta_inexistente_lanza_excepcion(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        _cargar_documentos(tmp_path / "no_existe")


def test_cargar_documentos_carpeta_vacia_lanza_excepcion(tmp_path: Path):
    carpeta_vacia = tmp_path / "vacia"
    carpeta_vacia.mkdir()
    with pytest.raises(FileNotFoundError):
        _cargar_documentos(carpeta_vacia)


def test_fragmentar_genera_multiples_fragmentos_para_documento_largo(
    carpeta_datos_temporal: Path,
):
    documentos = _cargar_documentos(carpeta_datos_temporal)
    fragmentos = _fragmentar(documentos)

    fragmentos_doc_largo = [
        f for f in fragmentos if f.metadata["source"] == "doc_largo.md"
    ]
    fragmentos_doc_corto = [
        f for f in fragmentos if f.metadata["source"] == "doc_corto.txt"
    ]

    # El documento largo (muchos párrafos repetidos) debe partirse en más
    # de un fragmento con chunk_size=500 tokens.
    assert len(fragmentos_doc_largo) > 1
    # El documento corto entra entero en un solo fragmento.
    assert len(fragmentos_doc_corto) == 1


def test_fragmentar_asigna_metadata_de_trazabilidad(carpeta_datos_temporal: Path):
    documentos = _cargar_documentos(carpeta_datos_temporal)
    fragmentos = _fragmentar(documentos)

    for indice, fragmento in enumerate(
        f for f in fragmentos if f.metadata["source"] == "doc_corto.txt"
    ):
        assert fragmento.metadata["chunk_index"] == indice
        assert fragmento.metadata["fuente_id"] == f"doc_corto.txt#fragmento_{indice}"
