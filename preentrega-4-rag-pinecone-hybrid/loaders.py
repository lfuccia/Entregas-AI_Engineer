"""
loaders.py
==========

Carga de documentos multi-formato para el pipeline de ingesta: Markdown,
JSON (estructurado, tipo FAQ) y PDF. Cada `Document` resultante lleva
metadatos avanzados y consistentes entre formatos:

    - "source":   nombre del archivo de origen.
    - "page":     número de página (1-indexado) para PDFs; `None` para
                  Markdown/JSON (documentos de una sola "página" lógica).
    - "categoria": etiqueta temática, usada para filtrar/organizar en
                  Pinecone y para inspeccionar resultados de evaluación.

Estos son justamente los "metadatos avanzados (fuente, página, etiquetas
de categoría)" que pide el enunciado, y viajan intactos a través del
chunking hasta terminar guardados en Pinecone.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger("rag_loaders")

# Categoría por archivo. En un proyecto real esto podría venir de una
# convención de carpetas (data/<categoria>/archivo.md) en vez de un mapa
# hardcodeado; acá se mantiene explícito y simple para el dataset de
# ejemplo.
CATEGORIA_POR_ARCHIVO: dict[str, str] = {
    "01_fastapi_routing.md": "routing",
    "02_fastapi_dependencias.md": "inyeccion_de_dependencias",
    "03_fastapi_validacion_pydantic.md": "validacion",
    "04_faq_fastapi.json": "faq",
    "05_notas_migracion_pydantic_v2.pdf": "migracion",
}


def _categoria_de(nombre_archivo: str) -> str:
    return CATEGORIA_POR_ARCHIVO.get(nombre_archivo, "general")


def _cargar_markdown(ruta: Path) -> list[Document]:
    texto = ruta.read_text(encoding="utf-8")
    return [
        Document(
            page_content=texto,
            metadata={
                "source": ruta.name,
                "page": None,
                "categoria": _categoria_de(ruta.name),
            },
        )
    ]


def _cargar_json_faq(ruta: Path) -> list[Document]:
    """Espera una lista de objetos {"pregunta", "respuesta", "categoria"?}.
    Cada entrada se convierte en un Document independiente (no se fragmenta
    después: una entrada de FAQ ya es una unidad semántica autocontenida)."""
    entradas = json.loads(ruta.read_text(encoding="utf-8"))
    documentos = []
    for indice, entrada in enumerate(entradas):
        pregunta = entrada.get("pregunta", "").strip()
        respuesta = entrada.get("respuesta", "").strip()
        texto = f"Pregunta: {pregunta}\nRespuesta: {respuesta}"
        categoria = entrada.get("categoria") or _categoria_de(ruta.name)
        documentos.append(
            Document(
                page_content=texto,
                metadata={
                    "source": ruta.name,
                    "page": None,
                    "categoria": categoria,
                    "entrada_index": indice,
                },
            )
        )
    return documentos


def _cargar_pdf(ruta: Path) -> list[Document]:
    from pypdf import PdfReader

    lector = PdfReader(str(ruta))
    documentos = []
    for indice, pagina in enumerate(lector.pages):
        texto = pagina.extract_text() or ""
        if not texto.strip():
            continue
        documentos.append(
            Document(
                page_content=texto,
                metadata={
                    "source": ruta.name,
                    "page": indice + 1,  # 1-indexado, más natural para citar
                    "categoria": _categoria_de(ruta.name),
                },
            )
        )
    return documentos


_CARGADORES = {
    ".md": _cargar_markdown,
    ".txt": _cargar_markdown,
    ".json": _cargar_json_faq,
    ".pdf": _cargar_pdf,
}


def cargar_documentos(data_dir: Path) -> list[Document]:
    """Carga todos los .md/.txt/.json/.pdf de `data_dir` como Document de
    LangChain, con metadatos de fuente/página/categoría ya asignados."""
    if not data_dir.exists():
        raise FileNotFoundError(
            f"No existe la carpeta de datos '{data_dir}'. Creala y poné ahí "
            f"tus documentos (.md, .json, .pdf)."
        )

    rutas = sorted(
        p for p in data_dir.iterdir() if p.suffix.lower() in _CARGADORES
    )
    if not rutas:
        raise FileNotFoundError(
            f"No se encontraron archivos .md/.txt/.json/.pdf dentro de "
            f"'{data_dir}'."
        )

    documentos: list[Document] = []
    for ruta in rutas:
        cargador = _CARGADORES[ruta.suffix.lower()]
        nuevos = cargador(ruta)
        documentos.extend(nuevos)
        logger.info(
            "Cargado '%s' (%s) -> %d documento(s).",
            ruta.name,
            ruta.suffix.lower(),
            len(nuevos),
        )

    return documentos
