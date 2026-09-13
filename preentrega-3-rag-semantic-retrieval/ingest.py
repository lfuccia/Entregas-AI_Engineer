"""
ingest.py
=========

Módulo de Ingesta (Setup) del sistema de recuperación semántica (RAG).

Lee los documentos de `./data` (.txt / .md), los fragmenta con
`RecursiveCharacterTextSplitter` (chunking basado en TOKENS, no en
caracteres: mínimo 500 tokens por fragmento con 50 de overlap) y los
persiste en una colección local de ChromaDB (`./vectorstore`).

Punto crítico: el modelo de embeddings usado acá para INDEXAR tiene que ser
exactamente el mismo que se use después para CONSULTAR (ver `embeddings.py`).
Si no coinciden, la distancia vectorial no tiene ningún sentido y la
recuperación devuelve resultados esencialmente al azar.

Uso:
    python ingest.py            # indexa (solo si el vectorstore no existe)
    python ingest.py --force    # fuerza re-indexado desde cero
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from embeddings import get_embeddings

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("rag_ingest")

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "./vectorstore"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "documentos_tecnicos")

# --- Chunking: mínimo 500 tokens con 50 de overlap ---
# Se mide la longitud en TOKENS (con el tokenizer de tiktoken) y no en
# caracteres, para respetar el tamaño pedido de forma literal e
# independiente del proveedor de embeddings/LLM que se use después.
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "500"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))


def _cargar_documentos(data_dir: Path) -> list[Document]:
    """Lee todos los .txt/.md de `data_dir` como Document de LangChain,
    guardando el nombre de archivo en `metadata["source"]`."""
    if not data_dir.exists():
        raise FileNotFoundError(
            f"No existe la carpeta de datos '{data_dir}'. Creala y poné "
            f"ahí tus archivos .txt/.md."
        )

    rutas = sorted(
        [p for p in data_dir.iterdir() if p.suffix.lower() in {".txt", ".md"}]
    )
    if not rutas:
        raise FileNotFoundError(
            f"No se encontraron archivos .txt/.md dentro de '{data_dir}'."
        )

    documentos: list[Document] = []
    for ruta in rutas:
        texto = ruta.read_text(encoding="utf-8")
        documentos.append(Document(page_content=texto, metadata={"source": ruta.name}))
        logger.info("Cargado '%s' (%d caracteres).", ruta.name, len(texto))

    return documentos


def _fragmentar(documentos: list[Document]) -> list[Document]:
    """Aplica RecursiveCharacterTextSplitter midiendo la longitud en tokens
    (tiktoken), y numera cada fragmento dentro de su documento de origen
    (metadata['chunk_index']) para poder referenciarlo luego como
    'archivo.md#fragmento_N'."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        separators=["\n## ", "\n\n", "\n", ". ", " ", ""],
    )

    fragmentos: list[Document] = []
    for documento in documentos:
        trozos = splitter.split_documents([documento])
        for indice, trozo in enumerate(trozos):
            trozo.metadata["chunk_index"] = indice
            trozo.metadata["fuente_id"] = f"{trozo.metadata['source']}#fragmento_{indice}"
            fragmentos.append(trozo)
        logger.info(
            "  '%s' -> %d fragmentos (%d tokens c/u aprox., overlap %d).",
            documento.metadata["source"],
            len(trozos),
            CHUNK_SIZE_TOKENS,
            CHUNK_OVERLAP_TOKENS,
        )

    return fragmentos


def _vectorstore_ya_poblado(persist_dir: Path, collection_name: str) -> bool:
    """Chequea si ya existe una colección de Chroma persistida con al menos
    un documento adentro, para NO volver a indexar (evita gastar tiempo y,
    si el proveedor de embeddings es de pago, costo) en cada corrida."""
    if not persist_dir.exists():
        return False

    try:
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=get_embeddings(),
            persist_directory=str(persist_dir),
        )
        cantidad = vectorstore._collection.count()  # noqa: SLF001
        logger.info(
            "Vectorstore existente en '%s': %d fragmentos ya indexados.",
            persist_dir,
            cantidad,
        )
        return cantidad > 0
    except Exception:  # noqa: BLE001
        # Carpeta corrupta/incompleta: mejor re-indexar que fallar.
        logger.warning(
            "No se pudo leer el vectorstore existente en '%s'; se "
            "re-indexará desde cero.",
            persist_dir,
        )
        return False


def ejecutar_ingesta(force: bool = False) -> Chroma:
    """
    Punto de entrada del módulo de ingesta.

    Si el vectorstore ya existe y tiene documentos, NO vuelve a indexar
    (salvo `force=True`). Devuelve el `Chroma` vectorstore listo para
    consultar.
    """
    if not force and _vectorstore_ya_poblado(PERSIST_DIR, COLLECTION_NAME):
        logger.info("Ingesta omitida: el vectorstore ya está poblado.")
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=str(PERSIST_DIR),
        )

    logger.info("Iniciando ingesta desde '%s'...", DATA_DIR)
    documentos = _cargar_documentos(DATA_DIR)
    fragmentos = _fragmentar(documentos)
    logger.info("Total de fragmentos a indexar: %d", len(fragmentos))

    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=fragmentos,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIR),
    )
    logger.info(
        "Ingesta completa: %d fragmentos persistidos en '%s' (colección "
        "'%s').",
        len(fragmentos),
        PERSIST_DIR,
        COLLECTION_NAME,
    )
    return vectorstore


if __name__ == "__main__":
    parser_cli = argparse.ArgumentParser(description="Ingesta de documentos para el RAG.")
    parser_cli.add_argument(
        "--force",
        action="store_true",
        help="Fuerza el re-indexado completo aunque ya exista un vectorstore.",
    )
    args = parser_cli.parse_args()
    ejecutar_ingesta(force=args.force)
