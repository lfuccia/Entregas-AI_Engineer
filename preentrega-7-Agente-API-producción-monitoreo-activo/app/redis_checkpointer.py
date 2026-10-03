"""
redis_checkpointer.py
=====================

`RedisSaver`: un `BaseCheckpointSaver` de LangGraph que persiste los
checkpoints del grafo en Redis en vez de en memoria o en SQLite (como en
la pre-entrega 5). Esto es lo que permite:

1. Retomar una tarea desde donde quedó (incluyendo el "pause" de
   Human-in-the-loop de `hitl.py`) aunque el proceso de la API se
   reinicie entremedio — el estado no vive en RAM, vive en Redis.
2. Que `POST /tasks` dispare la ejecución en un worker en background y
   `POST /tasks/{id}/approve`, en una petición HTTP COMPLETAMENTE
   distinta (y potencialmente minutos u horas después), pueda retomar
   exactamente el mismo grafo en el mismo punto.

Implementación
---------------
En vez de envolver el paquete oficial `langgraph-checkpoint-redis` (que
requiere Redis Stack con los módulos RedisJSON/RediSearch, no
disponibles en una imagen `redis:7-alpine` común), esta clase es un
puerto directo del algoritmo de `InMemorySaver` (la referencia oficial
de LangGraph, en `langgraph.checkpoint.memory`) que reemplaza los
`dict` en memoria por estructuras de datos equivalentes en Redis
(hashes), serializando cada valor con el mismo `self.serde`
(`JsonPlusSerializer`) que usa el resto de LangGraph. Así funciona con
CUALQUIER Redis estándar (el `redis:7-alpine` de `docker-compose.yml`
alcanza y sobra).

Solo se implementan los métodos ASYNC (`aget_tuple`, `alist`, `aput`,
`aput_writes`, `adelete_thread`) porque toda la app usa el grafo en modo
async (`ainvoke`/`astream`) — igual que se hizo con `AsyncSqliteSaver`
en la pre-entrega 5.
"""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)

try:  # pragma: no cover - solo para type hints, no es obligatorio en runtime
    from redis.asyncio import Redis
except ImportError:  # pragma: no cover
    Redis = Any  # type: ignore[assignment,misc]


def _enc(tipado: tuple[str, bytes]) -> str:
    """(tipo, bytes) -> string JSON-safe, para guardar como valor de un
    campo de un hash de Redis."""
    tipo, data = tipado
    return json.dumps({"t": tipo, "d": base64.b64encode(data).decode("ascii")})


def _dec(valor: str) -> tuple[str, bytes]:
    obj = json.loads(valor)
    return obj["t"], base64.b64decode(obj["d"])


class RedisSaver(BaseCheckpointSaver[str]):
    """Checkpointer de LangGraph respaldado por Redis.

    Parameters
    ----------
    redis_client:
        Un cliente `redis.asyncio.Redis` ya conectado (se reutiliza la
        misma conexión/pool que usa el resto de la app para el estado de
        los jobs, ver `app/redis_state.py`).
    key_prefix:
        Prefijo de namespacing para no pisar otras claves de Redis que
        use la app en la misma instancia (por default `"lg"`, de
        "LangGraph").
    """

    def __init__(self, redis_client: "Redis", *, key_prefix: str = "lg") -> None:
        super().__init__()
        self._redis = redis_client
        self._prefix = key_prefix

    # -- claves ------------------------------------------------------

    def _k_checkpoints(self, thread_id: str, ns: str) -> str:
        return f"{self._prefix}:ckpt:{thread_id}:{ns}"

    def _k_blobs(self, thread_id: str, ns: str) -> str:
        return f"{self._prefix}:blob:{thread_id}:{ns}"

    def _k_writes(self, thread_id: str, ns: str, checkpoint_id: str) -> str:
        return f"{self._prefix}:writes:{thread_id}:{ns}:{checkpoint_id}"

    def _k_namespaces(self, thread_id: str) -> str:
        return f"{self._prefix}:ns:{thread_id}"

    # -- helpers internos ---------------------------------------------

    async def _load_blobs(self, thread_id: str, ns: str, versions: dict[str, Any]) -> dict[str, Any]:
        if not versions:
            return {}
        hkey = self._k_blobs(thread_id, ns)
        campos = [f"{canal}::{version}" for canal, version in versions.items()]
        valores = await self._redis.hmget(hkey, campos)
        resultado: dict[str, Any] = {}
        for (canal, _version), crudo in zip(versions.items(), valores):
            if crudo is None:
                continue
            tipo, data = _dec(crudo.decode() if isinstance(crudo, bytes) else crudo)
            if tipo == "empty":
                continue
            resultado[canal] = self.serde.loads_typed((tipo, data))
        return resultado

    async def _cargar_writes(self, thread_id: str, ns: str, checkpoint_id: str) -> list[tuple[str, str, Any]]:
        hkey = self._k_writes(thread_id, ns, checkpoint_id)
        crudos = await self._redis.hgetall(hkey)
        items = []
        for campo, valor in crudos.items():
            campo_s = campo.decode() if isinstance(campo, bytes) else campo
            valor_s = valor.decode() if isinstance(valor, bytes) else valor
            envelope = json.loads(valor_s)
            task_id = envelope["task_id"]
            idx = envelope["idx"]
            canal = envelope["channel"]
            valor_deserializado = self.serde.loads_typed((envelope["t"], base64.b64decode(envelope["d"])))
            items.append((idx, task_id, canal, valor_deserializado))
        items.sort(key=lambda x: x[0])
        return [(task_id, canal, valor) for _idx, task_id, canal, valor in items]

    # -- API async ------------------------------------------------------

    async def aget_tuple(self, config: dict) -> CheckpointTuple | None:
        thread_id: str = config["configurable"]["thread_id"]
        ns: str = config["configurable"].get("checkpoint_ns", "")
        hkey = self._k_checkpoints(thread_id, ns)

        checkpoint_id = get_checkpoint_id(config)
        if checkpoint_id:
            crudo = await self._redis.hget(hkey, checkpoint_id)
            if crudo is None:
                return None
        else:
            campos = await self._redis.hkeys(hkey)
            if not campos:
                return None
            checkpoint_id = max(c.decode() if isinstance(c, bytes) else c for c in campos)
            crudo = await self._redis.hget(hkey, checkpoint_id)
            if crudo is None:
                return None

        crudo_s = crudo.decode() if isinstance(crudo, bytes) else crudo
        entrada = json.loads(crudo_s)
        checkpoint_tipo, checkpoint_bytes = entrada["checkpoint"]["t"], base64.b64decode(entrada["checkpoint"]["d"])
        metadata_tipo, metadata_bytes = entrada["metadata"]["t"], base64.b64decode(entrada["metadata"]["d"])
        parent_checkpoint_id = entrada.get("parent")

        checkpoint: Checkpoint = self.serde.loads_typed((checkpoint_tipo, checkpoint_bytes))
        metadata: CheckpointMetadata = self.serde.loads_typed((metadata_tipo, metadata_bytes))

        canales = await self._load_blobs(thread_id, ns, checkpoint["channel_versions"])
        writes = await self._cargar_writes(thread_id, ns, checkpoint_id)

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": ns,
                    "checkpoint_id": checkpoint_id,
                }
            },
            checkpoint={**checkpoint, "channel_values": canales},
            metadata=metadata,
            pending_writes=writes,
            parent_config=(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": ns,
                        "checkpoint_id": parent_checkpoint_id,
                    }
                }
                if parent_checkpoint_id
                else None
            ),
        )

    async def alist(
        self,
        config: dict | None,
        *,
        filter: dict[str, Any] | None = None,
        before: dict | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        if config is None:
            return
        thread_id = config["configurable"]["thread_id"]
        ns = config["configurable"].get("checkpoint_ns", "")
        hkey = self._k_checkpoints(thread_id, ns)
        crudos = await self._redis.hgetall(hkey)

        ids = sorted(
            (c.decode() if isinstance(c, bytes) else c for c in crudos.keys()),
            reverse=True,
        )
        before_id = get_checkpoint_id(before) if before else None
        emitidos = 0
        for checkpoint_id in ids:
            if before_id and checkpoint_id >= before_id:
                continue
            tupla = await self.aget_tuple(
                {"configurable": {"thread_id": thread_id, "checkpoint_ns": ns, "checkpoint_id": checkpoint_id}}
            )
            if tupla is None:
                continue
            if filter and not all(v == tupla.metadata.get(k) for k, v in filter.items()):
                continue
            if limit is not None and emitidos >= limit:
                break
            emitidos += 1
            yield tupla

    async def aput(self, config: dict, checkpoint: Checkpoint, metadata: CheckpointMetadata, new_versions: dict) -> dict:
        thread_id = config["configurable"]["thread_id"]
        ns = config["configurable"]["checkpoint_ns"]

        c = dict(checkpoint)
        valores: dict[str, Any] = c.pop("channel_values")  # type: ignore[assignment]

        if new_versions:
            hkey_blobs = self._k_blobs(thread_id, ns)
            mapping = {}
            for canal, version in new_versions.items():
                if canal in valores:
                    tipado = self.serde.dumps_typed(valores[canal])
                else:
                    tipado = ("empty", b"")
                mapping[f"{canal}::{version}"] = _enc(tipado)
            await self._redis.hset(hkey_blobs, mapping=mapping)

        entrada = {
            "checkpoint": {"t": None, "d": None},
            "metadata": {"t": None, "d": None},
            "parent": config["configurable"].get("checkpoint_id"),
        }
        ck_tipo, ck_bytes = self.serde.dumps_typed(c)
        md_tipo, md_bytes = self.serde.dumps_typed(get_checkpoint_metadata(config, metadata))
        entrada["checkpoint"] = {"t": ck_tipo, "d": base64.b64encode(ck_bytes).decode("ascii")}
        entrada["metadata"] = {"t": md_tipo, "d": base64.b64encode(md_bytes).decode("ascii")}

        hkey_ckpt = self._k_checkpoints(thread_id, ns)
        await self._redis.hset(hkey_ckpt, checkpoint["id"], json.dumps(entrada))
        await self._redis.sadd(self._k_namespaces(thread_id), ns)

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    async def aput_writes(self, config: dict, writes: Sequence[tuple[str, Any]], task_id: str, task_path: str = "") -> None:
        thread_id = config["configurable"]["thread_id"]
        ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]
        hkey = self._k_writes(thread_id, ns, checkpoint_id)

        existentes = await self._redis.hkeys(hkey)
        existentes_decoded = {e.decode() if isinstance(e, bytes) else e for e in existentes}

        mapping = {}
        for idx, (canal, valor) in enumerate(writes):
            idx_real = WRITES_IDX_MAP.get(canal, idx)
            campo = f"{task_id}::{idx_real}"
            if idx_real >= 0 and campo in existentes_decoded:
                continue
            tipo, data = self.serde.dumps_typed(valor)
            mapping[campo] = json.dumps(
                {
                    "task_id": task_id,
                    "idx": idx_real,
                    "channel": canal,
                    "t": tipo,
                    "d": base64.b64encode(data).decode("ascii"),
                    "task_path": task_path,
                }
            )
        if mapping:
            await self._redis.hset(hkey, mapping=mapping)

    async def adelete_thread(self, thread_id: str) -> None:
        namespaces = await self._redis.smembers(self._k_namespaces(thread_id))
        claves = [self._k_namespaces(thread_id)]
        for ns_raw in namespaces:
            ns = ns_raw.decode() if isinstance(ns_raw, bytes) else ns_raw
            hkey_ckpt = self._k_checkpoints(thread_id, ns)
            campos = await self._redis.hkeys(hkey_ckpt)
            for cid in campos:
                cid_s = cid.decode() if isinstance(cid, bytes) else cid
                claves.append(self._k_writes(thread_id, ns, cid_s))
            claves.append(hkey_ckpt)
            claves.append(self._k_blobs(thread_id, ns))
        if claves:
            await self._redis.delete(*claves)
