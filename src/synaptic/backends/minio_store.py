"""MinIO blob storage backend — large content offloading."""

from __future__ import annotations

import io
import logging

try:
    from miniopy_async import Minio
except ImportError as e:
    msg = "Install synaptic-memory[minio] for MinIO backend: pip install synaptic-memory[minio]"
    raise ImportError(msg) from e

logger = logging.getLogger(__name__)


class MinIOBackend:
    """Blob storage backend for large content (PDF, images, code, etc.).

    This is NOT a StorageBackend — it's a helper used by CompositeBackend
    to offload large Node.content to object storage.
    """

    __slots__ = ("_access_key", "_bucket", "_client", "_endpoint", "_secret_key", "_secure")

    def __init__(
        self,
        endpoint: str = "localhost:9000",
        *,
        bucket: str = "synaptic",
        access_key: str = "minio",
        secret_key: str = "minio123",  # noqa: S107
        secure: bool = False,
    ) -> None:
        self._endpoint = endpoint
        self._bucket = bucket
        self._access_key = access_key
        self._secret_key = secret_key
        self._secure = secure
        self._client: Minio | None = None

    async def connect(self) -> None:
        self._client = Minio(
            self._endpoint,
            access_key=self._access_key,
            secret_key=self._secret_key,
            secure=self._secure,
        )
        # Create bucket if it doesn't exist
        if not await self._client.bucket_exists(self._bucket):
            await self._client.make_bucket(self._bucket)
            logger.info("Created MinIO bucket '%s'", self._bucket)
        else:
            logger.info("MinIO bucket '%s' already exists", self._bucket)

    async def close(self) -> None:
        self._client = None

    def _get_client(self) -> Minio:
        if self._client is None:
            msg = "Not connected. Call connect() first."
            raise RuntimeError(msg)
        return self._client

    async def upload(
        self,
        node_id: str,
        content: str | bytes,
        content_type: str = "text/plain",
    ) -> str:
        """Upload content to MinIO. Returns the object path."""
        client = self._get_client()
        if isinstance(content, str):
            data = content.encode("utf-8")
        else:
            data = content
        await client.put_object(
            self._bucket,
            node_id,
            io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )
        return f"{self._bucket}/{node_id}"

    async def download(self, node_id: str) -> bytes:
        """Download content from MinIO."""
        client = self._get_client()
        response = await client.get_object(self._bucket, node_id)
        try:
            return await response.read()
        finally:
            response.close()
            await response.release()

    async def delete(self, node_id: str) -> None:
        """Delete an object from MinIO."""
        client = self._get_client()
        await client.remove_object(self._bucket, node_id)

    async def exists(self, node_id: str) -> bool:
        """Check if an object exists."""
        client = self._get_client()
        try:
            await client.stat_object(self._bucket, node_id)
            return True
        except Exception:
            return False
