"""MinIO backend integration tests.

Requires running MinIO: localhost:9000 (minio/minio123)
"""

import pytest

try:
    from synaptic.backends.minio_store import MinIOBackend

    HAS_MINIO = True
except ImportError:
    HAS_MINIO = False

pytestmark = [
    pytest.mark.minio,
    pytest.mark.skipif(not HAS_MINIO, reason="miniopy-async not installed"),
]

TEST_BUCKET = "test-synaptic"


@pytest.fixture
async def backend():
    b = MinIOBackend(
        "localhost:9000",
        bucket=TEST_BUCKET,
        access_key="minio",
        secret_key="minio123",
        secure=False,
    )
    try:
        await b.connect()
    except Exception:
        pytest.skip("MinIO server not available")
    yield b
    # Cleanup: remove all objects in test bucket
    client = b._get_client()
    try:
        objects = await client.list_objects(TEST_BUCKET)
        async for obj in objects:
            await client.remove_object(TEST_BUCKET, obj.object_name)
        await client.remove_bucket(TEST_BUCKET)
    except Exception:
        pass
    await b.close()


class TestMinIOLifecycle:
    @pytest.mark.asyncio
    async def test_connect_creates_bucket(self, backend: MinIOBackend) -> None:
        client = backend._get_client()
        assert await client.bucket_exists(TEST_BUCKET)


class TestMinIOUploadDownload:
    @pytest.mark.asyncio
    async def test_upload_and_download_str(self, backend: MinIOBackend) -> None:
        content = "Hello, this is a test document with Korean: 안녕하세요"
        path = await backend.upload("doc1", content)
        assert TEST_BUCKET in path

        data = await backend.download("doc1")
        assert data.decode("utf-8") == content

    @pytest.mark.asyncio
    async def test_upload_and_download_bytes(self, backend: MinIOBackend) -> None:
        content = b"\x00\x01\x02\x03 binary data"
        await backend.upload("bin1", content, content_type="application/octet-stream")

        data = await backend.download("bin1")
        assert data == content

    @pytest.mark.asyncio
    async def test_upload_overwrites(self, backend: MinIOBackend) -> None:
        await backend.upload("doc1", "version 1")
        await backend.upload("doc1", "version 2")

        data = await backend.download("doc1")
        assert data.decode("utf-8") == "version 2"


class TestMinIODelete:
    @pytest.mark.asyncio
    async def test_delete(self, backend: MinIOBackend) -> None:
        await backend.upload("doc1", "content")
        await backend.delete("doc1")
        assert await backend.exists("doc1") is False

    @pytest.mark.asyncio
    async def test_exists(self, backend: MinIOBackend) -> None:
        assert await backend.exists("nonexistent") is False
        await backend.upload("doc1", "content")
        assert await backend.exists("doc1") is True
