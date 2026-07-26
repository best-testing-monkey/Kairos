import threading
import time
from pathlib import Path

import pytest

from kairos import ops


class TestGpuLock:
    def test_lock_is_released_on_exit(self, tmp_path: Path) -> None:
        lock_path = tmp_path / "test.lock"
        with ops.GpuLock(lock_path=lock_path, timeout=1):
            assert lock_path.exists()
        # Released cleanly; file may remain, but no exception.

    def test_second_lock_waits(self, tmp_path: Path) -> None:
        lock_path = tmp_path / "test.lock"
        acquired_second = threading.Event()

        def hold() -> None:
            with ops.GpuLock(lock_path=lock_path, timeout=1):
                time.sleep(0.2)

        def try_acquire() -> None:
            try:
                with ops.GpuLock(lock_path=lock_path, timeout=1):
                    acquired_second.set()
            except ops.OpsError:
                pass

        t1 = threading.Thread(target=hold)
        t2 = threading.Thread(target=try_acquire)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert acquired_second.is_set()

    def test_timeout_raises(self, tmp_path: Path) -> None:
        lock_path = tmp_path / "test.lock"
        acquired = threading.Event()
        release = threading.Event()

        def holder() -> None:
            with ops.GpuLock(lock_path=lock_path, timeout=1):
                acquired.set()
                release.wait(timeout=5)

        t = threading.Thread(target=holder)
        t.start()
        acquired.wait(timeout=5)
        try:
            with pytest.raises(ops.OpsError):
                with ops.GpuLock(lock_path=lock_path, timeout=1):
                    pass
        finally:
            release.set()
            t.join(timeout=5)


class TestSendTelegram:
    def test_missing_credentials_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        with pytest.raises(ops.OpsError):
            ops.send_telegram("hello")

    def test_api_call_posts_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")

        captured: dict = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self) -> bytes:
                return b'{"ok": true}'

        def fake_urlopen(request, **kwargs):  # noqa: ANN001, ANN202
            captured["request"] = request
            return FakeResponse()

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        ops.send_telegram("hello")

        assert captured["request"].full_url == "https://api.telegram.org/bottoken/sendMessage"
        body = captured["request"].data.decode("utf-8")
        assert '"chat_id": "123"' in body
        assert '"text": "hello"' in body


class TestRequireGpu:
    def test_healthy_gpu_returns_immediately(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ops, "gpu_healthy", lambda: True)
        monkeypatch.setattr(ops, "recover_gpu", lambda **kwargs: True)  # should not be called
        ops.require_gpu()  # no exception

    def test_unhealthy_with_recovery_disabled_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ops, "gpu_healthy", lambda: False)
        with pytest.raises(ops.OpsError):
            ops.require_gpu(allow_recover=False)

    def test_unhealthy_with_recovery_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ops, "gpu_healthy", lambda: False)
        monkeypatch.setattr(ops, "recover_gpu", lambda **kwargs: True)
        ops.require_gpu()
