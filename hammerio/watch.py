"""HammerIO Watch Daemon — filesystem-based auto-compression pipeline.

Monitors ``compress/`` and ``decompress/`` folders for new files and
automatically processes them using the optimal GPU/CPU encoder.

Copyright 2026 ResilientMind AI | ResilientMindai.com | Joseph C McGinty Jr
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import os

os.environ.setdefault("EGL_LOG_LEVEL", "fatal")

import logging
import queue
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime
from pathlib import Path
from typing import Optional

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger("hammerio.watch")

# Extensions to skip — temp / partial downloads
_SKIP_EXTENSIONS = {".tmp", ".part", ".crdownload"}

# Minimum interval between disk-space checks (seconds)
_DISK_CHECK_INTERVAL = 30


def _human_size(size_bytes: int) -> str:
    """Format byte count as a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 ** 3):.2f} GB"


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class _WatchHandler(FileSystemEventHandler):
    """Watchdog handler that enqueues new files for processing."""

    def __init__(self, work_queue: queue.Queue, action: str) -> None:
        super().__init__()
        self._queue = work_queue
        self._action = action

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._queue.put((self._action, Path(event.src_path)))

    def on_moved(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._queue.put((self._action, Path(event.dest_path)))


class WatchDaemon:
    """Filesystem watcher that auto-compresses / decompresses files.

    Parameters
    ----------
    watch_root:
        Base directory containing ``compress/`` and ``decompress/`` folders.
    compress_output:
        Where compressed files are written.  Defaults to ``<watch_root>/compressed``.
    decompress_output:
        Where decompressed files are written.  Defaults to ``<watch_root>/decompressed``.
    gpu_threshold_mb:
        Files at or above this size (in MB) use GPU nvCOMP LZ4; smaller
        files use CPU zstd.
    workers:
        Thread-pool size for concurrent processing.
    stable_wait:
        Seconds to wait between file-stability size checks.
    move_to_processed:
        Whether to move source files into ``processed/`` after success.
    """

    def __init__(
        self,
        watch_root: str | Path = "./hammer-watch",
        compress_output: Optional[str | Path] = None,
        decompress_output: Optional[str | Path] = None,
        gpu_threshold_mb: int = 100,
        workers: int = 4,
        stable_wait: float = 0.5,
        move_to_processed: bool = True,
    ) -> None:
        self.watch_root = Path(watch_root).resolve()
        self.compress_dir = self.watch_root / "compress"
        self.decompress_dir = self.watch_root / "decompress"
        self.compress_output = Path(compress_output).resolve() if compress_output else self.watch_root / "compressed"
        self.decompress_output = Path(decompress_output).resolve() if decompress_output else self.watch_root / "decompressed"
        self.processed_compress = self.watch_root / "processed" / "compress"
        self.processed_decompress = self.watch_root / "processed" / "decompress"

        self.gpu_threshold_mb = gpu_threshold_mb
        self.stable_wait = stable_wait
        self.move_to_processed = move_to_processed
        self._workers = workers

        # Statistics
        self._stats_lock = threading.Lock()
        self._total_files = 0
        self._total_bytes_in = 0
        self._total_bytes_out = 0
        self._total_time_saved = 0.0  # estimated vs baseline
        self._start_time: Optional[float] = None

        # Internal machinery
        self._queue: queue.Queue = queue.Queue()
        self._observer: Optional[Observer] = None
        self._stop_event = threading.Event()
        self._log_file: Optional[Path] = None

        # Lazily initialised encoders (avoid heavy imports at class level)
        self._general_encoder = None
        self._bulk_encoder = None
        self._hw = None

    # ------------------------------------------------------------------
    # Folder setup
    # ------------------------------------------------------------------

    def _ensure_folders(self) -> None:
        """Create the full folder tree under *watch_root*."""
        for d in (
            self.compress_dir,
            self.decompress_dir,
            self.processed_compress,
            self.processed_decompress,
            self.compress_output,
            self.decompress_output,
        ):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _setup_logging(self) -> None:
        """Configure dual logging — stdout + file."""
        self._log_file = self.watch_root / "hammer-watch.log"

        watch_logger = logging.getLogger("hammerio.watch")
        watch_logger.setLevel(logging.INFO)

        # Clear existing handlers to avoid duplicates on restart
        watch_logger.handlers.clear()

        formatter = logging.Formatter("[WATCH] %(message)s")

        # Stdout handler
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        watch_logger.addHandler(sh)

        # File handler
        fh = logging.FileHandler(self._log_file, mode="a", encoding="utf-8")
        fh.setFormatter(formatter)
        watch_logger.addHandler(fh)

    def _log(
        self,
        action: str,
        filename: str,
        method: str = "",
        size: int = 0,
        duration: float = 0.0,
        throughput: float = 0.0,
    ) -> None:
        """Emit a structured log line."""
        parts = [_timestamp(), action, filename]
        if method:
            parts.append(method)
        if size:
            parts.append(_human_size(size))
        if duration > 0:
            parts.append(f"{duration:.2f}s")
        if throughput > 0:
            parts.append(f"{throughput:.1f} MB/s")
        logger.info(" ".join(parts))

    # ------------------------------------------------------------------
    # Encoder initialisation (lazy)
    # ------------------------------------------------------------------

    def _init_encoders(self) -> None:
        from hammerio.core.hardware import detect_hardware
        from hammerio.encoders.general import GeneralEncoder
        from hammerio.encoders.bulk import BulkEncoder

        self._hw = detect_hardware()
        self._general_encoder = GeneralEncoder(self._hw)
        self._bulk_encoder = BulkEncoder(self._hw)

    # ------------------------------------------------------------------
    # File-stability check
    # ------------------------------------------------------------------

    def _wait_stable(self, path: Path, retries: int = 5) -> bool:
        """Return True once the file size is stable across two checks."""
        for _ in range(retries):
            if self._stop_event.is_set():
                return False
            try:
                size1 = path.stat().st_size
            except OSError:
                return False
            time.sleep(self.stable_wait)
            try:
                size2 = path.stat().st_size
            except OSError:
                return False
            if size1 == size2 and size1 > 0:
                return True
        return False

    # ------------------------------------------------------------------
    # Skip logic
    # ------------------------------------------------------------------

    def _should_skip(self, path: Path) -> bool:
        """Return True for hidden files, temp files, or files in output dirs."""
        name = path.name

        # Hidden files
        if name.startswith("."):
            return True

        # Temp extensions
        if path.suffix.lower() in _SKIP_EXTENSIONS:
            return True

        # Avoid reprocessing files in output / processed folders
        try:
            resolved = path.resolve()
            for skip_dir in (
                self.processed_compress,
                self.processed_decompress,
                self.compress_output,
                self.decompress_output,
            ):
                try:
                    resolved.relative_to(skip_dir.resolve())
                    return True
                except ValueError:
                    pass
        except OSError:
            return True

        return False

    # ------------------------------------------------------------------
    # Disk-space guard
    # ------------------------------------------------------------------

    def _check_disk_space(self) -> None:
        """Block while the processed/ partition is > 90 % full."""
        while not self._stop_event.is_set():
            usage = shutil.disk_usage(self.watch_root)
            pct_used = usage.used / usage.total * 100
            if pct_used < 90:
                return
            logger.warning(
                "%s DISK_SPACE_WARNING partition %.1f%% full — pausing",
                _timestamp(),
                pct_used,
            )
            self._stop_event.wait(timeout=_DISK_CHECK_INTERVAL)

    # ------------------------------------------------------------------
    # Compression handler
    # ------------------------------------------------------------------

    def _handle_compress(self, path: Path) -> None:
        """Compress a file from the compress/ folder."""
        if self._should_skip(path) or not path.exists():
            return

        if not self._wait_stable(path):
            self._log("SKIP_UNSTABLE", path.name)
            return

        self._check_disk_space()

        file_size = path.stat().st_size
        threshold_bytes = self.gpu_threshold_mb * 1024 * 1024

        start = time.time()
        method = ""
        out_path_str = ""

        try:
            if file_size >= threshold_bytes and self._hw and self._hw.has_nvcomp:
                # GPU path: nvCOMP LZ4 via BulkEncoder
                method = "GPU-nvCOMP-LZ4"
                out_file = self.compress_output / (path.name + ".hammer")
                out_path_str = self._bulk_encoder.process(
                    input_path=path,
                    output_path=out_file,
                    algorithm="lz4",
                    quality="balanced",
                )
            else:
                # CPU path: zstd via GeneralEncoder
                method = "CPU-zstd"
                out_file = self.compress_output / (path.name + ".zst")
                out_path_str = self._general_encoder.process(
                    input_path=path,
                    output_path=out_file,
                    algorithm="zstd",
                    quality="balanced",
                )

            elapsed = time.time() - start
            out_size = Path(out_path_str).stat().st_size if out_path_str and Path(out_path_str).exists() else 0
            throughput = (file_size / (1024 * 1024)) / elapsed if elapsed > 0 else 0.0

            self._log("COMPRESSED", path.name, method, file_size, elapsed, throughput)

            # Move source to processed
            if self.move_to_processed:
                dest = self.processed_compress / path.name
                shutil.move(str(path), str(dest))

            with self._stats_lock:
                self._total_files += 1
                self._total_bytes_in += file_size
                self._total_bytes_out += out_size

        except Exception as exc:
            elapsed = time.time() - start
            self._log("ERROR", path.name, method or "unknown", file_size, elapsed)
            logger.error("Compression failed for %s: %s", path.name, exc)

            # Write error log next to the source
            error_log = path.parent / f".{path.name}.error.log"
            try:
                error_log.write_text(
                    f"timestamp: {_timestamp()}\n"
                    f"file: {path.name}\n"
                    f"method: {method}\n"
                    f"error: {exc}\n",
                    encoding="utf-8",
                )
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Decompression handler
    # ------------------------------------------------------------------

    def _handle_decompress(self, path: Path) -> None:
        """Decompress a file from the decompress/ folder."""
        if self._should_skip(path) or not path.exists():
            return

        ext = path.suffix.lower()
        if ext not in (".lz4", ".zst", ".gz", ".bz2", ".hammer"):
            self._log("SKIP_UNKNOWN_FORMAT", path.name)
            return

        if not self._wait_stable(path):
            self._log("SKIP_UNSTABLE", path.name)
            return

        self._check_disk_space()

        file_size = path.stat().st_size
        start = time.time()
        method = ""

        try:
            if ext == ".hammer":
                method = "BulkEncoder"
                out_path_str = self._bulk_encoder.decompress(
                    input_path=path,
                    output_path=self.decompress_output / path.stem,
                )
            else:
                method = "GeneralEncoder"
                # Determine output name by stripping the compression extension
                stem = path.name
                for comp_ext in (".zst", ".lz4", ".gz", ".bz2"):
                    if stem.endswith(comp_ext):
                        stem = stem[: -len(comp_ext)]
                        break
                out_path_str = self._general_encoder.decompress(
                    input_path=path,
                    output_path=self.decompress_output / stem,
                )

            elapsed = time.time() - start
            out_size = 0
            if out_path_str:
                op = Path(out_path_str)
                if op.exists():
                    out_size = op.stat().st_size if op.is_file() else sum(
                        f.stat().st_size for f in op.rglob("*") if f.is_file()
                    )
            throughput = (file_size / (1024 * 1024)) / elapsed if elapsed > 0 else 0.0

            self._log("DECOMPRESSED", path.name, method, file_size, elapsed, throughput)

            if self.move_to_processed:
                dest = self.processed_decompress / path.name
                shutil.move(str(path), str(dest))

            with self._stats_lock:
                self._total_files += 1
                self._total_bytes_in += file_size
                self._total_bytes_out += out_size

        except Exception as exc:
            elapsed = time.time() - start
            self._log("ERROR", path.name, method or "unknown", file_size, elapsed)
            logger.error("Decompression failed for %s: %s", path.name, exc)

            error_log = path.parent / f".{path.name}.error.log"
            try:
                error_log.write_text(
                    f"timestamp: {_timestamp()}\n"
                    f"file: {path.name}\n"
                    f"method: {method}\n"
                    f"error: {exc}\n",
                    encoding="utf-8",
                )
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Queue consumer
    # ------------------------------------------------------------------

    def _process_queue(self) -> None:
        """Drain the queue and dispatch items to the thread pool."""
        pool = ThreadPoolExecutor(max_workers=self._workers)
        futures: list[Future] = []

        try:
            while not self._stop_event.is_set():
                try:
                    action, path = self._queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                if action == "compress":
                    fut = pool.submit(self._handle_compress, path)
                elif action == "decompress":
                    fut = pool.submit(self._handle_decompress, path)
                else:
                    continue

                futures.append(fut)

                # Prune completed futures periodically
                futures = [f for f in futures if not f.done()]
        finally:
            pool.shutdown(wait=True)

    # ------------------------------------------------------------------
    # Scan existing files on startup
    # ------------------------------------------------------------------

    def _scan_existing(self) -> None:
        """Queue any files already present in the watch folders."""
        for path in sorted(self.compress_dir.iterdir()):
            if path.is_file() and not self._should_skip(path):
                self._queue.put(("compress", path))

        for path in sorted(self.decompress_dir.iterdir()):
            if path.is_file() and not self._should_skip(path):
                self._queue.put(("decompress", path))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start watching. Blocks until interrupted with Ctrl+C."""
        self._ensure_folders()
        self._setup_logging()
        self._init_encoders()

        self._start_time = time.time()

        logger.info(
            "%s STARTED watch_root=%s threshold=%d MB workers=%d",
            _timestamp(),
            self.watch_root,
            self.gpu_threshold_mb,
            self._workers,
        )
        logger.info(
            "%s   compress/  -> %s",
            _timestamp(),
            self.compress_output,
        )
        logger.info(
            "%s   decompress/ -> %s",
            _timestamp(),
            self.decompress_output,
        )

        # Scan existing files before starting the observer
        self._scan_existing()

        # Set up watchdog observer
        self._observer = Observer()
        compress_handler = _WatchHandler(self._queue, "compress")
        decompress_handler = _WatchHandler(self._queue, "decompress")
        self._observer.schedule(compress_handler, str(self.compress_dir), recursive=False)
        self._observer.schedule(decompress_handler, str(self.decompress_dir), recursive=False)
        self._observer.start()

        # Start the queue-processing thread
        consumer = threading.Thread(target=self._process_queue, daemon=True)
        consumer.start()

        try:
            while not self._stop_event.is_set():
                self._stop_event.wait(timeout=1.0)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        """Gracefully stop the daemon and print summary."""
        self._stop_event.set()

        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)

        elapsed = time.time() - self._start_time if self._start_time else 0

        with self._stats_lock:
            summary = (
                f"\n{'=' * 60}\n"
                f"HammerIO Watch Summary\n"
                f"{'=' * 60}\n"
                f"  Total files processed : {self._total_files}\n"
                f"  Data in               : {_human_size(self._total_bytes_in)}\n"
                f"  Data out              : {_human_size(self._total_bytes_out)}\n"
                f"  Uptime                : {elapsed:.1f}s\n"
                f"{'=' * 60}"
            )

        logger.info(summary)
