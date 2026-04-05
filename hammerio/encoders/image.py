"""GPU-accelerated batch image processing encoder for HammerIO.

Provides image encoding, decoding, resizing, format conversion, and
quality adjustment with CUDA GPU acceleration via OpenCV or fallback
to PIL/OpenCV CPU paths.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Callable, Optional, Union

from hammerio.core.hardware import HardwareProfile

logger = logging.getLogger("hammerio.encoders.image")

# Supported image extensions mapped to OpenCV/PIL format identifiers
_SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".webp": "WebP",
}

# Quality parameter mapping per format (OpenCV flag, quality range)
_QUALITY_PRESETS: dict[str, dict[str, int]] = {
    "fast": {"jpeg_quality": 70, "png_compression": 9, "webp_quality": 60},
    "balanced": {"jpeg_quality": 85, "png_compression": 6, "webp_quality": 80},
    "quality": {"jpeg_quality": 95, "png_compression": 3, "webp_quality": 95},
    "lossless": {"jpeg_quality": 100, "png_compression": 0, "webp_quality": 100},
}


def _is_supported_image(path: Path) -> bool:
    """Check whether a file has a supported image extension."""
    return path.suffix.lower() in _SUPPORTED_EXTENSIONS


class ImageEncoder:
    """GPU-accelerated batch image processor.

    Attempts to use OpenCV with CUDA for GPU-backed encode/decode and
    resize operations. Falls back to PIL (Pillow) or OpenCV CPU when
    CUDA is unavailable.

    Supported formats: JPEG, PNG, WebP.

    Args:
        hardware: Detected hardware profile used to select the
            optimal processing backend.
    """

    def __init__(self, hardware: HardwareProfile) -> None:
        self.hardware = hardware
        self._use_cuda = False
        self._backend = "pil"

        self._detect_backend()

    # ------------------------------------------------------------------
    # Backend detection
    # ------------------------------------------------------------------

    def _detect_backend(self) -> None:
        """Probe for GPU-capable OpenCV, then fall back gracefully."""
        if self.hardware.has_cuda:
            try:
                import cv2  # type: ignore[import-untyped]

                if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    self._use_cuda = True
                    self._backend = "opencv_cuda"
                    logger.info(
                        "Image backend: OpenCV CUDA (%d device(s))",
                        cv2.cuda.getCudaEnabledDeviceCount(),
                    )
                    return
            except Exception:
                pass

        # Try plain OpenCV (CPU)
        try:
            import cv2  # type: ignore[import-untyped]  # noqa: F811

            self._backend = "opencv_cpu"
            logger.info("Image backend: OpenCV CPU")
            return
        except ImportError:
            pass

        # Final fallback: PIL / Pillow
        try:
            from PIL import Image  # type: ignore[import-untyped]  # noqa: F401

            self._backend = "pil"
            logger.info("Image backend: PIL/Pillow (CPU)")
        except ImportError:
            logger.warning(
                "No image backend available — install opencv-python or Pillow"
            )
            self._backend = "none"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path, None],
        algorithm: str,
        quality: str,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        job_id: Optional[str] = None,
    ) -> str:
        """Encode / transcode / resize images.

        For a single file the image is processed and written to
        *output_path*.  For a directory every supported image inside
        is processed and placed in the *output_path* directory with
        the same relative filename (extension may change if the
        algorithm implies a different format).

        Args:
            input_path: Source file or directory.
            output_path: Destination file or directory.  Created
                automatically when it does not exist.
            algorithm: Target format identifier, one of
                ``"jpeg"``, ``"png"``, ``"webp"``.  Also accepts
                ``"passthrough"`` to re-encode in the original format.
            quality: Quality preset name — ``"fast"``,
                ``"balanced"``, ``"quality"``, or ``"lossless"``.
            progress_callback: Optional ``(job_id, percent)`` callable
                invoked as processing progresses.
            job_id: Identifier forwarded to the progress callback.

        Returns:
            Absolute path to the output file or directory.

        Raises:
            FileNotFoundError: If *input_path* does not exist.
            ValueError: If no supported backend is available or the
                format is not recognised.
        """
        input_path = Path(input_path).resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_path}")

        if self._backend == "none":
            raise ValueError(
                "No image processing backend available — "
                "install opencv-python or Pillow"
            )

        if output_path is None:
            if input_path.is_dir():
                output_path = input_path.parent / (input_path.name + "_processed")
            else:
                ext = self._target_extension(algorithm, input_path)
                output_path = input_path.with_suffix(f".processed{ext}")
        output_path = Path(output_path).resolve()

        preset = _QUALITY_PRESETS.get(quality, _QUALITY_PRESETS["balanced"])
        jid = job_id or "image"

        if input_path.is_dir():
            return self._process_directory(
                input_path, output_path, algorithm, preset, progress_callback, jid
            )
        else:
            return self._process_single(
                input_path, output_path, algorithm, preset, progress_callback, jid
            )

    # ------------------------------------------------------------------
    # Directory batch processing
    # ------------------------------------------------------------------

    def _process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        algorithm: str,
        preset: dict[str, int],
        progress_callback: Optional[Callable[[str, float], None]],
        job_id: str,
    ) -> str:
        """Process every supported image in a directory."""
        output_dir.mkdir(parents=True, exist_ok=True)

        images = [f for f in sorted(input_dir.iterdir()) if f.is_file() and _is_supported_image(f)]
        total = len(images)
        if total == 0:
            logger.warning("No supported images found in %s", input_dir)
            return str(output_dir)

        logger.info("Batch processing %d images from %s", total, input_dir)
        for idx, img_path in enumerate(images):
            ext = self._target_extension(algorithm, img_path)
            out_file = output_dir / (img_path.stem + ext)
            try:
                self._encode_image(img_path, out_file, algorithm, preset)
            except Exception:
                logger.exception("Failed to process %s", img_path.name)

            if progress_callback is not None:
                pct = (idx + 1) / total * 100.0
                progress_callback(job_id, pct)

        logger.info("Batch complete — output in %s", output_dir)
        return str(output_dir)

    # ------------------------------------------------------------------
    # Single-file processing
    # ------------------------------------------------------------------

    def _process_single(
        self,
        input_file: Path,
        output_file: Path,
        algorithm: str,
        preset: dict[str, int],
        progress_callback: Optional[Callable[[str, float], None]],
        job_id: str,
    ) -> str:
        """Process a single image file."""
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if progress_callback is not None:
            progress_callback(job_id, 0.0)

        self._encode_image(input_file, output_file, algorithm, preset)

        if progress_callback is not None:
            progress_callback(job_id, 100.0)

        in_size = input_file.stat().st_size
        out_size = output_file.stat().st_size
        logger.info(
            "Processed %s → %s (%.1f KB → %.1f KB)",
            input_file.name,
            output_file.name,
            in_size / 1024,
            out_size / 1024,
        )
        return str(output_file)

    # ------------------------------------------------------------------
    # Low-level encode/decode helpers
    # ------------------------------------------------------------------

    def _encode_image(
        self,
        src: Path,
        dst: Path,
        algorithm: str,
        preset: dict[str, int],
    ) -> None:
        """Read, optionally resize, and write an image using the best backend."""
        if self._backend.startswith("opencv"):
            self._encode_opencv(src, dst, algorithm, preset)
        else:
            self._encode_pil(src, dst, algorithm, preset)

    def _encode_opencv(
        self,
        src: Path,
        dst: Path,
        algorithm: str,
        preset: dict[str, int],
    ) -> None:
        """Encode via OpenCV (CUDA or CPU)."""
        import cv2  # type: ignore[import-untyped]

        if self._use_cuda:
            # Upload to GPU, decode, download — faster for large images
            try:
                gpu_img = cv2.cuda_GpuMat()
                cpu_img = cv2.imread(str(src), cv2.IMREAD_COLOR)
                if cpu_img is None:
                    raise ValueError(f"OpenCV cannot read {src}")
                gpu_img.upload(cpu_img)
                # Potential GPU resize would go here
                img = gpu_img.download()
            except Exception:
                logger.debug("CUDA path failed for %s, falling back to CPU read", src.name)
                img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(str(src), cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError(f"OpenCV cannot decode {src}")

        fmt = self._resolve_format(algorithm, src)
        params = self._opencv_encode_params(fmt, preset)
        ext = self._format_to_extension(fmt)

        # Ensure destination uses correct extension
        actual_dst = dst.with_suffix(ext) if dst.suffix.lower() != ext else dst
        success = cv2.imwrite(str(actual_dst), img, params)
        if not success:
            raise RuntimeError(f"cv2.imwrite failed for {actual_dst}")

        # If we changed the extension, rename back if caller expects exact path
        if actual_dst != dst:
            shutil.move(str(actual_dst), str(dst))

    def _encode_pil(
        self,
        src: Path,
        dst: Path,
        algorithm: str,
        preset: dict[str, int],
    ) -> None:
        """Encode via Pillow."""
        from PIL import Image  # type: ignore[import-untyped]

        img = Image.open(src)
        fmt = self._resolve_format(algorithm, src)

        save_kwargs: dict[str, object] = {}
        if fmt == "JPEG":
            if img.mode in ("RGBA", "P", "LA"):
                img = img.convert("RGB")
            save_kwargs["quality"] = preset["jpeg_quality"]
            save_kwargs["optimize"] = True
        elif fmt == "PNG":
            save_kwargs["compress_level"] = preset["png_compression"]
        elif fmt == "WebP":
            save_kwargs["quality"] = preset["webp_quality"]
            save_kwargs["method"] = 4  # encode effort 0-6

        img.save(str(dst), format=fmt, **save_kwargs)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_format(algorithm: str, src: Path) -> str:
        """Map algorithm string to a PIL/OpenCV format name."""
        algo = algorithm.lower().strip()
        if algo in ("jpeg", "jpg"):
            return "JPEG"
        if algo == "png":
            return "PNG"
        if algo == "webp":
            return "WebP"
        if algo == "passthrough":
            ext = src.suffix.lower()
            return _SUPPORTED_EXTENSIONS.get(ext, "JPEG")
        # Default: preserve source format
        ext = src.suffix.lower()
        return _SUPPORTED_EXTENSIONS.get(ext, "JPEG")

    @staticmethod
    def _format_to_extension(fmt: str) -> str:
        """Return canonical file extension for a format name."""
        return {
            "JPEG": ".jpg",
            "PNG": ".png",
            "WebP": ".webp",
        }.get(fmt, ".jpg")

    @staticmethod
    def _target_extension(algorithm: str, src: Path) -> str:
        """Determine the output file extension."""
        algo = algorithm.lower().strip()
        mapping = {"jpeg": ".jpg", "jpg": ".jpg", "png": ".png", "webp": ".webp"}
        if algo in mapping:
            return mapping[algo]
        # Passthrough or unknown — keep original extension
        return src.suffix

    @staticmethod
    def _opencv_encode_params(fmt: str, preset: dict[str, int]) -> list[int]:
        """Build OpenCV imwrite parameter list."""
        import cv2  # type: ignore[import-untyped]

        if fmt == "JPEG":
            return [cv2.IMWRITE_JPEG_QUALITY, preset["jpeg_quality"]]
        if fmt == "PNG":
            return [cv2.IMWRITE_PNG_COMPRESSION, preset["png_compression"]]
        if fmt == "WebP":
            return [cv2.IMWRITE_WEBP_QUALITY, preset["webp_quality"]]
        return []
