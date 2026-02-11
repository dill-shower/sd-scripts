"""
MLDB Dataset Support for sd-scripts
Поддержка формата MLDB для тренировки. Автоматическое обнаружение и интеграция.

Usage:
    from library.mldb_dataset import enable_mldb_support
    enable_mldb_support()
"""

import os
import io
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

# ============================================================================
#                              MLDB IMPORT
# ============================================================================

MLDB_AVAILABLE = False
MLDBReader = None
UNTAGGED_TAG = "__UNTAGGED__"

try:
    from mldb32 import MLDBReader, UNTAGGED_TAG
    MLDB_AVAILABLE = True
except ImportError:
    try:
        parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent not in sys.path:
            sys.path.insert(0, parent)
        from mldb32 import MLDBReader, UNTAGGED_TAG
        MLDB_AVAILABLE = True
    except ImportError:
        pass

# ============================================================================
#                              GLOBALS
# ============================================================================

MLDB_PREFIX = "mldb://"
_readers: Dict[Tuple[str, int], Any] = {}  # (path, pid) → reader
_discovery_cache: Dict[str, List[str]] = {}
_patches_installed = False
_originals = {}

# ============================================================================
#                              PATH UTILS
# ============================================================================

def is_mldb_path(path: str) -> bool:
    return isinstance(path, str) and path.startswith(MLDB_PREFIX)

def parse_mldb_path(path: str) -> Tuple[str, int]:
    rest = path[len(MLDB_PREFIX):]
    dir_path, idx = rest.rsplit('#', 1)
    return dir_path, int(idx)

def make_mldb_path(mldb_dir: str, index: int) -> str:
    return f"{MLDB_PREFIX}{mldb_dir}#{index}"

# ============================================================================
#                              READER MANAGEMENT
# ============================================================================

def get_reader(mldb_dir: str):
    mldb_dir = os.path.normpath(os.path.abspath(mldb_dir))
    key = (mldb_dir, os.getpid())
    if key not in _readers:
        logger.info(f"Opening MLDB: {mldb_dir} (pid={os.getpid()})")
        _readers[key] = MLDBReader(mldb_dir, use_mmap=True)
    return _readers[key]

def close_readers():
    pid = os.getpid()
    to_close = [k for k in _readers if k[1] == pid]
    for k in to_close:
        try:
            _readers[k].close()
        except Exception:
            pass
        del _readers[k]
    _discovery_cache.clear()

# ============================================================================
#                              DISCOVERY
# ============================================================================

def find_mldb_datasets(root_dir: str) -> List[str]:
    if not MLDB_AVAILABLE:
        return []
    root_dir = os.path.normpath(os.path.abspath(root_dir))
    if root_dir in _discovery_cache:
        return _discovery_cache[root_dir]

    datasets = []
    root = Path(root_dir)
    if root.exists() and root.is_dir():
        for manifest in root.rglob("manifest.json"):
            if (manifest.parent / "tag_index.json").exists():
                datasets.append(str(manifest.parent.absolute()))

    _discovery_cache[root_dir] = datasets
    return datasets

def get_mldb_dirs_set(directory: str) -> Set[str]:
    return {os.path.normpath(os.path.abspath(d)) for d in find_mldb_datasets(directory)}

def is_inside_mldb(path: str, mldb_dirs: Set[str]) -> bool:
    path_abs = os.path.normpath(os.path.abspath(path))
    for mldb_dir in mldb_dirs:
        if path_abs.startswith(mldb_dir + os.sep) or path_abs == mldb_dir:
            return True
    return False

# ============================================================================
#                              JXL SIZE FROM BYTES
# ============================================================================

class _JXLBits:
    def __init__(self, data: bytes, offset: int = 0, offsets=None):
        self.data = data
        self.pos = offset
        self.shift = 0
        self.buf = bytearray()
        self.offsets = offsets
        if offsets:
            self.pos = offsets[0][1]
            self.prev_len = 0
            self.idx = 0

    def get(self, n: int) -> int:
        if self.offsets and self.shift + n > self.prev_len + self.offsets[self.idx][2]:
            self._partial(n)
        else:
            need = (self.shift + n + 7) // 8
            while len(self.buf) < need and self.pos < len(self.data):
                self.buf.append(self.data[self.pos])
                self.pos += 1
        mask = (1 << n) - 1
        val = (int.from_bytes(self.buf, "little") >> self.shift) & mask
        self.shift += n
        return val

    def _partial(self, n: int):
        while self.shift + n > self.prev_len + self.offsets[self.idx][2]:
            chunk_end = self.prev_len + self.offsets[self.idx][2]
            to_read = chunk_end - self.shift
            if to_read > 0:
                end = min(self.pos + to_read, len(self.data))
                self.buf.extend(self.data[self.pos:end])
                self.pos = end
            self.prev_len += self.offsets[self.idx][2]
            self.idx += 1
            if self.idx < len(self.offsets):
                self.pos = self.offsets[self.idx][1]


def _jxl_decode_size(bits: _JXLBits) -> Tuple[int, int]:
    bits.get(16)
    div8 = bits.get(1)
    if div8:
        h = 8 * (1 + bits.get(5))
    else:
        d = bits.get(2)
        h = 1 + bits.get([9, 13, 18, 30][d])
    ratio = bits.get(3)
    if div8 and not ratio:
        w = 8 * (1 + bits.get(5))
    elif not ratio:
        d = bits.get(2)
        w = 1 + bits.get([9, 13, 18, 30][d])
    else:
        w = [h, h, h * 12 // 10, h * 4 // 3, h * 3 // 2, h * 16 // 9, h * 5 // 4, h * 2][ratio]
    return w, h


def get_jxl_size_from_bytes(data: bytes) -> Tuple[int, int]:
    if data[:2] == b'\xff\x0a':
        return _jxl_decode_size(_JXLBits(data))

    if data[:12] != bytes.fromhex("0000000C4A584C200D0A870A"):
        raise ValueError("Invalid JXL signature")
    if data[12:32] != bytes.fromhex("000000146674796A786C20000000006A786C20"):
        raise ValueError("Invalid JXL ftyp")

    ptr = 32
    offset = 0
    offsets = []
    file_size = len(data)

    while ptr < file_size:
        lbox = int.from_bytes(data[ptr:ptr + 4], "big")
        if lbox == 1:
            xlbox = int.from_bytes(data[ptr + 8:ptr + 16], "big")
            hdr_len, box_len = 16, xlbox
        elif lbox == 0:
            hdr_len, box_len = 8, file_size - ptr
        else:
            hdr_len, box_len = 8, lbox

        box_type = data[ptr + 4:ptr + 8]
        if box_type == b'jxlc':
            offset = ptr + hdr_len
            break
        elif box_type == b'jxlp':
            idx = int.from_bytes(data[ptr + hdr_len:ptr + hdr_len + 4], "big")
            offsets.append([idx, ptr + hdr_len + 4, box_len - hdr_len - 4])
        ptr += box_len

    if offsets:
        offsets.sort(key=lambda x: x[0])
    return _jxl_decode_size(_JXLBits(data, offset, offsets if offsets else None))

# ============================================================================
#                              IMAGE SIZE
# ============================================================================

def get_mldb_image_size(mldb_dir: str, index: int) -> Tuple[int, int]:
    """Получить размер: meta → imagesize(BytesIO) → JXL decoder → PIL"""
    try:
        reader = get_reader(mldb_dir)
        meta = reader.get_meta(index)
        if meta is None:
            return (512, 512)

        # 1. Из метаданных
        if meta.width > 0 and meta.height > 0:
            return (meta.width, meta.height)

        # 2. Нужно загрузить данные
        record = reader[index]
        data = record.image_data
        fmt = (meta.format or "").lower()

        # 3. JXL — специальный декодер
        if fmt == "jxl":
            try:
                return get_jxl_size_from_bytes(data)
            except Exception:
                pass

        # 4. imagesize через BytesIO (быстрее PIL, не декодирует пиксели)
        try:
            import imagesize
            w, h = imagesize.get(io.BytesIO(data))
            if w > 0 and h > 0:
                return (w, h)
        except Exception:
            pass

        # 5. PIL fallback
        try:
            with Image.open(io.BytesIO(data)) as img:
                return img.size
        except Exception:
            pass

        return (512, 512)
    except Exception as e:
        logger.warning(f"get_mldb_image_size failed: {mldb_dir}#{index}: {e}")
        return (512, 512)

# ============================================================================
#                              LOAD IMAGE
# ============================================================================

def load_mldb_image(mldb_dir: str, index: int, alpha: bool = False) -> np.ndarray:
    reader = get_reader(mldb_dir)
    record = reader[index]
    img = Image.open(io.BytesIO(record.image_data))
    if alpha:
        if img.mode != "RGBA":
            img = img.convert("RGBA")
    else:
        if img.mode != "RGB":
            img = img.convert("RGB")
    return np.array(img, np.uint8)

# ============================================================================
#                              PATCHES
# ============================================================================

def _patched_load_image(image_path: str, alpha: bool = False) -> np.ndarray:
    if is_mldb_path(image_path):
        d, i = parse_mldb_path(image_path)
        return load_mldb_image(d, i, alpha)
    return _originals['load_image'](image_path, alpha)


def _patched_get_image_size(self, image_path: str) -> Tuple[int, int]:
    if is_mldb_path(image_path):
        d, i = parse_mldb_path(image_path)
        return get_mldb_image_size(d, i)
    return _originals['get_image_size'](self, image_path)


def _patched_glob_images(directory: str, base: str = "*") -> List[str]:
    mldb_dirs = get_mldb_dirs_set(directory)
    images = _originals['glob_images'](directory, base)
    if mldb_dirs:
        images = [p for p in images if not is_inside_mldb(p, mldb_dirs)]
    return images


def _make_patched_dreambooth():
    import library.train_util as tu
    Original = _originals['DreamBoothDataset']
    ImageInfo = tu.ImageInfo

    class MLDBDreamBoothDataset(Original):
        def __init__(self, *args, **kwargs):
            # Извлекаем subsets до того, как parent отбросит пустые
            if 'subsets' in kwargs:
                all_subsets = list(kwargs['subsets'])
            elif args:
                all_subsets = list(args[0])
            else:
                all_subsets = []

            is_training = kwargs.get('is_training_dataset', args[1] if len(args) > 1 else True)

            super().__init__(*args, **kwargs)

            if not MLDB_AVAILABLE:
                return

            processed = set()
            added = 0

            for subset in all_subsets:
                if not subset.image_dir:
                    continue

                for mldb_dir in find_mldb_datasets(subset.image_dir):
                    if mldb_dir in processed:
                        continue
                    processed.add(mldb_dir)

                    try:
                        reader = get_reader(mldb_dir)
                        total = len(reader)
                        logger.info(f"  MLDB {os.path.basename(mldb_dir)}: {total:,} images")

                        num_repeats = subset.num_repeats if is_training else 1
                        is_reg = getattr(subset, 'is_reg', False)

                        for idx, meta in reader.iter_meta():
                            if reader.is_untagged(meta):
                                caption = getattr(subset, 'class_tokens', "") or ""
                            else:
                                caption = ", ".join(reader.get_tags_by_ids(meta.tag_ids))

                            mldb_path = make_mldb_path(mldb_dir, idx)
                            info = ImageInfo(
                                image_key=mldb_path,
                                num_repeats=num_repeats,
                                caption=caption,
                                is_reg=is_reg,
                                absolute_path=mldb_path
                            )

                            if meta.width > 0 and meta.height > 0:
                                info.image_size = (meta.width, meta.height)

                            info.resize_interpolation = (
                                subset.resize_interpolation
                                if subset.resize_interpolation else self.resize_interpolation
                            )

                            self.register_image(info, subset)
                            added += 1

                            if is_reg:
                                self.num_reg_images += num_repeats
                            else:
                                self.num_train_images += num_repeats

                        # Subset мог быть отброшен parent'ом — добавляем обратно
                        if subset not in self.subsets:
                            subset.img_count = total
                            self.subsets.append(subset)

                    except Exception as e:
                        logger.error(f"Failed to load MLDB {mldb_dir}: {e}")

            if added:
                logger.info(f"Added {added:,} images from MLDB")

    return MLDBDreamBoothDataset
# ============================================================================
#                              PUBLIC API
# ============================================================================

def enable_mldb_support() -> bool:
    global _patches_installed

    if _patches_installed:
        return True
    if not MLDB_AVAILABLE:
        logger.warning("mldb32 not available — MLDB support disabled")
        return False

    try:
        import library.train_util as tu

        _originals['load_image'] = tu.load_image
        _originals['get_image_size'] = tu.BaseDataset.get_image_size
        _originals['glob_images'] = tu.glob_images
        _originals['DreamBoothDataset'] = tu.DreamBoothDataset

        tu.load_image = _patched_load_image
        tu.BaseDataset.get_image_size = _patched_get_image_size
        tu.glob_images = _patched_glob_images
        tu.DreamBoothDataset = _make_patched_dreambooth()

        _patches_installed = True
        logger.info("MLDB support enabled")
        return True

    except Exception as e:
        logger.error(f"Failed to enable MLDB: {e}")
        return False


def disable_mldb_support():
    global _patches_installed
    if not _patches_installed:
        return
    try:
        import library.train_util as tu
        for k, v in _originals.items():
            if k == 'load_image':
                tu.load_image = v
            elif k == 'get_image_size':
                tu.BaseDataset.get_image_size = v
            elif k == 'glob_images':
                tu.glob_images = v
            elif k == 'DreamBoothDataset':
                tu.DreamBoothDataset = v
    except Exception:
        pass
    _originals.clear()
    close_readers()
    _patches_installed = False


def is_mldb_enabled() -> bool:
    return _patches_installed


import atexit
atexit.register(close_readers)
