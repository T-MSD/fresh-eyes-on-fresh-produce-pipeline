from dataclasses import dataclass
from pathlib import Path

from src.dataset.errors import InvalidInput
from src.dataset.frames import FrameFile, FrameScan, scan_frames
from src.dataset.planogram import Planogram, parse_planogram


@dataclass(frozen=True)
class Dataset:
    planogram_path: Path
    images_dir: Path
    planogram: Planogram
    frames: tuple[FrameFile, ...]
    skipped: tuple[Path, ...]
    cameras_without_frames: tuple[str, ...]


def read_dataset(planogram_path: Path, images_dir: Path) -> Dataset:
    """Read, parse and cross-check a planogram and its images; raise InvalidInput on problems."""
    planogram_path, images_dir = planogram_path.resolve(), images_dir.resolve()
    planogram = parse_planogram(planogram_path.read_bytes())
    return build_dataset(planogram_path, planogram, scan_frames(images_dir))


def build_dataset(planogram_path: Path, planogram: Planogram, scan: FrameScan) -> Dataset:
    framed = {frame.camera for frame in scan.frames}
    unknown = sorted(framed - set(planogram.cameras))
    if unknown:
        raise InvalidInput(
            [f"camera folder {camera!r} has frames but no planogram entry" for camera in unknown]
        )
    if not scan.frames:
        raise InvalidInput([f"no frames found in {scan.images_dir}"])
    return Dataset(
        planogram_path=planogram_path,
        images_dir=scan.images_dir,
        planogram=planogram,
        frames=scan.frames,
        skipped=scan.skipped,
        cameras_without_frames=tuple(c for c in planogram.cameras if c not in framed),
    )
