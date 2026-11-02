from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

from detector import DEFAULT_MODEL, ObjectDetector, annotate_frame

ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "runs" / "detect"
IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
WINDOW_NAME = "YOLO Detection"
STREAM_PREFIXES = ("http://", "https://", "rtsp://", "rtmp://")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yolo-detect",
        description="Deteksi objek YOLO dari webcam, gambar, video, atau stream.",
    )
    parser.add_argument(
        "-s",
        "--source",
        default="0",
        help="Indeks webcam (0), file gambar/video, folder gambar, atau URL stream. Default: 0",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help="Path model, contoh: yolo26n.pt. Default: yolo26n.pt di folder proyek",
    )
    parser.add_argument(
        "-c",
        "--conf",
        type=float,
        default=0.35,
        help="Ambang confidence 0-1. Default: 0.35",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Ukuran inferensi. Default: 640",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, atau cuda:0. Default: auto",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=None,
        metavar="ID",
        help="Filter kelas, contoh: --classes 0 2 5",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Simpan hasil anotasi ke runs/detect",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Path output file atau folder (otomatis mengaktifkan --save)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Jalankan tanpa jendela preview",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Tampilkan detail objek untuk setiap frame",
    )
    parser.add_argument(
        "--list-classes",
        action="store_true",
        help="Tampilkan daftar kelas model lalu keluar",
    )
    return parser


def is_webcam(source: str) -> bool:
    return source.isdigit()


def is_stream_url(source: str) -> bool:
    return source.lower().startswith(STREAM_PREFIXES)


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def probe_fps(source: str) -> float:
    if is_webcam(source) or is_stream_url(source):
        return 30.0
    capture = cv2.VideoCapture(source)
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
    finally:
        capture.release()
    return fps if 0 < fps < 1000 else 30.0


def resolve_output(
    args: argparse.Namespace,
    source: str,
    single_image: bool,
    source_is_dir: bool,
) -> Path | None:
    if not args.save and args.output is None:
        return None
    if args.output is not None:
        return Path(args.output)

    stamp = timestamp()
    directory = OUTPUT_ROOT / f"predict-{stamp}"
    if single_image:
        return directory / Path(source).name
    if is_webcam(source):
        return directory / f"webcam-{stamp}.mp4"
    if source_is_dir:
        return directory
    return directory / f"{Path(source).stem}-{stamp}.mp4"


def print_summary(totals: Counter) -> None:
    if not totals:
        print("Tidak ada objek yang terdeteksi.")
        return
    print("Ringkasan deteksi:")
    for name, count in totals.most_common():
        print(f"  {name:<20} {count}")


def show_frame(frame, blocking: bool = False) -> bool:
    cv2.imshow(WINDOW_NAME, frame)
    if blocking:
        while True:
            if cv2.waitKey(30) != -1:
                return False
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                return False
    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
        return False
    try:
        return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


class VideoOutput:
    def __init__(self, path: Path, fps: float = 30.0) -> None:
        self.path = path
        self.fps = fps
        self.writer: cv2.VideoWriter | None = None

    def write(self, frame) -> None:
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            size = (frame.shape[1], frame.shape[0])
            self.writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, size)
            if not self.writer.isOpened():
                raise RuntimeError(f"tidak bisa membuka penulis video {self.path}")
        self.writer.write(frame)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None


def run_image(
    detector: ObjectDetector,
    source: str,
    args: argparse.Namespace,
    target: Path | None,
) -> int:
    result = detector.predict(source, stream=False, classes=args.classes)[0]
    counts = detector.counts(result)
    frame = result.plot()

    if not args.no_show:
        show_frame(frame, blocking=True)
        cv2.destroyAllWindows()

    if target is not None:
        if target.is_dir():
            target = target / Path(source).name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(target), frame):
            print(f"Gagal menyimpan hasil ke {target}", file=sys.stderr)
            return 1
        print(f"Hasil disimpan: {target}")

    print_summary(counts)
    return 0


def run_stream(
    detector: ObjectDetector,
    source: str,
    args: argparse.Namespace,
    target: Path | None,
) -> int:
    source_is_dir = not is_webcam(source) and not is_stream_url(source) and Path(source).is_dir()
    writer = None
    if target is not None and not source_is_dir:
        writer = VideoOutput(target, probe_fps(source))
    totals: Counter = Counter()
    smoothed_fps = 0.0
    last_time = time.perf_counter()

    try:
        results = detector.predict(
            int(source) if is_webcam(source) else source,
            stream=True,
            classes=args.classes,
        )
        for result in results:
            now = time.perf_counter()
            delta = now - last_time
            last_time = now
            if delta > 0:
                current_fps = 1.0 / delta
                smoothed_fps = (
                    current_fps if smoothed_fps == 0 else 0.9 * smoothed_fps + 0.1 * current_fps
                )

            counts = detector.counts(result)
            totals.update(counts)

            if args.verbose:
                detail = ", ".join(detector.details(result)) or "tidak ada objek"
                print(f"{result.path or source}: {detail}")

            frame = result.plot()
            if not args.no_show:
                annotate_frame(frame, counts, smoothed_fps)

            if writer is not None:
                writer.write(frame)

            if source_is_dir and target is not None:
                target.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(target / Path(result.path).name), frame)

            if not args.no_show and not show_frame(frame):
                break
    except KeyboardInterrupt:
        print("\nDihentikan pengguna.")
    except ConnectionError as exc:
        print(f"Gagal membuka sumber '{source}': {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"Gagal menyimpan hasil: {exc}", file=sys.stderr)
        return 1
    finally:
        if writer is not None:
            writer.close()
        if not args.no_show:
            cv2.destroyAllWindows()

    if writer is not None and target is not None:
        print(f"Hasil disimpan: {target}")
    if source_is_dir and target is not None:
        print(f"Hasil disimpan di: {target}")
    print_summary(totals)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model = args.model or (str(DEFAULT_MODEL) if DEFAULT_MODEL.exists() else "yolo26n.pt")
    detector = ObjectDetector(
        model,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
    )

    if args.list_classes:
        for class_id, name in detector.names.items():
            print(f"{class_id:>3}  {name}")
        return 0

    source = args.source
    if not is_webcam(source) and not is_stream_url(source) and not Path(source).exists():
        print(f"Sumber tidak ditemukan: {source}", file=sys.stderr)
        return 1

    if args.classes is not None:
        invalid = [class_id for class_id in args.classes if class_id not in detector.names]
        if invalid:
            print(f"ID kelas tidak valid: {invalid}", file=sys.stderr)
            return 1

    single_image = (
        not is_webcam(source)
        and not is_stream_url(source)
        and Path(source).is_file()
        and is_image(Path(source))
    )
    source_is_dir = (
        not is_webcam(source) and not is_stream_url(source) and Path(source).is_dir()
    )
    target = resolve_output(args, source, single_image, source_is_dir)

    print(f"Model  : {model}")
    print(f"Device : {detector.device}")
    print(f"Sumber : {source}")

    if single_image:
        return run_image(detector, source, args, target)
    return run_stream(detector, source, args, target)


if __name__ == "__main__":
    raise SystemExit(main())
