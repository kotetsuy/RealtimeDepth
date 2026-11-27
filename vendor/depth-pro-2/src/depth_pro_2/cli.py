#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

from __future__ import annotations

import logging
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Callable, Mapping, Sequence

import av
import click
import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image

from .model import (
    DEFAULT_MODEL,
    DEFAULT_OVERLAP,
    DEFAULT_WINDOW,
    PRESETS,
    DepthPro2,
    create_model_and_transforms,
)
from .utils import default_device, get_supported_image_extensions, load_rgb, upright

VIDEO_EXTENSIONS = (".mov", ".mp4", ".m4v", ".avi", ".mkv", ".webm", ".mpg", ".mpeg", ".mts")

COLOR_LUT = (colormaps["turbo"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
#: Room above the nearest point of the frame that fixes the color scale, for nearer ones later.
COLOR_HEADROOM = 0.2

PRECISIONS = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

FRAGMENTED_MP4 = "frag_keyframe+empty_moov+default_base_moof"

#: Log-spaced depth bins from 1 cm to 10 km, and the pixel stride, of the summary's depth range.
DEPTH_BINS = np.logspace(-2, 4, 1001)
DEPTH_STRIDE = 8


@click.group()
def main_cli():
    """Run Depth Pro 2 inference."""


@main_cli.command("predict")
@click.option(
    "-i",
    "--input-path",
    type=click.Path(path_type=Path, exists=True),
    multiple=True,
    required=True,
    help="An image, a video, or a folder of them. Repeat -i for more; the model loads once.",
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
    help="Folder to write the depth to.",
)
@click.option(
    "-m",
    "--model",
    type=click.Choice(sorted(PRESETS)),
    default=DEFAULT_MODEL,
    show_default=True,
    help="Which Depth Pro 2 variant to run.",
)
@click.option(
    "--mode",
    type=click.Choice(["buffered", "unbuffered"]),
    default="buffered",
    show_default=True,
    help="Video input only: buffered processes overlapping clips of --window frames, so a frame "
    "also sees the frames after it. unbuffered predicts each frame as it arrives, from itself "
    "and the frames before it, for a fraction of the latency.",
)
@click.option(
    "--window",
    type=click.IntRange(min=1),
    default=DEFAULT_WINDOW,
    show_default=True,
    help="Video input only: frames attended together: the clip length (buffered) or the cache "
    "size, the current frame included (unbuffered).",
)
@click.option(
    "--overlap",
    type=click.IntRange(min=0),
    default=DEFAULT_OVERLAP,
    show_default=True,
    help="Video input only: frames shared between consecutive clips, fewer than --window. "
    "Buffered mode only; unbuffered has no clips to overlap.",
)
@click.option(
    "--precision",
    type=click.Choice(sorted(PRECISIONS)),
    default=None,
    help="Autocast dtype of the backbones and the decoder, the heads staying float32. By default "
    "bf16, as trained and evaluated, on a GPU and fp32 on the CPU, where bf16 is much slower. "
    "fp16 is closer to fp32 than bf16 is; fp32 runs without autocast.",
)
@click.option(
    "--save-npz",
    is_flag=True,
    help="Also write depth in meters and the focal length as .npz. Large for a video, which "
    "gets one file per frame: a 4K frame is 33 MB, so a minute of footage is about 60 GB.",
)
@click.option(
    "--save-png16",
    is_flag=True,
    help="Also write inverse depth as 16-bit grayscale PNG, for editing and compositing tools: "
    "white is near and black is infinitely far, on the color preview's scale.",
)
def predict_cli(
    input_path: tuple[Path, ...],
    output_path: Path,
    model: str,
    mode: str,
    window: int,
    overlap: int,
    precision: str | None,
    save_npz: bool,
    save_png16: bool,
):
    """Predict metric depth for images and videos.

    \b
    Writes into the output folder, for each input NAME.ext:
      NAME-depth.png                        inverse depth in color, for an image
      NAME-depth.mp4                        the same for a video
      NAME-depth.npz                        with --save-npz: depth in meters and focallength_px
      NAME-depth/000000.npz                 the same for a video, one file per frame
      NAME-depth16.png                      with --save-png16: inverse depth in 16-bit gray
      NAME-depth16/NAME-depth16.000000.png  the same for a video, one file per frame
    Colors follow the Turbo map: dark blue is infinitely far, where depth is inf.
    """  # noqa: D301 - click keeps a paragraph after \b as written.
    logging.basicConfig(format="%(message)s")
    logging.getLogger("depth_pro_2").setLevel(logging.INFO)

    if mode == "buffered" and overlap >= window:
        raise click.BadParameter(
            f"{overlap} must be less than --window ({window}).", param_hint="'--overlap'"
        )
    inputs = collect_inputs(input_path, output_path)
    device = default_device()
    net, transform = create_model_and_transforms(
        model, device, None if precision is None else PRECISIONS[precision]
    )
    click.echo(f"Loading {model} onto {device} in {str(net.precision).removeprefix('torch.')}.")
    output_path.mkdir(parents=True, exist_ok=True)
    for path in inputs:
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            predict_video(
                net, transform, path, output_path, mode, window, overlap, save_npz, save_png16
            )
        else:
            predict_image(net, transform, path, output_path, save_npz, save_png16)


def collect_inputs(paths: Sequence[Path], output_path: Path) -> list[Path]:
    """The images and videos given, folders expanded, checked to have distinct output names."""
    supported = VIDEO_EXTENSIONS + get_supported_image_extensions()
    inputs = []
    for path in paths:
        if path.is_dir():
            found = sorted(
                p
                for p in path.iterdir()
                if p.is_file() and p.suffix.lower() in supported and not p.name.startswith(".")
            )
            if path.resolve() == output_path.resolve():
                outputs = {f"{p.stem}-depth{tag}" for p in found for tag in ("", "16")}
                found = [p for p in found if p.stem not in outputs]
            if not found:
                raise click.UsageError(f"No supported image or video in {path}.")
            inputs += found
        elif path.suffix.lower() in supported:
            inputs.append(path)
        else:
            raise click.UsageError(
                f"No supported image or video found at {path}. Images are any format Pillow "
                f"reads, such as .jpg, .png, .heic and .avif. Videos are "
                f"{', '.join(VIDEO_EXTENSIONS)}."
            )
    # A photo and a video may share a name (a Live Photo): one writes a .png, the other an .mp4.
    # Names compare without case, as on macOS's default file system.
    names = Counter((p.stem.casefold(), p.suffix.lower() in VIDEO_EXTENSIONS) for p in inputs)
    shared = sorted({stem for (stem, _), count in names.items() if count > 1})
    if shared:
        raise click.UsageError(
            f"Inputs share a name, so their outputs would overwrite each other: "
            f"{', '.join(shared)}."
        )
    return inputs


def predict_image(
    net: DepthPro2,
    transform: Callable[[np.ndarray], torch.Tensor],
    input_path: Path,
    output_path: Path,
    save_npz: bool,
    save_png16: bool,
) -> None:
    """Predict one image, write its preview and arrays, and print a summary."""
    image, f_px = load_rgb(input_path)
    prediction = net.infer(transform(image), f_px=f_px)
    inverse_depth = inverse_metric_depth(prediction)

    png = output_path / f"{input_path.stem}-depth.png"
    scale = color_scale(inverse_depth)
    Image.fromarray(colorize(inverse_depth, scale)).save(png, compress_level=1)
    written = [str(png)]
    if save_npz:
        write_npz(png.with_suffix(".npz"), prediction)
        written.append(str(png.with_suffix(".npz")))
    if save_png16:
        png16 = output_path / f"{input_path.stem}-depth16.png"
        grayscale16(inverse_depth, scale).save(png16, compress_level=1)
        written.append(str(png16))

    counts = np.zeros(len(DEPTH_BINS), dtype=np.int64)
    count_depth(counts, inverse_depth)
    source = "from EXIF" if f_px is not None else "estimated"
    print_summary(written, prediction, source, counts, scale if save_png16 else None)


def predict_video(
    net: DepthPro2,
    transform: Callable[[np.ndarray], torch.Tensor],
    input_path: Path,
    output_path: Path,
    mode: str,
    window: int,
    overlap: int,
    save_npz: bool,
    save_png16: bool,
) -> None:
    """Predict a video frame by frame, write its preview and arrays, and print a summary."""
    video_path = output_path / f"{input_path.stem}-depth.mp4"
    npz_dir = output_path / f"{input_path.stem}-depth"
    png16_dir = output_path / f"{input_path.stem}-depth16"
    if save_npz:
        npz_dir.mkdir(exist_ok=True)
    if save_png16:
        png16_dir.mkdir(exist_ok=True)
    counts = np.zeros(len(DEPTH_BINS), dtype=np.int64)
    scale = prediction = None
    count = 0

    with av.open(str(input_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        fps = float(stream.average_rate or 30.0)
        expected = stream.frames or round((container.duration or 0) / av.time_base * fps)
        frames = (transform(upright(frame)) for frame in container.decode(stream))
        predictions = net.infer_video(frames, window, overlap, mode=mode)
        with (
            VideoWriter(video_path, fps) as writer,
            click.progressbar(predictions, expected or None, label=input_path.name) as bar,
        ):
            for prediction in bar:
                inverse_depth = inverse_metric_depth(prediction)
                if scale is None:
                    scale = color_scale(inverse_depth)
                writer.write(colorize(inverse_depth, scale))
                if save_npz:
                    write_npz(npz_dir / f"{count:06d}.npz", prediction)
                if save_png16:
                    grayscale16(inverse_depth, scale).save(
                        png16_dir / f"{png16_dir.name}.{count:06d}.png", compress_level=1
                    )
                count_depth(counts, inverse_depth)
                count += 1

    if prediction is None:
        raise click.ClickException(f"No frames decoded from {input_path}.")
    written = [f"{count} frames at {fps:.5g} fps to {video_path}"]
    if save_npz:
        written.append(f"{npz_dir}/")
    if save_png16:
        written.append(f"{png16_dir}/")
    source = f"estimated from the first {min(window, count)} frames"
    print_summary(written, prediction, source, counts, scale if save_png16 else None)


def inverse_metric_depth(prediction: Mapping[str, torch.Tensor]) -> np.ndarray:
    """``1 / depth`` in 1/m as a float32 array on the host, zero where infinitely far."""
    inverse_depth = prediction["inverse_depth"].float().cpu().numpy()
    return inverse_depth * (inverse_depth.shape[-1] / float(prediction["focallength_px"]))


def write_npz(path: Path, prediction: Mapping[str, torch.Tensor]) -> None:
    """Write depth in meters and the focal length in pixels to an uncompressed npz."""
    np.savez(
        path,
        depth=prediction["depth"].float().cpu().numpy(),
        focallength_px=np.float32(prediction["focallength_px"].item()),
    )


def color_scale(inverse_depth: np.ndarray) -> float:
    """Inverse depth mapped to the top of the color map, anchored on one frame of a clip."""
    high = float(inverse_depth.max())
    return high * (1.0 + COLOR_HEADROOM) if high > 0 else 1.0


def colorize(inverse_depth: np.ndarray, scale: float) -> np.ndarray:
    """Color-map inverse depth from 0, at infinite depth, to ``scale`` as RGB uint8."""
    return COLOR_LUT[np.clip(inverse_depth * (256 / scale), 0, 255).astype(np.int32)]


def grayscale16(inverse_depth: np.ndarray, scale: float) -> Image.Image:
    """Inverse depth from 0, at infinite depth, to ``scale`` as a 16-bit grayscale image."""
    return Image.fromarray((np.clip(inverse_depth / scale, 0, 1) * 65535).round().astype(np.uint16))


class VideoWriter:
    """Encode frames into an H.264 MP4 that stays playable while it is being written."""

    def __init__(self, path: Path, fps: float):
        """Defer opening until the first frame fixes the resolution."""
        self.path = path
        self.fps = fps
        self.container: av.container.OutputContainer | None = None

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *_) -> None:
        """Flush the encoder and finish the file."""
        if self.container is not None:
            self.container.mux(self.stream.encode())
            self.container.close()

    def write(self, image: np.ndarray) -> None:
        """Append one ``(H, W, 3)`` uint8 frame, edge-padded to even sides if it has an odd one."""
        height, width = image.shape[:2]
        if height % 2 or width % 2:
            image = np.pad(image, ((0, height % 2), (0, width % 2), (0, 0)), mode="edge")
        if self.container is None:
            self.container = av.open(
                str(self.path), "w", options={"movflags": FRAGMENTED_MP4, "flush_packets": "1"}
            )
            self.stream = self.container.add_stream(
                "libx264", rate=Fraction(self.fps).limit_denominator(1001)
            )
            self.stream.height, self.stream.width = image.shape[:2]
            self.stream.pix_fmt = "yuv420p"
            # Tag what the RGB to YUV conversion produces, or QuickTime, Finder and Photos assume
            # BT.709 and shift the colors: BT.601 limited range, from sRGB.
            self.stream.colorspace = 6  # BT.601 (SMPTE 170M)
            self.stream.color_range = 1  # limited (MPEG)
            self.stream.color_primaries = 1  # BT.709, the primaries of sRGB
            self.stream.color_trc = 13  # sRGB
            self.stream.gop_size = max(1, round(self.fps))
            # B-frames would delay the first frame: a fragmented MP4 has no edit list to undo it.
            self.stream.max_b_frames = 0
        self.container.mux(self.stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")))


def count_depth(counts: np.ndarray, inverse_depth: np.ndarray) -> None:
    """Add a strided sample of one frame's depth to ``counts``, the last bin infinite depth."""
    sample = inverse_depth[::DEPTH_STRIDE, ::DEPTH_STRIDE]
    near = sample[sample > 0]
    counts[:-1] += np.histogram(1 / near, DEPTH_BINS)[0]
    counts[-1] += sample.size - near.size


def print_summary(
    written: list[str],
    prediction: Mapping[str, torch.Tensor],
    source: str,
    counts: np.ndarray,
    png16_scale: float | None,
) -> None:
    """Print what was written, the focal length of the last frame, and the depth range.

    Given ``png16_scale``, also how to turn the 16-bit PNG's values into meters.
    """
    click.echo(f"Wrote {', '.join(written)}.")
    click.echo(f"Focal length: {float(prediction['focallength_px']):.1f} px, {source}.")
    finite = counts[:-1]
    depth = ""
    if finite.sum():
        centers = np.sqrt(DEPTH_BINS[:-1] * DEPTH_BINS[1:])
        low, high = centers[np.searchsorted(np.cumsum(finite) / finite.sum(), (0.05, 0.95))]
        depth = f"{low:.2f} m to {high:.2f} m (5th to 95th percentile), "
    click.echo(f"Depth: {depth}{100 * counts[-1] / max(1, counts.sum()):.1f}% infinitely far.")
    if png16_scale is not None:
        k = 65535 / png16_scale
        click.echo(f"16-bit PNG: depth in meters = {k:.2f} / value, 0 is infinitely far.")
