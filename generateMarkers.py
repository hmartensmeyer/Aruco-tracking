import argparse
import importlib.util
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

# helper to generate markers
# --------------------------
# usage: python generateMarkers.py <NUMBER>
# it will save the marker in buoy_tracking_data/generated_patterns/aruco_marker_ID<NUMBER>.png

def load_aruco_app():
    script_path = Path(__file__).with_name("1_aruco_13.py")
    spec = importlib.util.spec_from_file_location("aruco_app", script_path)
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    return app


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate ArUco marker PNG files using the tracker app settings."
    )
    parser.add_argument(
        "marker_ids",
        nargs="+",
        type=int,
        help="One or more marker IDs to generate, for example: 23 24 25",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=600,
        help="Marker image size in pixels, excluding border. Default: 600",
    )
    parser.add_argument(
        "--border",
        type=int,
        default=50,
        help="White border width in pixels. Default: 50",
    )
    parser.add_argument(
        "--physical-size-cm",
        type=float,
        default=None,
        help=(
            "If set, also create a print-ready PDF where the ArUco square "
            "itself has this side length in centimeters."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI used when deriving pixel size from --physical-size-cm. Default: 300",
    )
    return parser.parse_args()


def marker_output_path(app, marker_id, suffix):
    filename_suffix = f"_ID{marker_id}"
    if marker_id == app.REFERENCE_MARKER_ID and app.USE_REFERENCE_MARKER:
        filename_suffix += "_REFERENCE"

    output_dir = Path(app.DATA_OUTPUT_DIRECTORY) / app.PATTERNS_DIRECTORY_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"aruco_marker{filename_suffix}.{suffix}"


def save_exact_size_pdf(image_bgr, pdf_path, marker_size_cm, size_px, border_px, dpi):
    total_size_cm = marker_size_cm * (size_px + 2 * border_px) / size_px
    total_size_inches = total_size_cm / 2.54

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(total_size_inches, total_size_inches), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(image_rgb)
    ax.axis("off")
    fig.savefig(pdf_path, format="pdf", dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()
    app = load_aruco_app()

    size_px = args.size
    if args.physical_size_cm is not None:
        size_px = round((args.physical_size_cm / 2.54) * args.dpi)

    for marker_id in args.marker_ids:
        image = app.generate_aruco_marker(
            marker_id,
            size_px=size_px,
            border_px=args.border,
            save=True,
        )

        if image is not None and args.physical_size_cm is not None:
            pdf_path = marker_output_path(app, marker_id, "pdf")
            save_exact_size_pdf(
                image,
                pdf_path,
                args.physical_size_cm,
                size_px,
                args.border,
                args.dpi,
            )
            print(
                f"Print-ready PDF saved to: {pdf_path} "
                f"(marker square: {args.physical_size_cm} cm)"
            )


if __name__ == "__main__":
    main()
