import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt

# generate calibration pattern
# usage: python generatePattern.py --square-size-cm 2.5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a print-ready chessboard calibration pattern."
    )
    parser.add_argument(
        "--corners-x",
        type=int,
        default=9,
        help="Number of internal chessboard corners horizontally. Default: 9",
    )
    parser.add_argument(
        "--corners-y",
        type=int,
        default=6,
        help="Number of internal chessboard corners vertically. Default: 6",
    )
    parser.add_argument(
        "--square-size-cm",
        type=float,
        default=3.5,
        help="Physical side length of one chessboard square in cm. Default: 3.5",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution used for the PNG export. Default: 300",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("buoy_tracking_data/generated_patterns"),
        help="Directory where the PDF and PNG are saved.",
    )
    parser.add_argument(
        "--app-file",
        type=Path,
        default=Path("1_aruco_14_octogon.py"),
        help="Tracker app file that provides create_chessboard(). Default: 1_aruco_14_octogon.py",
    )
    return parser.parse_args()


def load_aruco_app(app_file):
    spec = importlib.util.spec_from_file_location("aruco_app", app_file)
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    return app


def save_pattern(image, square_size_cm, square_size_px, output_base, dpi):
    height_px, width_px = image.shape[:2]
    cm_per_px = square_size_cm / square_size_px
    width_in = (width_px * cm_per_px) / 2.54
    height_in = (height_px * cm_per_px) / 2.54

    fig = plt.figure(figsize=(width_in, height_in), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(image, cmap="gray", interpolation="nearest", vmin=0, vmax=255)
    ax.axis("off")

    pdf_path = output_base.with_suffix(".pdf")
    png_path = output_base.with_suffix(".png")
    fig.savefig(pdf_path, format="pdf", dpi=dpi)
    fig.savefig(png_path, format="png", dpi=dpi)
    plt.close(fig)

    return pdf_path, png_path


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    app = load_aruco_app(args.app_file)
    square_size_px = round((args.square_size_cm / 2.54) * args.dpi)
    app.CHESSBOARD_SQUARE_SIZE_MM = args.square_size_cm * 10
    image = app.create_chessboard(
        args.corners_x,
        args.corners_y,
        square_size_px=square_size_px,
        save=False,
    )

    square_label = str(args.square_size_cm).replace(".", "p")
    output_base = args.output_dir / (
        f"chessboard_{args.corners_x}x{args.corners_y}_corners_"
        f"{square_label}cm"
    )
    pdf_path, png_path = save_pattern(
        image,
        args.square_size_cm,
        square_size_px,
        output_base,
        args.dpi,
    )

    rows = args.corners_y + 1
    cols = args.corners_x + 1
    board_width_cm = cols * args.square_size_cm
    board_height_cm = rows * args.square_size_cm

    print(f"Saved PDF: {pdf_path}")
    print(f"Saved PNG: {png_path}")
    print(f"Internal corners: {args.corners_x} x {args.corners_y}")
    print(f"Squares: {cols} x {rows}")
    print(f"Square size: {args.square_size_cm} cm")
    print(f"Printed board size: {board_width_cm:.1f} cm x {board_height_cm:.1f} cm")
    if board_width_cm > 29.7 or board_height_cm > 21.0:
        print("Note: this is larger than A4 landscape. Use a smaller square size or larger paper.")
    print("Print the PDF at 100% / actual size.")


if __name__ == "__main__":
    main()
