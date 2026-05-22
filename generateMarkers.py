import argparse
import importlib.util
from pathlib import Path

# helper to generate markers
# --------------------------
# usage: python generateMarker.py <NUMBER>
# it will wave the marker in buoy_tracking_data/generate_patterns/aruco_marker_ID<NUMBER>.png

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
    return parser.parse_args()


def main():
    args = parse_args()
    app = load_aruco_app()

    for marker_id in args.marker_ids:
        app.generate_aruco_marker(
            marker_id,
            size_px=args.size,
            border_px=args.border,
            save=True,
        )


if __name__ == "__main__":
    main()