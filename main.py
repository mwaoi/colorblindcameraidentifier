import sys
from color_detector import ColorDetector


def main() -> None:
    try:
        ColorDetector(camera_index=0).run()
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
