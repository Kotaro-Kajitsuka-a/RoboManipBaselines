import argparse
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show an image and print clicked pixel coordinates."
    )
    parser.add_argument("image_path", type=Path)
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Display scale. Coordinates are always printed in original image pixels.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    assert args.scale > 0.0

    image = cv2.imread(str(args.image_path), cv2.IMREAD_COLOR)
    assert image is not None, f"Failed to read image: {args.image_path}"

    display_image = image
    if args.scale != 1.0:
        display_image = cv2.resize(image, None, fx=args.scale, fy=args.scale)

    window_name = str(args.image_path)

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        original_x = int(round(x / args.scale))
        original_y = int(round(y / args.scale))
        original_x = min(max(original_x, 0), image.shape[1] - 1)
        original_y = min(max(original_y, 0), image.shape[0] - 1)

        b, g, r = image[original_y, original_x]
        print(
            f"x={original_x}, y={original_y}, "
            f"rgb=({int(r)}, {int(g)}, {int(b)}), "
            f"hex=#{int(r):02X}{int(g):02X}{int(b):02X}"
        )

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)

    print(f"image: {args.image_path}")
    print(f"size : width={image.shape[1]}, height={image.shape[0]}")
    print("left click: print pixel coordinate")
    print("q or Esc   : quit")

    while True:
        cv2.imshow(window_name, display_image)
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
