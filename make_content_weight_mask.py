import os.path
from typing import Optional

import cv2
import numpy as np


def resize_to_max_scale(image, max_scale: int):
    if image.shape[0] > max_scale or image.shape[1] > max_scale:
        scale_factor = min(max_scale / image.shape[0], max_scale / image.shape[1])
        disp_image = cv2.resize(image, dsize=None, fx=scale_factor, fy=scale_factor)
    else:
        disp_image = image
    return disp_image


WINDOW_NAME = 'Image'


def draw_mask_on_image(image: 'Array[(H,W,3),uint8]', max_scale: int = 1000) -> 'Array[(H,W,3),bool]':
    class Nonlocals:
        boxes = []
        done = False

    disp_image = resize_to_max_scale(image, max_scale=max_scale)

    def on_mouse(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            sbox = [x, y]
            Nonlocals.boxes.append(sbox)
            cv2.circle(disp_image, center=(x, y), radius=4, color=(0, 0, 255), thickness=2)

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse, 0)
    print('Click the corners of the polygon then click space to finish')
    mask = np.zeros(disp_image.shape[:2], dtype=np.uint8)
    while not Nonlocals.done:
        cv2.imshow(WINDOW_NAME, disp_image)
        key = cv2.waitKey(10)
        if key == 32:  # Space
            cv2.fillPoly(mask, np.array([Nonlocals.boxes]), 255)
            Nonlocals.boxes.clear()
        elif key == 13:  # Enter
            if len(Nonlocals.boxes) > 2:
                cv2.fillPoly(mask, np.array([Nonlocals.boxes]), 255)
            break

    final_mask = cv2.resize(mask, dsize=(image.shape[1], image.shape[0]))
    cv2.destroyAllWindows()
    return final_mask


def fade_mask_with_half_life(mask, half_life: float, base: float = 0):
    distances = cv2.distanceTransform((mask == 0).astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=5)
    scales = base + (1 - base) * 2 ** (-distances / half_life)
    return (scales * 255).astype(np.uint8)


def make_mask_image(content_path: str, mask_path: str, max_scale=1000):
    img = cv2.imread(content_path)
    assert img is not None, f"No image at {content_path}"

    mask = np.full(shape=img.shape[:2], fill_value=255, dtype=np.uint8)

    undo_history = []

    while True:

        alpha_img = (img * (mask[:, :, None] / 255.)).astype(np.uint8)
        cv2.imshow(WINDOW_NAME, resize_to_max_scale(alpha_img, max_scale=max_scale))
        cv2.waitKey(100)

        cmd, *args = input(f'Image is  ({img.shape[1]}x{img.shape[0]}) Options: "draw", "fade <halflife>", "undo", "save <path>"?  >>').split(' ')

        if cmd == 'draw':
            mask = draw_mask_on_image(img, max_scale=max_scale)
            undo_history.append(mask)
        elif cmd == 'fade':
            half_life = int(args[0])
            base = float(args[1]) if len(args) > 1 else 0.
            mask = fade_mask_with_half_life(mask, half_life=half_life, base=base)
            undo_history.append(mask)
        elif cmd == 'undo':
            if len(undo_history) == 0:
                print("Undo history empty")
            else:
                mask = undo_history.pop()
        elif cmd == 'save':
            file_path = os.path.expanduser(args[0] if len(args) > 0 else mask_path)
            cv2.imwrite(file_path, mask)
            print(f'Saved to {file_path}')
        elif cmd in ('q', 'quit'):
            break


if __name__ == '__main__':
    make_mask_image(
        # content_path='/home/peter.oconnor/Downloads/style_images/src/padmad.jpeg',
        # mask_path='/home/peter.oconnor/Downloads/style_images/src/padmad_mask.png',
        content_path='/home/peter.oconnor/Downloads/style_images/src/pad.jpeg',
        mask_path='/home/peter.oconnor/Downloads/style_images/src/pad_mask.png',
    )
