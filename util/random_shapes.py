from PIL import Image, ImageDraw
from math import sqrt, ceil
from random import randrange, choice, sample

ELIPSE, RECT, CHORD, TRI = 0, 1, 2, 3
SHAPE_TYPES = [ELIPSE, RECT, CHORD, TRI]


def random_shapes(image_shape=(256, 256)):
    img = Image.new('RGB', image_shape, color=(240, 240, 240))
    d = ImageDraw.Draw(img)

    w, h = image_shape
    msw, msh = ceil(w / 3), ceil(h / 3)

    y = 0
    while y < h:
        x = 0
        max_y = y
        while x < w:
            bw = randrange(ceil(msw / 2), msw)
            bh = randrange(ceil(msh / 2), msh)
            hp = ceil(bw / 3)
            vp = ceil(bh / 3)
            shape_type = choice(SHAPE_TYPES)

            l = x  # left
            t = y  # top
            r = x + bw  # right
            b = y + bh  # bottom

            # Bounding box with skew
            corners = [
                (randrange(l, l + hp), randrange(t, t + vp)),  # tl
                (randrange(r - hp, r), randrange(t, t + hp)),  # tr
                (randrange(r - hp, r), randrange(b - vp, b)),  # tr
                (randrange(l, l + hp), randrange(b - vp, b)),  # bl
            ]

            if shape_type == RECT:
                d.polygon(corners, outline='#333')
            elif shape_type == TRI:
                d.polygon(sample(corners, 3), outline='#333')
            elif shape_type == CHORD:
                d.chord(sample(corners, 2), randrange(0, 360), randrange(0, 360), outline='#333')
            elif shape_type == ELIPSE:
                d.ellipse(choice([(corners[0], corners[2]), (corners[1], corners[3])]), outline='#333')

            x += bw
            if y + bh > max_y:
                max_y = y + bh
        y = max_y
    del d

    return img
