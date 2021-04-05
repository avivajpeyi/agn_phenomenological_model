import glob
from typing import List

from PIL import Image
from bilby_report.tools import image_utils
from fpdf import FPDF
from tqdm import tqdm

from ..agn_logger import logger


def join_images(*rows, bg_color=(0, 0, 0, 0), alignment=(0.5, 0.5)):
    rows = [
        [image.convert('RGBA') for image in row]
        for row
        in rows
    ]

    heights = [
        max(image.height for image in row)
        for row
        in rows
    ]

    widths = [
        max(image.width for image in column)
        for column
        in zip(*rows)
    ]

    tmp = Image.new(
        'RGBA',
        size=(sum(widths), sum(heights)),
        color=bg_color
    )

    for i, row in enumerate(rows):
        for j, image in enumerate(row):
            y = sum(heights[:i]) + int((heights[i] - image.height) * alignment[1])
            x = sum(widths[:j]) + int((widths[j] - image.width) * alignment[0])
            tmp.paste(image, (x, y))

    return tmp


def join_images_horizontally(*row, bg_color=(0, 0, 0), alignment=(0.5, 0.5)):
    return join_images(
        row,
        bg_color=bg_color,
        alignment=alignment
    )


def join_images_vertically(*column, bg_color=(0, 0, 0), alignment=(0.5, 0.5)):
    return join_images(
        *[[image] for image in column],
        bg_color=bg_color,
        alignment=alignment
    )


def save_gif(gifname, regex="*.png", outdir="gif", loop=False, ):
    image_paths = glob.glob(f"{outdir}/{regex}")
    gif_filename = os.path.join(outdir, gifname)
    orig_len = len(image_paths)
    image_paths.sort()
    if loop:
        image_paths += image_paths[::-1]
    assert orig_len <= len(image_paths)
    image_utils.make_gif(
        image_paths=image_paths,
        duration=200,
        gif_save_path=gif_filename
    )
    print(f"Saved gif {gif_filename}")


def combine_images_to_pdf(imgpaths: List[str], pdfpath: str):
    pdf = FPDF()
    w, h = 0, 0
    for i, imgpath in enumerate(tqdm(imgpaths, desc="Adding images to pdf")):
        if i == 0:  # setup size of PDF pages
            cover = Image.open(imgpath)
            w, h = cover.size
            pdf = FPDF(unit="pt", format=[w, h])
        image = imgpath
        pdf.add_page()
        pdf.image(image, 0, 0, w, h)
    pdf.output(pdfpath, "F")
    logger.info(f"Saved PDF to {pdfpath}")
