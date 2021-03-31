import logging
from typing import List

from PIL import Image
from fpdf import FPDF
from tqdm import tqdm

logger = logging.getLogger(__name__)


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
