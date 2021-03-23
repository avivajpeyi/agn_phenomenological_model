import os

from PIL import Image
from fpdf import FPDF


def combine_images_to_pdf():
    pdf = FPDF()
    sdir = "/home/avi.vajpeyi/public_html/agn_pop/simulated_events/pe_set_3/"
    w, h = 0, 0

    for i in range(0, 100):
        fname = sdir + f"inj{i}_data0_0_analysis_H1L1_dynesty_result.png"
        if os.path.exists(fname):
            if i == 1:
                cover = Image.open(fname)
                w, h = cover.size
                pdf = FPDF(unit="pt", format=[w, h])
            image = fname
            pdf.add_page()
            pdf.image(image, 0, 0, w, h)
        else:
            print("File not found:", fname)
        print("processed %d" % i)
    pdf.output(sdir + "output.pdf", "F")
    print("done")


if __name__ == '__main__':
    combine_images_to_pdf()
