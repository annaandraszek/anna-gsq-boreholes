import pandas as pd
from PyPDF2 import PdfFileReader
import os
import re
import img2pdf
from PIL.Image import DecompressionBombError


def count_pages(file):
    text = True
    type = 'not text'
    if '.tif' in file:  # convert tif to pdf, save to new file name, and continue with new file name
        fname_out = re.sub('.tif', '.pdf', file)
        with open(fname_out, "wb") as f:
            try:
                f.write(img2pdf.convert(open(file, "rb")))
            except DecompressionBombError as e:
                print(e, "\n", file)
                return 0, 'too large to convert'
        file = fname_out
        text = False

    if '.pdf' in file:
        try:
            reader = PdfFileReader(open(file, "rb"))
        except Exception as e:
            print(e, "\n", file)
            return 0, 'not openable'
        if text:
            type = find_doc_type(reader)
        try:
            numpages = reader.getNumPages()
        except Exception as e:
            print(e, '\n', file)
            return 0, 'cannot get numpages'
    elif '.jpeg' in file or '.jpg' in file: # pages will be 1
        numpages = 1
    else:  # any other file type eg json
        numpages = 0

    return numpages, type


def is_doc_text_readable(docloc):
    try:
        reader = PdfFileReader(open(docloc, "rb"))
    except Exception as e:
        print(e, "\n", docloc)
        return False
    type = find_doc_type(reader)
    if type == 'text':
        return True
    else: return False


def find_doc_type(reader):
    try:
        page = reader.getPage(0)
        page_content = page.extractText()
        extractedText = page_content.encode('utf-8')

        if extractedText == b'':
            return 'not text'
        else:
            return 'text'

    except Exception as e:
        return 'not text'


def count_all_pages():
    data = []
    InputDir = r'C:\Users\andraszeka\OneDrive - ITP (Queensland Government)\gsq-boreholes\1000sample'

    for rootdir, subdirs, _ in os.walk(InputDir):
        for dir in subdirs:
            files = os.walk(InputDir + '/' + dir)
            for obj in files:
                for f in obj[2]:
                    Path = obj[0] + '/' + f
                    pages, type = count_pages(Path)
                    data.append([obj[0], f, type, pages])
            print("dir " + str(obj[0]) + " done")

    pgdf = pd.DataFrame(data, columns=['reportID', 'file', 'type', 'pages'])
    pgdf.to_csv('1000sampleinfo.csv', index=False)


if __name__ == "__main__":
    count_all_pages()