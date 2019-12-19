import json
import pandas as pd
import numpy as np
import toc_classification
import settings



# transform document pages into dataset of pages for toc classification, classify pages, and isolate toc
# from toc page, transform content into dataset of headings for heading identification, identify headings, and return headings and subheadings
#for page in doc:


def create_toc_dataset(docid, outfile='_toc_dataset.csv'):
    pageinfo = settings.get_restructpageinfo_file(docid)
    pagelines = settings.get_restructpagelines_file(docid)
    pi = json.load(open(pageinfo, "r"))
    pl = json.load(open(pagelines, "r"))

    docset = np.zeros((len(pi.items()), 5))
    docid = pageinfo.split('/')[-1].replace('_1_restructpageinfo.json', '').strip('cr_')

    for info, lines, j, in zip(pi.items(), pl.items(), range(len(pi.items()))):
        toc = 0
        c = 0
        for line in lines[1]:
            if 'contents' in line.lower():
                c = 1
                if 'table of contents' in line.lower():
                    toc = 1

        docset[j] = np.array([docid, info[0], len(lines[1]), toc, c])
    pgdf = pd.DataFrame(data=docset, columns=['DocID', 'PageNum', 'NumChildren', 'ContainsTOCPhrase', 'ContainsContentsWord'])
    out = settings.production_path + docid + outfile
    pgdf.to_csv(out, index=False)
    return pgdf


if __name__ == '__main__':
    docid = '26525'
    create_toc_dataset(docid)
    toc_pages = toc_classification.get_toc_pages(docid)
    print(toc_pages)