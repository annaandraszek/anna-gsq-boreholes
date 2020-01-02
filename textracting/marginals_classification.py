import numpy as np
import pandas as pd
import json
import glob
import settings
import re

def create_dataset():
    columns = ['DocID', 'PageNum', 'LineNum', 'Text', 'WordsWidth', 'Width', 'Height', 'Left', 'Top', 'ContainsNum',
               'ContainsTab', 'ContainsPage', 'Marginal']
    pageinfos = glob.glob('training/restructpageinfo/*')
    docset = []
    for pagesinfo in pageinfos:
        pi = json.load(open(pagesinfo))
        docid = pagesinfo.split('\\')[-1].replace('_1_restructpageinfo.json', '').strip('cr_')

        for info in pi.items():
            page = info[0]
            for line in info[1]:
                contains_num = 0
                contains_tab = 0
                contains_page = 0
                bb = line['BoundingBox']
                if re.search(r'[0-9]+', line['Text']):
                    contains_num = 1
                if re.search(r'\t', line['Text']):
                    contains_tab = 1
                if 'page' in line['Text'].lower():
                    contains_page = 1
                docset.append([docid, int(page), line['LineNum'], line['Text'], line['WordsWidth'],
                               bb['Width'], bb['Height'], bb['Left'], bb['Top'], contains_num, contains_tab,
                               contains_page, 0])

    df = pd.DataFrame(data=docset, columns=columns)
    return df


if __name__ == "__main__":
    df = create_dataset()
    df.to_excel(settings.dataset_path + 'marginals_dataset.xlsx', index=False)