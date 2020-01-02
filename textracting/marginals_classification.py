import numpy as np
import pandas as pd
import json
import glob
import settings
import re

def create_dataset():
    columns = ['DocID', 'PageNum', 'LineNum', 'NormedLineNum','Text', 'WordsWidth', 'Width', 'Height', 'Left', 'Top', 'ContainsNum',
               'ContainsTab', 'ContainsPage', 'Centrality', 'Marginal']
    pageinfos = glob.glob('training/restructpageinfo/*')
    df = pd.DataFrame(columns=columns)
    for pagesinfo in pageinfos:
        docset = []
        pi = json.load(open(pagesinfo))
        docid = pagesinfo.split('\\')[-1].replace('_1_restructpageinfo.json', '').strip('cr_')
        for info in pi.items():
            page = info[0]
            lines = len(info[1])
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
                #normed_line = line['LineNum'] / lines
                centrality = 0.5 - abs(bb['Left'] + (bb['Width']/2) - 0.5)  # the higher value the more central

                docset.append([docid, int(page), line['LineNum'], 0, line['Text'], line['WordsWidth'],
                               bb['Width'], bb['Height'], bb['Left'], bb['Top'], contains_num, contains_tab,
                               contains_page, centrality, 0])

            temp = pd.DataFrame(data=docset, columns=columns)
            temp['NormedLineNum'] = (temp['LineNum'] - min(temp['LineNum'])) / (max(temp['LineNum']) - min(temp['LineNum']))
            df = df.append(temp, ignore_index=True)

    unnormed = np.array(df['Centrality'])
    normalized = (unnormed - min(unnormed)) / (max(unnormed) - min(unnormed))
    df['Centrality'] = normalized

    return df


if __name__ == "__main__":
    df = create_dataset()
    df.to_excel(settings.dataset_path + 'marginals_dataset.xlsx', index=False)