from toc_classification import get_toc_pages
import glob
import pandas as pd
import json
import numpy as np
import re

def create_dataset():
    df = pd.DataFrame(columns=['DocID', 'LineNum', 'LineText', 'Heading', 'HeadingType'])
    lines_docs = sorted(glob.glob('training/restructpagelines/*'))
    toc_pages = get_toc_pages()
    for lines_doc in lines_docs:
        pages = json.load(open(lines_doc))
        docid = int(lines_doc.split('\\')[-1].replace('_1_restructpagelines.json', '').strip('cr_'))
        tocpg = toc_pages.loc[toc_pages['DocID'] == docid]
        try:
            page = tocpg.PageNum.values[0]
            for lines in pages.items():
                if lines[0] == str(page):
                    docset = []
                    for line, i in zip(lines[1], range(len(lines[1]))):
                        heading = 'None'
                        if 'introduction' in line.lower() and i < 15:  # heuristic to help tagging
                            heading = 'Intro'
                        elif 'summary' in line.lower() and i < 10:  # heuristic to help tagging
                            heading = 'Summ'
                        elif 'work' in line.lower():
                            heading = 'Work'
                        docset.append([docid, i, line, heading])
                    pgdf = pd.DataFrame(data=docset, columns=['DocID', 'LineNum', 'LineText', 'HeadingType'])
                    df = df.append(pgdf, ignore_index=True)
        except IndexError:
            print("IndexError ", tocpg, docid)
    df.to_csv("heading_dataset.csv", index=False)
    return df


def strip_numbers(str):
    return re.sub(r'[0-9]', '', str)


def strip_punctuation(str):
    return str.strip('.')


def strip_all_but_words_spaces(str):
    return re.sub(r"[^\w\s]", "", str)


def pre_process_dataset(df=pd.read_csv("heading_dataset.csv")):
    df.LineText = df.LineText.apply(lambda x: strip_all_but_words_spaces(x))
    df.LineText = df.LineText.apply(lambda x: strip_numbers(x))
    #df.LineText = df.LineText.apply(lambda x: strip_punctuation(x))
    df.replace('', np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


#def annotate_more(df):
    #if df['LineText'].str.contains('exploration', case=False):
    # drill
    #



if __name__ == "__main__":
    df = create_dataset()
    df = pre_process_dataset()
    df.to_csv('processed_heading_dataset.csv', index=False)
    print(df)
