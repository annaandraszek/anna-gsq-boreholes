from toc_classification import get_toc_pages
import glob
import pandas as pd
import json
import numpy as np
import re
import sklearn

def create_class_dataset():
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
                        headingType = 'None'
                        if re.match(r'^([0-9]+\.[0-9]+\s+\w+)', line):
                            heading = 'Sub'
                        elif re.match(r'^[0-9]+\.*\s+\w+', line):
                            heading = 'Head'

                        if 'summary' in line.lower() and i < 10:  # heuristic to help tagging
                            headingType = 'Summ'

                        if heading == 'Head' or heading == 'Sub':
                            if 'introduction' in line.lower() and i < 15:  # heuristic to help tagging
                                headingType = 'Intro'

                            elif 'work' in line.lower() or 'exploration' in line.lower() or 'drill' in line.lower():
                                headingType = 'Work'

                        docset.append([docid, i, line, heading, headingType])
                    pgdf = pd.DataFrame(data=docset, columns=['DocID', 'LineNum', 'LineText', 'Heading', 'HeadingType'])
                    df = df.append(pgdf, ignore_index=True)
        except IndexError:
            print("IndexError ", tocpg, docid)
    df.to_csv("heading_datasetv2.csv", index=False)
    return df


def strip_numbers(string):
    return re.sub(r'[0-9]', '', string)


def strip_punctuation(string):
    return str.strip('.')


def strip_all_but_words_spaces(string):
    return re.sub(r"[^\w\s]", "", string)


def split_prefix(string):
    s = re.split(r'(^[0-9]+\.*[0-9]*\.*[0-9]*)', string, 1)
    if len(s) == 1:
        s = ['', s[0]]
    elif len(s) == 3:
        s = [s[-2], s[-1]]
    return s


def split_pagenum(string):
    s = re.split(r'(\t[0-9]+$)', string, 1) # if $ doesn't work try \Z
    if len(s) == 1:
        s = [s[0], '']
    elif len(s) == 3:
        s = [s[0], s[1]]
    return s

def num2cyfra(string):
    s = ''
    prev_c = ''
    for c in string:
        if re.match(r'[0-9]', c):
            if prev_c != 'num':
                s += 'cyfra '
                prev_c = 'num'
        elif c == '.':
            s += 'punkt '
            prev_c = '.'
    return s

def pre_process_id_dataset():
    df = pd.read_csv("heading_id_dataset.csv")
    # break up the LineText column into SectionPrefix, SectionText, and SectionPage
    newdf = pd.DataFrame(columns=['DocID', 'LineNum', 'SectionPrefix', 'SectionText', 'SectionPage', 'Heading'])
    newdf.DocID = df.DocID
    newdf.LineNum = df.LineNum
    newdf.Heading = df.Heading

    newdf.SectionPrefix, newdf.SectionText = zip(*df.LineText.map(split_prefix))
    newdf.SectionText, newdf.SectionPage = zip(*newdf.SectionText.map(split_pagenum))

    newdf.SectionPrefix = newdf.SectionPrefix.apply(lambda x: num2cyfra(x))
    newdf.SectionPage = newdf.SectionPage.apply(lambda x: num2cyfra(x))

    newdf.replace('', np.nan, inplace=True)
    newdf.dropna(inplace=True, subset=['SectionText'])
    newdf.replace(np.nan, '', inplace=True)  # nan values cause issues when adding columns

    newdf.SectionText = newdf.SectionPrefix + newdf.SectionText + newdf.SectionPage
    newdf.drop(axis=1, columns=['SectionPrefix', 'SectionPage'], inplace=True)
    return newdf


def create_identification_dataset():
    df = pd.DataFrame(columns=['DocID', 'LineNum', 'LineText', 'Heading'])
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
                        heading = 0
                        if re.match(r'^([0-9]+\.[0-9]+\s+\w+)', line):
                            heading = 2
                        elif re.match(r'^[0-9]+\.*\s+\w+', line):
                            heading = 1

                        docset.append([docid, i, line, heading])
                    pgdf = pd.DataFrame(data=docset, columns=['DocID', 'LineNum', 'LineText', 'Heading'])
                    df = df.append(pgdf, ignore_index=True)
        except IndexError:
            print("IndexError ", tocpg, docid)
    df.to_csv("heading_id_dataset.csv", index=False)
    return df





if __name__ == "__main__":
    #df = create_identification_dataset()
    df = pre_process_id_dataset()
    df.to_csv('processed_heading_id_datasetv2.csv', index=False)
    #print(df)
