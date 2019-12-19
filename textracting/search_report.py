import json
import pandas as pd
import numpy as np
import toc_classification
import settings
import os
import re
import heading_identification
import lstm_heading_identification

# transform document pages into dataset of pages for toc classification, classify pages, and isolate toc
# from toc page, transform content into dataset of headings for heading identification, identify headings, and return headings and subheadings
#for page in doc:

os.environ['KMP_AFFINITY'] = 'noverbose'

class Report():
    def __init__(self, docid):
        self.docid = docid
        self.toc_dataset_path = settings.production_path + docid + '_toc_dataset.csv'
        self.head_id_dataset_path = settings.production_path + docid + '_headid_dataset.csv'
        self.head_id_dataset_path_proc = settings.production_path +  docid + '_proc_headid_dataset.csv'
        self.toc_page = self.get_toc_page()
        self.headings, self.subheadings = self.get_headings()

    def create_toc_dataset(self):
        pageinfo = settings.get_restructpageinfo_file(self.docid)
        pagelines = settings.get_restructpagelines_file(self.docid)
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
        pgdf.to_csv(self.toc_dataset_path, index=False)
        return pgdf

    def get_toc_page(self):
        if not os.path.exists(self.toc_dataset_path):
            self.create_toc_dataset()
        toc_pages = toc_classification.get_toc_pages(self.docid)
        toc = int(toc_pages['PageNum'].values[0])
        return toc

    def get_headings(self):
        if not os.path.exists(self.head_id_dataset_path):
            self.create_identification_dataset()
        if not os.path.exists(self.head_id_dataset_path_proc):
            newdf = heading_identification.pre_process_id_dataset(pre='cyfra1', datafile=self.head_id_dataset_path, training=False)
            newdf.to_csv(self.head_id_dataset_path_proc)
        model = lstm_heading_identification.NeuralNetwork()
        df = pd.read_csv(settings.dataset_path + 'heading_id_dataset.csv')
        df = df.loc[df['DocID'] == float(self.docid)]
        x_labels = df['LineText']
        x = pd.read_csv(self.head_id_dataset_path_proc)['SectionText']
        _, res = model.predict(x)
        headings = []
        subheadings = []
        neither= []
        for line, pred in zip(x_labels, res):
            if pred == 0:
                neither.append(line)
            elif pred == 1:
                headings.append(line)
            elif pred == 2:
                subheadings.append(line)
        return headings, subheadings

    def create_identification_dataset(self):
        df = pd.DataFrame(columns=['DocID', 'LineNum', 'LineText'])
        pages = json.load(open(settings.get_restructpagelines_file(self.docid), 'r'))
        for lines in pages.items():
            if lines[0] == str(self.toc_page):
                docset = []
                for line, i in zip(lines[1], range(len(lines[1]))):
                    docset.append([self.docid, i, line])
                pgdf = pd.DataFrame(data=docset, columns=['DocID', 'LineNum', 'LineText'])
                df = df.append(pgdf, ignore_index=True)
        df.to_csv(self.head_id_dataset_path, index=False)
        return df


if __name__ == '__main__':
    r = Report('26525')
    print("Headings: \n", r.headings)
    print("Subheadings: \n", r.subheadings)



