import json
import pandas as pd
import numpy as np
import toc_classification
import settings
import os
import re
import heading_identification
import lstm_heading_identification


os.environ['KMP_AFFINITY'] = 'noverbose'

class Report():
    def __init__(self, docid):
        self.docid = docid
        self.toc_dataset_path = settings.production_path + docid + '_toc_dataset.csv'
        self.head_id_dataset_path = settings.production_path + docid + '_headid_dataset.csv'
        self.head_id_dataset_path_proc = settings.production_path +  docid + '_proc_headid_dataset.csv'
        self.docinfo = self.get_doc_info()
        self.doclines = self.get_doc_lines()
        self.toc_page = self.get_toc_page()
        self.headings, self.subheadings = self.get_headings()
        self.pagenum_pos = None # header or footer if pagenumbers
        self.section_ptrs = self.get_sections()  # section_ptrs = [{HeadingText: , PageNum: , LineNum: }]

    def get_doc_lines(self):
        pagelines = settings.get_restructpagelines_file(self.docid)
        pl = json.load(open(pagelines, "r"))
        return pl

    def get_doc_info(self):
        pageinfo = settings.get_restructpageinfo_file(self.docid)
        pi = json.load(open(pageinfo, "r"))
        return pi

    def create_toc_dataset(self):
        docset = np.zeros((len(self.docinfo.items()), 5))
        #docid = pageinfo.split('/')[-1].replace('_1_restructpageinfo.json', '').strip('cr_')

        for info, lines, j, in zip(self.docinfo.items(), self.doclines.items(), range(len(self.docinfo.items()))):
            toc = 0
            c = 0
            for line in lines[1]:
                if 'contents' in line.lower():
                    c = 1
                    if 'table of contents' in line.lower():
                        toc = 1

            docset[j] = np.array([self.docid, info[0], len(lines[1]), toc, c])
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
        refdf = pd.read_csv(settings.dataset_path + 'processed_heading_id_dataset.csv')
        refdf = refdf.loc[refdf['DocID'] == float(self.docid)]
        x_labels = refdf[['SectionPrefix', 'SectionText', 'SectionPage']]
        x_labels.reset_index(drop=True, inplace=True)
        x = pd.read_csv(self.head_id_dataset_path_proc)['SectionText']
        _, res = model.predict(x)
        headings = pd.DataFrame(columns=['SectionPrefix', 'SectionText', 'SectionPage'])
        subheadings = pd.DataFrame(columns=['SectionPrefix', 'SectionText', 'SectionPage'])
        #neither = []
        for i, pred in zip(range(len(res)), res):
            #if pred == 0:
            #print(x_labels.iloc[i])
            if pred == 1:
                headings = headings.append(x_labels.iloc[i], ignore_index=True)
            elif pred == 2:
                subheadings = subheadings.append(x_labels.iloc[i], ignore_index=True)
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

    def get_pagenum(self, page):
        if self.pagenum_pos == 'Header':
            if re.search(r'\t\d+', page[1][0]['Text']):
                pagenum = re.search(r'\t\d+', page[1][0]['Text']).group(0)
        else:
            if re.search(r'\t\d+', page[1][-1]['Text']):
                pagenum = re.search(r'\t\d+', page[1][-1]['Text']).group(0)
        return pagenum.strip()

    def get_sections(self):
        h = 0
        sections_ptrs = []
        doc = self.docinfo
        hpage = self.headings['SectionPage']
        hnum = self.headings['SectionPrefix']
        htext = self.headings['SectionText']

        for page in doc.items():
            if int(page[0]) != self.toc_page:
                first_line = page[1][0]
                last_line = page[1][-1]
                pagenum = False

                if re.search(r'\t\d+', first_line['Text']):
                    pagenum = re.search(r'\t\d+', first_line['Text']).group(0)
                    self.pagenum_pos = 'Header'
                elif re.search(r'\t\d+', last_line['Text']):
                    pagenum = re.search(r'\t\d+', last_line['Text']).group(0)
                    self.pagenum_pos = 'Footer'

                for line in page[1]:
                    if h >= len(htext):
                        break
                    hstr = re.sub('\t', '', htext[h]).strip()
                    if hstr in line['Text']:
                        try:
                            if str(hnum[h]) in line['Text']:
                                if pagenum:
                                    pg = self.get_pagenum(page)
                                    if int(pg) == int(hpage[h]):
                                        sections_ptrs.append({'HeadingText': str(hnum[h]) + " " + hstr + " " + str(hpage[h]),
                                                              'PageNum': page[0], 'LineNum': line['LineNum']})
                                    else:
                                        print("Pagenum in TOC doesn't match pagenum on page for heading: ", str(hnum[h]), hstr, str(hpage[h]))
                                        print("actual page: ", pagenum)
                        except TypeError:
                            print('Nan page')
                        else:
                            sections_ptrs.append({'HeadingText': str(hnum[h]) + " " + hstr + " " + str(hpage[h]),
                                                          'PageNum': page[0], 'LineNum': line['LineNum']})
                        h += 1
        return sections_ptrs
        # once have all header positions, can return the text in between them




if __name__ == '__main__':
    # transform document pages into dataset of pages for toc classification, classify pages, and isolate toc
    # from toc page, transform content into dataset of headings for heading identification, identify headings, and return headings and subheadings

    r = Report('26525')
    #print("Headings: \n", r.headings)
    #print("Subheadings: \n", r.subheadings)
    print("Sections at: \n", r.section_ptrs)