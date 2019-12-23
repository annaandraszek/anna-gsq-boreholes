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
        self.marginals_type = None # header or footer if pagenumbers
        self.marginals_bb = None
        self.section_ptrs = self.get_section_ptrs()  # section_ptrs = [{HeadingText: , PageNum: , LineNum: }]
        self.section_content = self.get_sections()

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
        pagenum = None
        if self.marginals_type == 'Header':
            line = page[1][0]
            if self.check_if_line_is_marginal(line):
                if re.search(r'\t\d+', line['Text']):
                    pagenum = re.search(r'\t\d+', line['Text']).group(0)
                    print(line['Text'], line['BoundingBox'])
        else:
            line = page[1][-1]
            if self.check_if_line_is_marginal(line):
                if re.search(r'\t\d+', line['Text']):
                    pagenum = re.search(r'\t\d+', line['Text']).group(0)
                    print(line['Text'], line['BoundingBox'])
        if pagenum:
            return pagenum.strip()

    def check_if_line_is_marginal(self, line):
        if self.marginals_type:
            line_bb = line['BoundingBox']
            if line_bb['Width'] - 0.05 <= self.marginals_bb['Width'] <= line_bb['Width'] + 0.05:
                if line_bb['Height'] - 0.005 <= self.marginals_bb['Height'] <= line_bb['Height'] + 0.005:
                    if line_bb['Left'] - 0.05 <= self.marginals_bb['Left'] <= line_bb['Left'] + 0.05:
                        if line_bb['Top'] - 0.05 <= self.marginals_bb['Top'] <= line_bb['Top'] + 0.05:
                            return True
        else:
            print('No marginals type')
        return False

        # compare line bb to marginal bb
        # width +- 0.05, height +- 0.005, left +- 0.05, top += 0.05
        # update maginals bb with rolling average ?

    def find_marginals(self):
        page = self.docinfo[str(self.toc_page)]  # refpage to find marginal
        first_line = page[0]
        last_line = page[-1]
        if re.search(r'\t\d+', first_line['Text']):
            pagenum = re.search(r'\t\d+', first_line['Text']).group(0)
            self.marginals_type = 'Header'
            self.marginals_bb = first_line['BoundingBox']
        elif re.search(r'\t\d+', last_line['Text']):
            pagenum = re.search(r'\t\d+', last_line['Text']).group(0)
            self.marginals_type = 'Footer'
            self.marginals_bb = last_line['BoundingBox']


    def get_section_ptrs(self):
        h = 0
        sections_ptrs = []
        doc = self.docinfo
        hpage = self.headings['SectionPage']
        hnum = self.headings['SectionPrefix']
        htext = self.headings['SectionText']

        self.find_marginals()

        for page in doc.items():
            if int(page[0]) != self.toc_page:
                first_line = page[1][0]
                last_line = page[1][-1]
                pagenum = False

                if self.marginals_type:
                    if re.search(r'\t\d+', first_line['Text']) and self.check_if_line_is_marginal(first_line):
                        pagenum = re.search(r'\t\d+', first_line['Text']).group(0)
                    elif re.search(r'\t\d+', last_line['Text']) and self.check_if_line_is_marginal(last_line):
                        pagenum = re.search(r'\t\d+', last_line['Text']).group(0)

                for line in page[1]:
                    if h >= len(htext):
                        break
                    hstr = re.sub('\t', '', htext[h]).strip()
                    if hnum[h] != hnum[h]:
                        hnum[h] = ''
                    if hpage[h] != hpage[h]:
                        hpage[h] = ''

                    if hstr in line['Text']:
                        if str(hnum[h]) in line['Text']:
                            if pagenum:
                                pg = self.get_pagenum(page)
                                if int(pg) == hpage[h] or hpage[h] == '':
                                    sections_ptrs.append({'HeadingText': str(hnum[h]) + " " + hstr + " " + str(hpage[h]),
                                                              'PageNum': int(page[0]), 'LineNum': int(line['LineNum'])})
                                else:
                                    print("Pagenum in TOC doesn't match pagenum on page for heading: ", str(hnum[h]), hstr, str(hpage[h]))
                                    print("actual page: ", pagenum)
                        else:
                            sections_ptrs.append({'HeadingText': str(hnum[h]) + " " + hstr + " " + str(hpage[h]),
                                                          'PageNum': int(page[0]), 'LineNum': int(line['LineNum'])})
                        h += 1
        return sections_ptrs
        # once have all header positions, can return the text in between them

    def get_sections(self):
        # from section ptrs, section = section ptr, reading until the start of the next section
        #for ptr in self.section_ptrs:
        section_num = 0
        ptr = self.section_ptrs[section_num]
        name = ptr['HeadingText']
        start_page = ptr['PageNum']
        start_line = ptr['LineNum']
        next_section = self.section_ptrs[section_num+1]
        end_page = next_section['PageNum']
        end_line = next_section['LineNum']

        content = []
        sections = []
        for page in range(start_page, len(self.doclines.items()) +2): # +1 because index starts at 1, +1 to include last element
            try:
                for linenum in range(len(self.doclines[str(page)])):

                    if linenum == 0 and self.marginals_type == 'Header':  # don't include header/footer in the section content
                        continue  # AND store header/footer bounding box (assuming will be the same on each page, or
                        # very close) and only delete if line within it - stops removing figure/table pages without header/footer
                    if linenum == len(self.doclines[str(page)]) and self.marginals_type == 'Footer':
                        continue

                    if page == end_page and linenum == end_line:  # if end of section
                        section = {'Heading': name, 'Content': content}
                        sections.append(section)

                        if section_num != len(self.section_ptrs) - 1: # if there is a next section
                            section_num +=1
                            content = []
                            line = self.doclines[str(page)][linenum]
                            content.append(line)

                            ptr = self.section_ptrs[section_num]
                            name = ptr['HeadingText']

                        if section_num != len(self.section_ptrs) - 1: # if next section is not last section
                            next_section = self.section_ptrs[section_num + 1]
                            end_page = next_section['PageNum']
                            end_line = next_section['LineNum']

                        else:   # if next section is last section
                            end_page = len(self.doclines.items())+1
                            end_line = len(self.doclines[str(end_page)])-1


                    elif page == start_page and linenum < start_line:  # if before start line on start page
                        continue

                    elif page == end_page and linenum+1 == end_line:  # stop the heading line being added to the end of the previous section
                        continue

                    else:  # add line to content
                        line = self.doclines[str(page)][linenum]
                        content.append(line)
            except KeyError:
                print('Page ' + str(page) + ' is missing')
        return sections


if __name__ == '__main__':
    # transform document pages into dataset of pages for toc classification, classify pages, and isolate toc
    # from toc page, transform content into dataset of headings for heading identification, identify headings, and return headings and subheadings

    r = Report('27972')
    #print("Headings: \n", r.headings)
    #print("Subheadings: \n", r.subheadings)
    print("Sections at: \n", r.section_ptrs)
    json.dump(r.section_content, open('sections.json', 'w'))
    #for section in r.section_content:
    #    print(section['Heading'])
    #    for line in section['Content']:
    #        print(line)