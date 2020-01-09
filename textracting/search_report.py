import json
import pandas as pd
import numpy as np
import toc_classification
import settings
import os
import re
import heading_identification
import lstm_heading_identification
import textdistance
import marginals_classification
import page_extraction
import fig_classification
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
        self.fig_pages = fig_classification.get_fig_pages(self.docid)
        self.headings, self.subheadings = self.get_headings()
        #self.marginals_type = None # header or footer if pagenumbers
        #self.marginals_bb = None
        self.marginals = marginals_classification.get_marginals(self.docid)  # a df containing many columns, key: pagenum, text
        self.page_nums = page_extraction.get_page_nums(self.marginals)
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

    def line_matches_header(self, line, header):
        # complete partial match
        header = str(header)
        line = line['Text']
        if header in line or line in header or line == header:
            return True
        if 'WORK' in header or 'EXPLORATION' in header:  # these are the contentious sections that warrant this
            # prefix + suffix match is greater than half the length of the line
            if textdistance.prefix.similarity(line, header) + textdistance.postfix.similarity(line, header) > len(line) * 0.4:
                print('prefix similarity: ', line, header)
                return True

            # words match
            #line_lst = line.split()
            #header_lst = header.split()

            if textdistance.overlap.similarity(line, header) > 0.80:
                print('overlap similarity: ', line, header)
                return True
        return False

    def get_section_ptrs(self):
        # two versions: if have page numbers, if do not
        h = 0
        sections_ptrs = []
        doc = self.docinfo
        hpage = self.headings['SectionPage']
        hnum = self.headings['SectionPrefix']
        htext = self.headings['SectionText']
        self.headings['PDFPage'] = np.nan

        if len(self.page_nums) > 0 and len(self.headings) > 0:  # if have page numbers, should be able to find headings with them
            # match hpage[i] to page_nums.page[j]
            # take page_nums.PageNum as page pointer
            # find line pointer by iterating down the page as below
            not_found = []
            for j, jow in self.headings.iterrows():
                if jow.SectionPage != jow.SectionPage:
                    not_found.append(j)
                    continue
                for i, row in self.page_nums.iterrows():
                    if int(row.Page) == int(jow.SectionPage):
                        self.headings['PDFPage'][j] = row.PageNum
                        break
            # todo: need to also deal with headings that don't have pagenumber - eg. summary
            # check for headings. that don't have a filled PDFPage, and give them to the 'else' below
            #   can restrict the pages to search for them as the pages between their surrounding headings start pages

            if len(not_found) > 0:  # if a heading didn't have a page number found
                # look for it between the previous and next headings
                for i in not_found:
                    prev_i, next_i = None, None
                    prev, next = 0, self.page_nums.tail(1).PageNum
                    if i - 1 >= 0:  # if there's an index before it
                        prev_i = i -1
                    if i + 1 <= len(self.headings.SectionText): # if there's an index after it
                        next_i = i+1
                    if prev_i:
                        prev = int(self.headings.PDFPage[prev_i])
                    if next_i:
                        next = int(self.headings.PDFPage[next_i]+1)

                    page_range = range(prev, next)  # +1 bc inclusive of end

                    for page in page_range:
                        page += 1
                        if page != self.toc_page:
                            ptr = self.find_header_on_page(self.headings.iloc[i], doc[str(page)])  # doc[page+1] because it starts from 1
                            if isinstance(ptr, dict):  # if return string, didn't find it
                                self.headings.PDFPage[i] = page
                                break
                            else:
                                print(ptr, ': ', page)

            # iterate the pages that should contain a header
            for i, header in self.headings.iterrows():
                page = doc[str(int(header.PDFPage))]
                ptr = self.find_header_on_page(header, page)
                if isinstance(ptr, dict): # if return string, didn't find it - that's weird
                    ptr['HeadingPage'] = int(header.PDFPage)
                    sections_ptrs.append(ptr)
                else:
                    print(ptr)

        else: # don't have page numbers
            # iterate through the document as below
            for page in doc.items():  # for page in doc
                if int(page[0]) != self.toc_page:  # if that page is not the toc page
                                                    # if that page is not a fig page
                    if int(page[0] != self.fig_pages.PageNum):  # not sure if I want to exclude fig pages - I do if they ONLY contain figs
                        for line in page[1]:
                            if h >= len(htext):
                                break
                            hstr = re.sub('\t', '', htext[h]).strip()
                            if hnum[h] != hnum[h]:
                                hnum[h] = ''
                            if hpage[h] != hpage[h]:
                                hpage[h] = ''

                            match = self.line_matches_header(line, str(hnum[h]) + " " + hstr)
                            if match:
                                sections_ptrs.append({'HeadingText': str(hnum[h]) + " " + hstr + " " + str(hpage[h]),
                                                              'PageNum': int(page[0]), 'LineNum': int(line['LineNum'])})
                                h += 1

        return sections_ptrs
        # once have all header positions, can return the text in between them

    def find_header_on_page(self, header, page):  # finding header when you know the page it's on
        hstr = re.sub('\t', '', header.SectionText).strip()
        if header.SectionPrefix != header.SectionPrefix:  # checking for empty value
            header.SectionPrefix = ''
        if header.SectionPage != header.SectionPage:  # checking for empty value
            header.SectionPage = ''
        for line in page:
            match = self.line_matches_header(line, str(header.SectionPrefix) + " " + hstr)
            if match:
                #return {'HeadingText': str(header.SectionText) + " " + hstr + " " + str(header.SectionPage),
                #                      'PageNum': int(page[0]), 'LineNum': int(line['LineNum'])}
                #return {'HeadingText': str(header.SectionPrefix) + " " + hstr + " " + str(header.SectionPage),
                #    'LineNum': int(line['LineNum'])}
                return {'HeadingText': line['Text'], 'HeadingLine': int(line['LineNum'])}
            #else:
            #    return "Could not find header on the page"

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
        end = False
        for page in range(start_page, len(self.doclines.items()) +2): # +1 because index starts at 1, +1 to include last element
            try:
                for linenum in range(len(self.doclines[str(page)])):
                    line = self.doclines[str(page)][linenum]
                    line_info = self.docinfo[str(page)][linenum]
                    # if linenum == 0 and self.marginals_type == 'Header':
                    #     if self.check_if_line_is_marginal(line_info):
                    #         continue
                    # if linenum == len(self.doclines[str(page)]) and self.marginals_type == 'Footer':
                    #     if self.check_if_line_is_marginal(line_info):
                    #         continue

                    if page == end_page and linenum == end_line:  # if end of section
                        section = {'Heading': name, 'Content': content}
                        sections.append(section)

                        if section_num != len(self.section_ptrs) - 1: # if there is a next section
                            content = []
                            content.append(line)
                            section_num +=1
                            ptr = self.section_ptrs[section_num]
                            name = ptr['HeadingText']

                        if section_num != len(self.section_ptrs) - 1: # if next section is not last section
                            next_section = self.section_ptrs[section_num + 1]
                            end_page = next_section['PageNum']
                            end_line = next_section['LineNum']

                        else:   # if next section is last section
                            end_page = len(self.doclines.items())+1
                            end_line = len(self.doclines[str(end_page)])
                            end = True

                    elif end and page == end_page and linenum == end_line-1: # if current section is last section, current line is last line in whole doc
                        content.append(line)
                        section = {'Heading': name, 'Content': content}
                        sections.append(section)

                    elif page == start_page and linenum < start_line:  # if before start line on start page
                        continue

                    elif page == end_page and linenum+1 == end_line:  # stop the heading line being added to the end of the previous section
                        continue

                    else:  # add line to content
                        #line = self.doclines[str(page)][linenum]
                        content.append(line)
            except KeyError:
                print('Page ' + str(page) + ' is missing')
        return sections


if __name__ == '__main__':
    # transform document pages into dataset of pages for toc classification, classify pages, and isolate toc
    # from toc page, transform content into dataset of headings for heading identification, identify headings, and return headings and subheadings

    r = Report('26525')
    #print("Headings: \n", r.headings)
    #print("Subheadings: \n", r.subheadings)
    print("Sections at: \n", r.section_ptrs)
   # json.dump(r.section_content, open('sections.json', 'w'))
    #for section in r.section_content:
    #    print(section['Heading'])
    #    for line in section['Content']:
    #        print(line)