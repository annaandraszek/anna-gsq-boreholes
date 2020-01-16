import json
import pandas as pd
import numpy as np
import toc_classification
import settings
import os
import heading_identification
import lstm_heading_identification
import marginals_classification
import page_extraction
import fig_classification
import heading_id_intext
import heading_classification
os.environ['KMP_AFFINITY'] = 'noverbose'
from pdf2image import convert_from_path
from PIL import ImageDraw, Image
import re
from PyPDF2 import PdfFileWriter, PdfFileReader
import time


class Report():
    def __init__(self, docid):
        self.docid = docid
        self.toc_dataset_path = settings.production_path + docid + '_toc_dataset.csv'
        #self.head_id_dataset_path = settings.production_path + docid + '_headid_dataset.csv'
        #self.head_id_dataset_path_proc = settings.production_path +  docid + '_proc_headid_dataset.csv'
        self.docinfo = self.get_doc_info()
        self.doclines = self.get_doc_lines()
        self.line_dataset = self.create_line_dataset()
        self.toc_page = self.get_toc_page()
        self.fig_pages = fig_classification.get_fig_pages(self.docid, self.docinfo, self.doclines)
        self.line_dataset = self.create_line_dataset()
        self.section_ptrs = self.get_section_ptrs()  # section_ptrs = [{HeadingText: , PageNum: , LineNum: }]
        self.section_content = self.get_sections()
        #self.toc_heading_classes, self.text_heading_classes = self.classify_headings()

    def classify_headings(self):

        heading_classification()

    def get_doc_lines(self):
        pagelines = {}
        for page in self.docinfo.items():
            pagenum = page[0]
            info = page[1]
            lines = []
            for line in info:
                lines.append(line['Text'])
            pagelines[pagenum] = lines
        return pagelines

    def get_doc_info(self):
        pageinfo = settings.get_restructpageinfo_file(self.docid)
        pi = json.load(open(pageinfo, "r"))
        return pi

    def create_toc_dataset(self):
        docset = np.zeros((len(self.docinfo.items()), 5))
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
        #pgdf.to_csv(self.toc_dataset_path, index=False)
        return pgdf

    def get_toc_page(self):
        #if not os.path.exists(self.toc_dataset_path):
        data = self.create_toc_dataset()
        toc_pages = toc_classification.get_toc_pages(data)
        toc = int(toc_pages['PageNum'].values[0])
        return toc

    def get_headings(self):
        #if not os.path.exists(self.head_id_dataset_path):
        df = self.create_identification_dataset()
        #if not os.path.exists(self.head_id_dataset_path_proc):
        newdf = heading_identification.pre_process_id_dataset(pre='cyfra1', datafile=df, training=False)
        #newdf.to_csv(self.head_id_dataset_path_proc)
        model = lstm_heading_identification.NeuralNetwork()
        #refdf = pd.read_csv(settings.dataset_path + 'processed_heading_id_dataset.csv')
        #refdf = refdf.loc[refdf['DocID'] == float(self.docid)]
        #x_labels = refdf[['SectionPrefix', 'SectionText', 'SectionPage']]
        #x_labels.reset_index(drop=True, inplace=True)
        #x = pd.read_csv(self.head_id_dataset_path_proc)['SectionText']
        x = newdf['SectionText']
        _, res = model.predict(x)
        columns = ['LineNum', 'SectionPrefix', 'SectionText', 'SectionPage']  #'PageNum',
        headings = pd.DataFrame(columns=columns)
        subheadings = pd.DataFrame(columns=columns)
        #neither = []
        for i, pred in zip(range(len(res)), res):
            #if pred == 0:
            #print(x_labels.iloc[i])
            if pred > 0:
                heading = self.docinfo[str(self.toc_page)][i]
                section_prefix, section_text = heading_identification.split_prefix(heading['Text'])
                section_text, section_page = heading_identification.split_pagenum(section_text)
                hrow = [heading['LineNum'], section_prefix, section_text, section_page]  # heading['PageNum'],
                if pred == 1:
                    headings.loc[len(headings)] = hrow
                    #headings = headings.append([[section_prefix, section_text, section_page]], ignore_index=True)
                elif pred == 2:
                    subheadings.loc[len(subheadings)] = hrow
                    #subheadings = subheadings.append([[section_prefix, section_text, section_page]], ignore_index=True)
        return headings, subheadings

    def create_line_dataset(self):
        # create a dataset of all lines in the document and their universally used attributes like words2width, centrality
            # to be used by any classifier which uses lines
        columns = ['DocID', 'PageNum', 'LineNum', 'NormedLineNum', 'Text', 'Words2Width', 'WordsWidth', 'Width',
                   'Height', 'Left', 'Top', 'Centrality', 'WordCount']
        df = pd.DataFrame(columns=columns)
        for info in self.docinfo.items():
            docset = []
            page = info[0]
            for line in info[1]:
                bb = line['BoundingBox']
                centrality = 0.5 - abs(bb['Left'] + (bb['Width'] / 2) - 0.5)  # the higher value the more central
                words2width = line['WordsWidth'] / bb['Width']
                wordcount = len(line['Text'].split())
                docset.append([self.docid, int(page), line['LineNum'], 0, line['Text'], words2width, line['WordsWidth'],
                               bb['Width'], bb['Height'], bb['Left'], bb['Top'], centrality, wordcount])

            temp = pd.DataFrame(data=docset, columns=columns)
            temp['NormedLineNum'] = (temp['LineNum'] - min(temp['LineNum'])) / (
                        max(temp['LineNum']) - min(temp['LineNum']))
            df = df.append(temp, ignore_index=True)

        unnormed = np.array(df['Centrality'])
        normalized = (unnormed - min(unnormed)) / (max(unnormed) - min(unnormed))
        df['Centrality'] = normalized
        return df



    def create_identification_dataset(self): # dataset for identifying headings in TOC
        columns = ['DocID', 'LineNum', 'Text']
        df = self.line_dataset.loc[self.line_dataset['PageNum'] == self.toc_page]
        df = df[columns]
        df.reset_index(inplace=True, drop=True)
        return df

        #df = pd.DataFrame(columns=['DocID', 'LineNum', 'LineText'])
        #pages = json.load(open(settings.get_restructpageinfo_file(self.docid), 'r'))
        # for lines in pages.items():
        #     if lines[0] == str(self.toc_page):
        #         docset = []
        #         for line, i in zip(lines[1], range(len(lines[1]))):
        #             docset.append([self.docid, i, line['Text']])
        #         pgdf = pd.DataFrame(data=docset, columns=['DocID', 'LineNum', 'LineText'])
        #         df = df.append(pgdf, ignore_index=True)
        # df.to_csv(self.head_id_dataset_path, index=False)
        # return df

    def create_intext_id_dataset(self):
        df = self.line_dataset.copy(deep=True)
        #columns = ['DocID', 'PageNum', 'LineNum', 'NormedLineNum', 'Text', 'Words2Width', 'WordsWidth',
        #                 'Width', 'Height', 'Left', 'Top', 'ContainsNum', 'Centrality', 'WordCount', 'Heading']

        # df = pd.DataFrame(columns=columns)
        # for info in self.docinfo.items():
        #     docset = []
        #     page = info[0]
        #     for line in info[1]:
        #         bb = line['BoundingBox']
        #         centrality = 0.5 - abs(bb['Left'] + (bb['Width'] / 2) - 0.5)  # the higher value the more central
        #         words2width = line['WordsWidth'] / bb['Width']
        #         docset.append([self.docid, int(page), line['LineNum'], 0, line['Text'], words2width, line['WordsWidth'],
        #                        bb['Width'], bb['Height'], bb['Left'], bb['Top'], 0, centrality, 0, 0])
        #
        #     temp = pd.DataFrame(data=docset, columns=columns)
        #     temp['NormedLineNum'] = (temp['LineNum'] - min(temp['LineNum'])) / (
        #                 max(temp['LineNum']) - min(temp['LineNum']))
        #     df = df.append(temp, ignore_index=True)

        # unnormed = np.array(df['Centrality'])
        # normalized = (unnormed - min(unnormed)) / (max(unnormed) - min(unnormed))
        # df['Centrality'] = normalized
        # update contains num to just re.search('[0-9]+')
        df['ContainsNum'] = df.Text.apply(lambda x: heading_id_intext.contains_num(x))
        df['Heading'] = 0
        # # add column: line word count
        # df['WordCount'] = df.Text.apply(lambda x: len(x.split()))
        return df

    def create_marginals_dataset(self):
        df = self.line_dataset.copy(deep=True)
        df['ContainsNum'] = df.Text.apply(lambda x: marginals_classification.contains_num(x))
        df['ContainsTab'] = df.Text.apply(lambda x: marginals_classification.contains_tab(x))
        df['ContainsPage'] = df.Text.apply(lambda x: marginals_classification.contains_page(x))
        df['Marginal'] = 0
        df.drop(columns=['WordCount'], inplace=True)
        return df
        #columns = ['DocID', 'PageNum', 'LineNum', 'NormedLineNum', 'Text', 'Words2Width', 'WordsWidth', 'Width',
        #           'Height', 'Left', 'Top', 'ContainsNum',
        #           'ContainsTab', 'ContainsPage', 'Centrality', 'Marginal']
        # pageinfo = settings.get_restructpageinfo_file(docid)
        # pi = json.load(open(pageinfo))
        # df = pd.DataFrame(columns=columns)
        # for info in pi.items():
        #     docset = []
        #     page = info[0]
        #     for line in info[1]:
        #         contains_num = 0
        #         contains_tab = 0
        #         contains_page = 0
        #         bb = line['BoundingBox']
        #         if re.search(r'(\s|^)[0-9]+(\s|$)', line['Text']):
        #             contains_num = 1
        #         if re.search(r'\t', line['Text']):
        #             contains_tab = 1
        #         if 'page' in line['Text'].lower():
        #             contains_page = 1
        #         centrality = 0.5 - abs(bb['Left'] + (bb['Width'] / 2) - 0.5)  # the higher value the more central
        #         words2width = line['WordsWidth'] / bb['Width']
        #         docset.append([docid, int(page), line['LineNum'], 0, line['Text'], words2width, line['WordsWidth'],
        #                        bb['Width'], bb['Height'], bb['Left'], bb['Top'], contains_num, contains_tab,
        #                        contains_page, centrality, 0])
        #
        #     temp = pd.DataFrame(data=docset, columns=columns)
        #     temp['NormedLineNum'] = (temp['LineNum'] - min(temp['LineNum'])) / (
        #                 max(temp['LineNum']) - min(temp['LineNum']))
        #     df = df.append(temp, ignore_index=True)
        #
        # unnormed = np.array(df['Centrality'])
        # normalized = (unnormed - min(unnormed)) / (max(unnormed) - min(unnormed))
        # df['Centrality'] = normalized
        #return df

    def get_section_ptrs(self):
        self.headings, self.subheadings = self.get_headings()
        self.marginals = marginals_classification.get_marginals(self.create_marginals_dataset())  # a df containing many columns, key: pagenum, text
        self.marginals_set = set([(p, l) for p, l in zip(self.marginals.PageNum, self.marginals.LineNum)])
        self.page_nums = page_extraction.get_page_nums(self.marginals)
        self.headings_intext = heading_id_intext.get_headings_intext(self.create_intext_id_dataset())
        section_ptrs = self.headings_intext.loc[self.headings_intext['Heading'] == 1]
        self.subsection_ptrs = self.headings_intext.loc[self.headings_intext['Heading'] == 2]
        self.subsection_ptrs.reset_index(inplace=True, drop=True)
        section_ptrs.reset_index(inplace=True, drop=True)
        return section_ptrs

    def get_sections(self):
        # from section ptrs, section = section ptr, reading until the start of the next section
        #for ptr in self.section_ptrs:
        if self.section_ptrs.shape[0] == 0:
            return []

        section_num = 0
        ptr = self.section_ptrs.iloc[section_num]
        name = ptr['Text']
        start_page = ptr['PageNum']
        start_line = ptr['LineNum']
        if self.section_ptrs.shape[0] > 1:
            next_section = self.section_ptrs.iloc[section_num+1]
            end_page = next_section['PageNum']
            end_line = next_section['LineNum']
        else:
            end_page = len(self.doclines)
            end_line = len(self.doclines[str(end_page)])
        content = []
        sections = []
        end = False
        for page in range(start_page, len(self.doclines.items()) +2): # +1 because index starts at 1, +1 to include last element
            try:
                for linenum in range(len(self.doclines[str(page)])):
                    if (page, linenum) not in self.marginals_set:  # if line is not a marginal
                        line = self.doclines[str(page)][linenum]
                        if page == end_page and linenum == end_line:  # if end of section
                            section = {'Heading': name, 'Content': content}
                            sections.append(section)

                            if section_num != len(self.section_ptrs) - 1: # if there is a next section
                                content = []
                                content.append(line)
                                section_num +=1
                                ptr = self.section_ptrs.iloc[section_num]
                                name = ptr['Text']

                            if section_num != len(self.section_ptrs) - 1: # if next section is not last section
                                next_section = self.section_ptrs.iloc[section_num + 1]
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


def print_sections(report):
    print("Sections at: \n", report.section_ptrs)
    # json.dump(r.section_content, open('sections.json', 'w'))
    for section in report.section_content:
        print(section['Heading'])
        for line in section['Content']:
            print(line)
        print('\n')


def draw_report(report):
    report_path = settings.get_report_name(report.docid, local_path=True, file_extension='pdf')
    images = convert_from_path(report_path)

    doc = report.docinfo
    drawn_images = []

    for page in doc.items():
        i = int(page[0])-1
        image = images[i]  # this has to be of type RGB
        width, height = image.size
        draw = ImageDraw.Draw(image, 'RGBA')

        if int(page[0]) in report.marginals['PageNum'].values:     # draw bb around marginals
            lnnum = report.marginals.loc[report.marginals['PageNum'] == int(page[0])]['LineNum']
            linenum = lnnum.values[0] - 1
            line = page[1][linenum]
            box = line['BoundingBox']
            left = width * box['Left']
            top = height * box['Top']
            draw.rectangle([left, top, left + (width * box['Width']), top + (height * box['Height'])], outline='orange')

            # draw bb around page number (by comparing marginal content to result of page number extraction)
            if int(page[0]) in report.page_nums['PageNum'].values:  # draw bb around marginals
                pg_marginal = report.page_nums.loc[report.page_nums['PageNum'] == int(page[0])]
                #pglnnum = pg_marginal['LineNum']
                #pglinenum = pglnnum.values[0] - 1
                text = pg_marginal.Text.values[0]
                split_text = text.split('\t')
                reg = r'(^|\s)' + str(pg_marginal['Page'].values[0]) + r'($|\s)'   # implement returning pagenum position instead? would make this MUCH easier
                pgnum_i = None
                for t, i in zip(split_text, range(len(split_text))):
                    if re.search(reg, t):
                        pgnum_i = i
                        break
                box = line['OriginalBBs'][pgnum_i]
                left = width * box['Left']
                top = height * box['Top']
                draw.rectangle([left, top, left + (width * box['Width']), top + (height * box['Height'])],
                               outline='red')

            #original_marginal_bb = docinfo[pagestr][lineindex]['OriginalBBs'][index in marginal]

        if page[0] == str(report.toc_page): # change colour of toc page
            img_copy = image.copy()
            background = ImageDraw.Draw(img_copy, 'RGBA')
            background.rectangle([0, 0, image.size[0], image.size[1]], fill='green')
            image = Image.blend(img_copy, image, alpha=0.3)

        if float(page[0]) in report.fig_pages['PageNum'].values: # change colour of fig pages
            img_copy = image.copy()
            background = ImageDraw.Draw(img_copy, 'RGBA')
            background.rectangle([0, 0, image.size[0], image.size[1]], fill='purple')
            image = Image.blend(image, img_copy, alpha=0.3)

        #else:
        # draw bb around section headers
        if int(page[0]) in report.section_ptrs['PageNum'].values:
            lnnums = report.section_ptrs.loc[report.section_ptrs['PageNum'] == int(page[0])]['LineNum']
            for line in lnnums.values:
                linenum = line - 1
                line = page[1][linenum]
                box = line['BoundingBox']
                left = width * box['Left']
                top = height * box['Top']
                draw.rectangle([left, top, left + (width * box['Width']), top + (height * box['Height'])], outline='blue')

        if int(page[0]) in report.subsection_ptrs['PageNum'].values:
            lnnums = report.subsection_ptrs.loc[report.subsection_ptrs['PageNum'] == int(page[0])]['LineNum']
            for line in lnnums.values:
                linenum = line - 1
                line = page[1][linenum]
                box = line['BoundingBox']
                left = width * box['Left']
                top = height * box['Top']
                draw.rectangle([left, top, left + (width * box['Width']), top + (height * box['Height'])],
                               outline='green')

        drawn_images.append(image)
    save_path = settings.result_path + report.docid + '_boxed.pdf'
    drawn_images[0].save(save_path, save_all=True, append_images=drawn_images[1:])


# add bookmarks to sections and sub-bookmarks to subsections
# if drawing report, after it has been drawn on. if not, need to download the report if it has not been converted from tif
def bookmark_report(report):
    report_file = 'results/' + report.docid + '_boxed.pdf'
    output = PdfFileWriter()
    input = PdfFileReader(open(report_file, 'rb'))
    ptrs = report.headings_intext
    for page in input.pages:
        output.addPage(page)
    for i, row in ptrs.iterrows():
        #page, line = row['PageNum'], row['LineNum']
        #lnbb = report.docinfo[page][line-1]['BoundingBox']
        if row['Heading'] == 1:
            section = output.addBookmark(row['Text'], row['PageNum']-1, fit='/FitB')
        elif row['Heading'] == 2:
            output.addBookmark(row['Text'], row['PageNum']-1, parent=section, fit='/FitB')  # catch if section doesn't exist?

    outfile = 'results/' + report.docid + '_bookmarked.pdf'
    output.write(open(outfile, 'wb'))

if __name__ == '__main__':
    # transform document pages into dataset of pages for toc classification, classify pages, and isolate toc
    # from toc page, transform content into dataset of headings for heading identification, identify headings, and return headings and subheadings

    start = time.time()
    r = Report('28184')
    draw_report(r)
    bookmark_report(r)
    end = time.time()
    print('time:', end - start)
    #print('TOC Headings: \n')
    #for string in r.doclines[str(r.toc_page)]:
    #    print(string)
    #print("In Text Headings: \n", r.headings)
    #print("Subheadings: \n", r.subheadings)
    #print_sections(r)
