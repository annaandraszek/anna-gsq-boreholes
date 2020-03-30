## @package textracting
#@file
# Transforming results of textract into other structures/files

import numpy as np
import boto3
import json
import re
from PIL import ImageDraw,Image, ImageFont
from pdf2image import convert_from_path
import settings
from textracting import textsettings
import os
import pandas as pd
textract = boto3.client('textract')
comprehend = boto3.client('comprehend')


def print_doc_lines(doc):
    # Print text
    print("\nText\n========")
    text = ""
    for item in doc["Blocks"]:
        if item["BlockType"] == "LINE":
            print('\033[94m' + item["Text"] + '\033[0m')
            text = text + " " + item["Text"]
    return text


# Saves pdf version of report with boxed boundaries of text
# Edit this function, or write new version, which will show the bbs of marginals, sections, page numbers, toc, fig page
def display_doc(docid): # doc has to be pageinfo type - made for restructpageinfo
    report_path = settings.get_report_name(docid, local_path=True, file_extension=True)
    images = convert_from_path(report_path)

    docfile = open(settings.get_restructpageinfo_file(docid), "r")
    doc = json.load(docfile)
    drawn_images = []

    # Create image showing bounding box/polygon the detected lines/text
    for page in doc.items():
        i = int(page[0])-1
        image = images[i]
        width, height = image.size
        #draw = ImageDraw.Draw(image)
        draw = ImageDraw.Draw(image)
        for line in page[1]:
            # Uncomment to draw bounding box
            box = line['BoundingBox']
            left = width * box['Left']
            top = height * box['Top']
            draw.rectangle([left, top, left + (width * box['Width']), top + (height * box['Height'])], outline='green')

        #image.save(docid + '_' + page[0] + ".jpeg", "JPEG")
        drawn_images.append(image)

    save_path = settings.result_path + docid + '_boxed.pdf'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    drawn_images[0].save(save_path, save_all=True, append_images=drawn_images[1:])


def save_tables(doc, file_id):
    table_csv = get_table_csv(doc)
    with open(settings.get_tables_file(file_id), "w") as fout:
        fout.write(table_csv)
    #table_csv.to_csv(settings.get_tables_file(file_id))
    #print('CSV OUTPUT FILE: ', settings.get_tables_file(file_id))


def save_kv_pairs(result, file_id):
    kvs = get_kv_pairs(result)
    o = open(settings.get_kvs_file(file_id), "w")
    for key, value in kvs.items():
        o.write(str(key + ',' + value + '\n'))


def clean_and_restruct(docid, save=True):
    json_file = settings.get_full_json_file(docid)
    with open(json_file, 'r') as file:
        json_doc = json.load(file)
    json_res = json2res(json_doc)
    pagelineinfo = get_pagelineinfo_map(json_res)  # takes json response
    clean_page = get_clean_page(pagelineinfo, docid)
    restructpageinfo = get_restructpagelines(clean_page)

    if save:
        fp = settings.get_restructpageinfo_file(docid)
        p = fp.rsplit('/', 1)[0]
        if not os.path.exists(p):
            os.makedirs(p)
        o = open(settings.get_restructpageinfo_file(docid), "w")
        json.dump(restructpageinfo, o)
    else:
        return restructpageinfo

# helps when want to use textshowing functions with existing json files instead of the textract response
def json2res(jsondoc):
    all_blocks = []
    for pages in jsondoc:#['response']:
        all_blocks.extend(pages['Blocks'])
    res = {'Blocks': all_blocks}
    return res


# Removes OCR noise from left page margins. Has been tuned for current set of reports, also catches some map/table
# lines but these are minimal and extreme and would not be desired in the text anyway. The exception to this is
# removing "Hole" from an appendix table in report 36198.
def get_clean_page(doc, file_id):
    cleaned = {}
    for page in doc.items():
        pagenum = page[0]
        info = page[1]
        for line in info:
            lmargin = line['BoundingBox']['Left']
            if lmargin < 0.04:  #and line['Confidence'] < 90
                if line['BoundingBox']['Width'] < 0.02:
                    #print(line['Text'])
                    continue
            if pagenum in cleaned:
                cleaned[pagenum].append(line)
            else:
                cleaned[pagenum] = [line]

    return cleaned


def get_rows_columns_map(table_result, blocks_map):
    rows = {}
    for relationship in table_result['Relationships']:
        if relationship['Type'] == 'CHILD':
            for child_id in relationship['Ids']:
                cell = blocks_map[child_id]
                if cell['BlockType'] == 'CELL':
                    row_index = cell['RowIndex']
                    col_index = cell['ColumnIndex']
                    if row_index not in rows:
                        # create new row
                        rows[row_index] = {}

                    # get the text value
                    rows[row_index][col_index] = get_text(cell, blocks_map)
    return rows


def generate_table_csv(table_result, blocks_map, table_index):
    rows = get_rows_columns_map(table_result, blocks_map)
    table_id = 'Table_' + str(table_index)
    # get cells.
    csv = 'Table: {0}\n\n'.format(table_id)
    for row_index, cols in rows.items():

        for col_index, text in cols.items():
            csv += '{}'.format(text) + "`"
        csv += '\n'
    csv += '\n\n\n'
    return csv


def get_text(result, blocks_map):
    text = ''
    if 'Relationships' in result:
        for relationship in result['Relationships']:
            if relationship['Type'] == 'CHILD':
                for child_id in relationship['Ids']:
                    word = blocks_map[child_id]
                    if word['BlockType'] == 'WORD':
                        text += word['Text'] + ' '
                    if word['BlockType'] == 'SELECTION_ELEMENT':
                        if word['SelectionStatus'] == 'SELECTED':
                            text += 'X '
    return text


def get_table_csv(doc):
    # Get the text blocks
    blocks = doc#['Blocks']

    blocks_map = {}
    table_blocks = []
    for block in blocks:
        blocks_map[block['Id']] = block
        if block['BlockType'] == "TABLE":
            table_blocks.append(block)

    if len(table_blocks) <= 0:
        return "<b> NO Table FOUND </b>"

    csv = ''
    for index, table in enumerate(table_blocks):
        csv += generate_table_csv(table, blocks_map, index + 1)
        csv += '\n\n'
    return csv


def get_pageline_map(doc):
    blocks = doc['Blocks']
    page_child_map = {}
    page_lines = {}

    for block in blocks:
        if block['BlockType'] == "PAGE":
            if 'CHILD' in block['Relationships'][0]['Type']:
                page_child_map[block['Page']] = block['Relationships'][0]['Ids']
        if block['BlockType'] == "LINE":
            if block['Id'] in page_child_map[block['Page']]:
                if block['Page'] in page_lines:
                    page_lines[block['Page']].append(block['Text'])
                else:
                    page_lines[block['Page']] = [block['Text']]
    return page_lines


def update_bb(bb, line):
    bb['width'] += line['BoundingBox']['Width']
    bb['height'].append(line['BoundingBox']['Height'])
    bb['left'].append(line['BoundingBox']['Left'])
    bb['top'].append(line['BoundingBox']['Top'])
    return bb


def get_restructpagelines(doc):
    #pagelines = {}
    pageinfo = {}

    for page in doc.items():
        first_y = None
        lines = []
        ln = ''
        conf = []
        original_bbs = []
        original_line_nums = []
        bb = {'width':0, 'height': [], 'left': [], 'top': []}
        lnnum = 0
        first_left = None
        prev_left = None
        prev_width = None
        for line in page[1]:
            text = line['Text']
            y = line['BoundingBox']['Top']
            left = line['BoundingBox']['Left']

            if len(ln) == 0:  # empty line has text added to
                ln = text
                conf.append(line['Confidence'])
                original_bbs.append(line['BoundingBox'])
                original_line_nums.append(line['LineNum'])
                bb = update_bb(bb, line)
                first_left = line['BoundingBox']['Left']
                prev_left = first_left
                prev_width = line['BoundingBox']['Width']
                first_y = y

            elif first_y - 0.0075 <= y <= first_y + 0.0075: # filled line has text added to
                conf.append(line['Confidence'])
                original_bbs.append(line['BoundingBox'])
                original_line_nums.append(line['LineNum'])
                bb = update_bb(bb, line)

                if left < prev_left: # trying to add left-more string to the right of the line
                    ln = text + " \t" + ln
                    if left < first_left:
                        first_left = left

                else:  # new string is naturally to the right
                    prev_left = left
                    prev_width = line['BoundingBox']['Width']
                    ln += " \t" + text

            elif len(ln) != 0: # line is emptied, new text is added UNLESS slope is acceptable
                test_prev_left = prev_left
                test_prev_width = prev_width
                test_new_left = line['BoundingBox']['Left']

                if (test_new_left > (test_prev_left + test_prev_width)): # if the last word is more to the left - has to be to continue the line
                    only_slope = (first_y - y) / (test_new_left - (test_prev_left + test_prev_width))

                    if abs(only_slope) < 0.014:  # filled line has text added to
                        conf.append(line['Confidence'])
                        original_bbs.append(line['BoundingBox'])
                        original_line_nums.append(line['LineNum'])
                        bb = update_bb(bb, line)
                        prev_left = test_new_left
                        prev_width = line['BoundingBox']['Width']
                        ln += " \t" + text
                        continue

                avgconf = np.average(np.array(conf))
                wordswidth = bb['width']
                totalwidth = prev_left + prev_width - first_left
                maxheight = np.max(np.array(bb['height']))
                minleft = np.min(np.array(bb['left']))
                avgtop = np.average(np.array(bb['top']))
                lnnum += 1
                new_entry = {'LineNum': lnnum, 'Text': ln, 'Confidence': avgconf, 'OriginalLines': original_line_nums,
                             'WordsWidth': wordswidth, 'BoundingBox': { 'Width': totalwidth, 'Height': maxheight,
                                                                        'Left': minleft, 'Top': avgtop},
                             'OriginalBBs': original_bbs}

                if page[0] in pageinfo:
                    pageinfo[page[0]].append(new_entry)
                else:
                    pageinfo[page[0]] = [new_entry]

                lines.append(ln)
                ln = text
                conf = [line['Confidence']]
                original_bbs = [line['BoundingBox']]
                original_line_nums = [(line['LineNum'])]
                first_left = line['BoundingBox']['Left']
                prev_left = first_left
                prev_width = line['BoundingBox']['Width']
                bb = {'width': line['BoundingBox']['Width'], 'height': [line['BoundingBox']['Height']],
                      'left': [line['BoundingBox']['Left']], 'top': [line['BoundingBox']['Top']]}
                first_y = y

            else: # text is added straight to lines (last non-restructed text in the doc)
                lines.append(text)
                if page[0] in pageinfo:
                    pageinfo[page[0]].append(line)
                else:
                    pageinfo[page[0]] = line
                pageinfo[page[0]][lnnum]['LineNum'] = lnnum
                pageinfo[page[0]][lnnum]['WordsWidth'] = pageinfo[page[0]][lnnum]['BoundingBox']['Width']

        #  (case: last line in the document)
        lines.append(ln)
        #pagelines[page[0]] = lines

        avgconf = np.average(np.array(conf))
        wordswidth = bb['width']
        totalwidth = prev_left + prev_width - first_left
        maxheight = np.max(np.array(bb['height']))
        minleft = np.min(np.array(bb['left']))
        avgtop = np.average(np.array(bb['top']))
        lnnum += 1
        new_entry = {'LineNum': lnnum, 'Text': ln, 'Confidence': avgconf, 'OriginalLines': original_line_nums,
                     'WordsWidth': wordswidth, 'BoundingBox': { 'Width': totalwidth, 'Height': maxheight,
                                                                'Left': minleft, 'Top': avgtop},
                     'OriginalBBs': original_bbs}

        if page[0] in pageinfo:
            pageinfo[page[0]].append(new_entry)
        else: # in the case a page has only one line
            pageinfo[page[0]] = [new_entry]

    return pageinfo#, pagelines,


def get_pagelineinfo_map(doc):
    blocks = doc['Blocks']
    page_child_map = {}
    pagelineinfo = {}

    for block in blocks:
        if block['BlockType'] == "PAGE":
            if 'Relationships' in block.keys():
                if 'CHILD' in block['Relationships'][0]['Type']:
                    page_child_map[block['Page']] = block['Relationships'][0]['Ids']
        if block['BlockType'] == "LINE":
            if block['Id'] in page_child_map[block['Page']]:
                if block['Page'] in pagelineinfo:
                    pagelineinfo[block['Page']].append({'LineNum':len(pagelineinfo[block['Page']])+1,
                                                        'Text': block['Text'], 'Confidence': block['Confidence'],
                                                       'BoundingBox': block['Geometry']['BoundingBox']})
                else:
                    pagelineinfo[block['Page']] = [{'LineNum': 1, 'Text': block['Text'], 'Confidence': block['Confidence'],
                                                        'BoundingBox': block['Geometry']['BoundingBox']}]
    return pagelineinfo


def get_pageinfo(doc):
    blocks = doc['Blocks']
    pages = {}
    for block in blocks:
        if block['BlockType'] == "PAGE":
            pages[block['Page']] = block
    return pages


def get_kv_map(doc):
    blocks = doc#['Blocks']
    # get key and value maps
    key_map = {}
    value_map = {}
    block_map = {}
    for block in blocks:
        block_id = block['Id']
        block_map[block_id] = block
        if block['BlockType'] == "KEY_VALUE_SET":
            if 'KEY' in block['EntityTypes']:
                key_map[block_id] = block
            else:
                value_map[block_id] = block
    return key_map, value_map, block_map


def get_kv_relationship(key_map, value_map, block_map):
    kvs = {}
    for block_id, key_block in key_map.items():
        value_block = find_value_block(key_block, value_map)
        key = get_text(key_block, block_map)
        val = get_text(value_block, block_map)
        kvs[key] = val
    return kvs


def find_value_block(key_block, value_map):
    for relationship in key_block['Relationships']:
        if relationship['Type'] == 'VALUE':
            for value_id in relationship['Ids']:
                value_block = value_map[value_id]
    return value_block


def print_kvs(kvs):
    for key, value in kvs.items():
        print(key, ":", value)


def get_kv_pairs(result, display=False):
    key_map, value_map, block_map = get_kv_map(result)

    # Get Key Value relationship
    kvs = get_kv_relationship(key_map, value_map, block_map)
    if display:
        show_kv_pairs(kvs)
    return kvs


def show_kv_pairs(kvs):
    print("\n\n== FOUND KEY : VALUE pairs ===\n")
    print_kvs(kvs)


def search_value(kvs, search_key):
    for key, value in kvs.items():
        if re.search(search_key, key, re.IGNORECASE):
            return value


def save_tables_and_kvs(docid):
    json_file = settings.get_full_json_file(docid)
    with open(json_file, 'r') as file:
        json_doc = json.load(file)
    json_res = json2res(json_doc)['Blocks']
    save_tables(json_res, docid)
    save_kv_pairs(json_res, docid)
    print('Completed textracting ' + str(docid))


if __name__ == "__main__":
    docids = ['32730', '44448', '37802', '2646', '44603']
    for docid in docids:
        save_tables_and_kvs(docid)
