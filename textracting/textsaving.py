import boto3
import json
import re
from PIL import ImageDraw,Image, ImageFont
import psutil
import io
from pdf2image import convert_from_path, convert_from_bytes
import textracting
import settings
import glob

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


def detect_entities(text):
    # Detect entities
    entities = comprehend.detect_entities(LanguageCode="en", Text=text)
    print("\nEntities\n========")
    for entity in entities["Entities"]:
        print("{}\t=>\t{}".format(entity["Type"], entity["Text"]))


def save_annotated_doc(doc, pdf):
    blocks = doc['Blocks']

    images = convert_from_path('C:/Users/andraszeka/Downloads/welcom/smaller_100697.pdf')
    image = images[0]
    width, height = image.size
    draw = ImageDraw.Draw(image)
    print('Detected Document Text')

# Create image showing bounding box/polygon the detected lines/text
    for block in blocks:
        # print('Type: ' + block['BlockType'])
        # if block['BlockType'] != 'PAGE':
        #     print('Detected: ' + block['Text'])
        #     print('Confidence: ' + "{:.2f}".format(block['Confidence']) + "%")
        #
        # print('Id: {}'.format(block['Id']))
        # if 'Relationships' in block:
        #     print('Relationships: {}'.format(block['Relationships']))
        # print('Bounding Box: {}'.format(block['Geometry']['BoundingBox']))
        # print('Polygon: {}'.format(block['Geometry']['Polygon']))
        # print()
        draw = ImageDraw.Draw(image)
        # Draw WORD - Green -  start of word, red - end of word
        # if block['BlockType'] == "WORD":
        #     draw.line([(width * block['Geometry']['Polygon'][0]['X'],
        #                 height * block['Geometry']['Polygon'][0]['Y']),
        #                (width * block['Geometry']['Polygon'][3]['X'],
        #                 height * block['Geometry']['Polygon'][3]['Y'])], fill='green',
        #               width=2)
        #
        #     draw.line([(width * block['Geometry']['Polygon'][1]['X'],
        #                 height * block['Geometry']['Polygon'][1]['Y']),
        #                (width * block['Geometry']['Polygon'][2]['X'],
        #                 height * block['Geometry']['Polygon'][2]['Y'])],
        #               fill='red',
        #               width=2)

            # Draw box around entire LINE
        if block['BlockType'] == "LINE":
            points = []

            for polygon in block['Geometry']['Polygon']:
                points.append((width * polygon['X'], height * polygon['Y']))

            draw.polygon((points), outline='black')

            # Uncomment to draw bounding box
            box=block['Geometry']['BoundingBox']
            left = width * box['Left']
            top = height * box['Top']
            draw.rectangle([left,top, left + (width * box['Width']), top +(height * box['Height'])],outline='black')

    # Display the image
    #image.show()
    image.save("doc_image.jpeg", "JPEG")

    # display image for 10 seconds


def save_lines(doc, file_id):
    blocks = doc['Blocks']
    with open(settings.get_text_file(file_id), "w") as o:
        for block in blocks:
            if block['BlockType'] == "LINE":
                o.write(block['Text'] + '\n')
            elif block['BlockType'] == "PAGE":
                o.write('\n')


def save_tables(doc, file_id):
    table_csv = textracting.get_table_csv(doc)
    with open(settings.get_tables_file(file_id), "w") as fout:
        fout.write(table_csv)
    #print('CSV OUTPUT FILE: ', settings.get_tables_file(file_id))


def save_kv_pairs(result, file_id):
    kvs = textracting.get_kv_pairs(result)
    o = open(settings.get_kvs_file(file_id), "w")
    for key, value in kvs.items():
        o.write(str(key + ',' + value + '\n'))


def save_pageinfo(doc, file_id):
    pageinfo = textracting.get_pageinfo(doc)
    o = open(settings.get_pageinfo_file(file_id), "w")
    json.dump(pageinfo, o)
    return pageinfo


def save_restructpagelines(doc, file_id):
    restructpagelines = textracting.get_restructpagelines(doc)
    o = open(settings.get_restructpagelines_file(file_id), "w")
    json.dump(restructpagelines, o)
    return restructpagelines


def save_pagelines(doc, file_id):
    pagelines = textracting.get_pageline_map(doc)
    o = open(settings.get_pagelines_file(file_id), "w")
    json.dump(pagelines, o)
    return pagelines


def save_pagelineinfo(doc, file_id):
    pagelineinfo = textracting.get_pagelineinfo_map(doc)
    o = open(settings.get_pagelineinfo_file(file_id), "w")
    json.dump(pagelineinfo, o)
    return pagelineinfo


# helps when want to use textshowing functions with existing json files instead of the textract response
def json2res(jsondoc):
    all_blocks = []
    for pages in jsondoc:
        all_blocks.extend(pages['Blocks'])
    res = {'Blocks': all_blocks}
    return res


def restructpagelines():
    files = glob.glob('training/cleanpage/*')
    for fname in files:
        with open(fname, "r") as f:
            try:
                doc = json.load(f)
                file_id = fname.rsplit('_')[-3]
                save_restructpagelines(doc, file_id)
                print(file_id + ' successful')
            except json.decoder.JSONDecodeError:
                print(fname)


def pagelineinfo():
    files = glob.glob('training/fulljson/*')
    for fname in files:
        with open(fname, "r") as f:
            try:
                j = json.load(f)
                doc = json2res(j)
                file_id = fname.rsplit('_')[-3]
                save_pagelineinfo(doc, file_id)
                print(file_id + ' successful')
            except json.decoder.JSONDecodeError:
                print(fname)


def save_cleanpage(doc, file_id):
    cleaned = {}
    for page in doc.items():
        pagenum = page[0]
        info = page[1]
        for line in info:
            lmargin = line['BoundingBox']['Left']
            if lmargin < 0.04:  #and line['Confidence'] < 90
                if line['BoundingBox']['Width'] < 0.02:
                    print(line['Text'])
                    continue
            if pagenum in cleaned:
                cleaned[pagenum].append(line)
            else:
                cleaned[pagenum] = [line]

    o = open(settings.get_cleanpage_file(file_id), "w")
    json.dump(cleaned, o)


# Removes OCR noise from left page margins. Has been tuned for current set of reports, also catches some map/table
# lines but these are minimal and extreme and would not be desired in the text anyway. The exception to this is
# removing "Hole" from an appendix table in report 36198.
def clean_page():
    files = glob.glob('training/pagelineinfo/*')
    for fname in files:
        with open(fname, "r") as f:
            try:
                doc = json.load(f)
                file_id = fname.rsplit('_')[-3]
                save_cleanpage(doc, file_id)
                print(file_id + ' successful')
            except json.decoder.JSONDecodeError:
                print(fname)



if __name__ == "__main__":
    #pagelineinfo()
    #clean_page()
    restructpagelines()
