import boto3
import json
import re
from PIL import Image, ImageDraw, ImageFont
import psutil
import io
from pdf2image import convert_from_path, convert_from_bytes

from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)

textract = boto3.client('textract')
comprehend = boto3.client('comprehend')


def analyse_doc():
    response = textract.start_document_analysis(
        ClientRequestToken = "two",
        DocumentLocation={
            #'Bytes': b'bytes',
            'S3Object': {
                'Bucket': 'gsq-ml',
                'Name': 'cr_100697_1.pdf',
                #'Version': 'string'
            }
        },
        FeatureTypes=[
            'TABLES','FORMS',
        ],
        JobTag = "well_report"
    )
    return response


def get_analysed(response):
    res = textract.get_document_analysis(
        JobId = response['JobId'],
        #NextToken = response['NextToken']
    )
    fp = open('textract_result.json', 'w')
    json.dump(res, fp)

    while True:
        try:
            if res['NextToken']:
                res = textract.get_document_analysis(
                    JobId=response['JobId'],
                    NextToken = res['NextToken']
                )
                json.dump(res, fp)
        except KeyError:
            break
    return res


def print_doc_lines(doc):
    # Print text
    print("\nText\n========")
    text = ""
    for item in doc["Blocks"]:
        if item["BlockType"] == "LINE":
            print('\033[94m' + item["Text"] + '\033[0m')
            text = text + " " + item["Text"]
    return text


def get_kv_map(doc):
    blocks = doc['Blocks']
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


def print_kvs(kvs):
    for key, value in kvs.items():
        print(key, ":", value)


def save_kv_pairs(result, outfile="doc_kv_pairs.csv", mode="w"):
    kvs = get_kv_pairs(result)
    o = open(outfile, mode)
    for key, value in kvs.items():
        o.write(str(key + ',' + value + '\n'))


def search_value(kvs, search_key):
    for key, value in kvs.items():
        if re.search(search_key, key, re.IGNORECASE):
            return value


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


def save_lines(doc, outfile="doc_lines.txt", mode='w'):
    blocks = doc['Blocks']
    with open(outfile, mode) as o:
        for block in blocks:
            if block['BlockType'] == "LINE":
                o.write(block['Text'] + '\n')
            elif block['BlockType'] == "PAGE":
                o.write('\n')


if __name__ == "__main__":
    response = analyse_doc()
    result = get_analysed(response)
    #print_doc_tables(result)
    fname = 'textract_result_cr1006971.json'
    fp = open(fname, 'r')
    result = json.load(fp)
    #show_doc(result, fname)
    #show_kv_pairs(result)
    save_lines(result)
    get_kv_pairs(result)
    #text = print_doc_tables(result)
    #detect_entities(text[:5000])