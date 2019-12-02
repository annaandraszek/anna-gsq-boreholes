import boto3
import json
import re
from PIL import Image, ImageDraw, ImageFont
import psutil
import io
from pdf2image import convert_from_path, convert_from_bytes
import textracting


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


def save_lines(doc, outfile="doc_lines.txt", mode='w'):
    blocks = doc['Blocks']
    with open(outfile, mode) as o:
        for block in blocks:
            if block['BlockType'] == "LINE":
                o.write(block['Text'] + '\n')
            elif block['BlockType'] == "PAGE":
                o.write('\n')


def save_tables(doc, outfile="doc_tables.csv", mode="w"):
    table_csv = textracting.get_table_csv(doc)

    with open(outfile, mode) as fout:
        fout.write(table_csv)

    print('CSV OUTPUT FILE: ', outfile)


def save_kv_pairs(result, outfile="doc_kv_pairs.csv", mode="w"):
    kvs = textracting.get_kv_pairs(result)
    o = open(outfile, mode)
    for key, value in kvs.items():
        o.write(str(key + ',' + value + '\n'))
