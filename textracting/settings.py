
report_local_path = 'training/QDEX/'


def get_report_name(file_id, local_path=False, file_extension=False):
    file = ''
    if local_path:
        file = report_local_path + file_id + '/'
    file += "cr_" + str(file_id) + "_1"
    if file_extension:
        file += '.pdf'
    return file


def get_pageinfo_file(file_id, local_path=True):
    file = ''
    if local_path:
        file = 'training/pageinfo/'
    file += str(file_id) + "_pageinfo.json"
    return file


def get_pagelines_file(file_id, local_path=True):
    file = ''
    if local_path:
        file = 'training/pagelines/'
    file += str(file_id) + "_pagelines.json"
    return file


def get_kvs_file(file_id, local_path=True):
    file = ''
    if local_path:
        file = 'training/kvs/'
    file += str(file_id) + "_kvs.csv"
    return file


def get_tables_file(file_id, local_path=True):
    file = ''
    if local_path:
        file = 'training/tables/'
    file += str(file_id) + '_tables.csv'
    return file


def get_text_file(file_id, local_path=True):
    file = ''
    if local_path:
        file = 'training/text/'
    file += str(file_id) + "_text.txt"
    return file


def get_full_json_file(file_id, local_path=True):
    file = ''
    if local_path:
        file = 'training/full_json/'
    file += str(file_id) + ".json"
    return file

