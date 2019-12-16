
report_local_path = 'training/QDEX/'
tree_model_file = 'tree_model.pkl'

def get_report_name(file_id, local_path=False, file_extension=False):
    file = ''
    if local_path:
        file = report_local_path + file_id + '/'
    file += "cr_" + str(file_id) + "_1"
    if file_extension:
        file += '.pdf'
    return file


def get_file_from_training(folder, file_id, local_path, extension='.json'):
    file = ''
    if local_path:
        file = 'training/' + folder + '/'
    if not 'cr_' in file_id:
        if not extension in file_id:
            file += get_report_name(file_id)
            return file + "_" + folder + extension
    return file + str(file_id) + "_" + folder + extension


def get_pageinfo_file(file_id, local_path=True):
    return get_file_from_training('pageinfo', file_id, local_path)


def get_pagelineinfo_file(file_id, local_path=True):
    return get_file_from_training('pagelineinfo', file_id, local_path)


def get_pagelines_file(file_id, local_path=True):
    return get_file_from_training('pagelines', file_id, local_path)


def get_restructpagelines_file(file_id, local_path=True):
    return get_file_from_training('restructpagelines', file_id, local_path)


def get_kvs_file(file_id, local_path=True):
    return get_file_from_training('kvs', file_id, local_path, extension='.csv')


def get_tables_file(file_id, local_path=True):
    return get_file_from_training('tables', file_id, local_path, extension='.csv')


def get_text_file(file_id, local_path=True):
    return get_file_from_training('text', file_id, local_path, extension='.txt')


def get_full_json_file(file_id, local_path=True):
    return get_file_from_training('fulljson', file_id, local_path)


def get_cleanpage_file(file_id, local_path=True):
    return get_file_from_training('cleanpage', file_id, local_path)