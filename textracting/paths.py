## @file
# Functions for getting file locations

import glob
import inspect
import os

report_folder = 'downloadedReports'
training_file_folder = 'trainingFiles'
not_training_file_foder = 'C:\\Users\\andraszeka\\OneDrive - ITP (Queensland Government)\\textract_result'
report_package_folder = 'report'
report_local_path = report_folder + '/QDEX/'
test_local_path = '../' + report_folder + '/test/'
model_path = report_package_folder + '/models/'
dataset_path = report_package_folder + '/datasets/'

dataset_version = 'expansion1'  # folder of models/data in progress of being developed
production = 'production'  # folder of last working models

production_path = production + '/'
result_path = 'results/'

## Dictionary of model names. don't currently use this, more of a reference, if needed
# ml = {'toc': 'toc',
#       'fig': 'fig',
#       'head_id_toc': 'heading_id_toc',
#       'head_id_intext': 'heading_id_intext',
#       'head_id_intext_no_toc': None,
#       'proc_head_id_toc': None,
#       'marginals': 'marginal_lines',
#       'page_id': 'page_id',
#       'page_ex': 'page_extraction',
#       'head_class': ' heading_classification'}
# ml['proc_head_id_toc'] = 'processed_' + ml['head_id_toc']
# ml['head_id_intext_no_toc'] = ml['head_id_intext'] + '_no_toc'


def get_model_path(model, mode=dataset_version, tokeniser=False, classes=False):
    path = ''
    if run_from_inside():
        path = '../'
    path += model_path + mode + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    # if type:
    #     path += '_' + type
    path += model + "_model"
    #if ('heading_id_toc' or 'page') in model and not tokeniser:
        #path += ".h5"
    if tokeniser:
        path += '_tokeniser.joblib'
    elif classes:
        path += '_class_dict.joblib'
    else:
        path += '.pkl'
    return path


def get_report_page_path(report_id, page):
    rep = get_report_name(report_id, local_path=True, file_extension=".pdf")
    page_file = rep.split('.pdf')[0]
    page_file += '_page' + str(page) + '.png'
    return page_file


## Want to tell where this function is being run from, to correctly get relative filename path
def run_from_inside():
    # i = 2
    # for x in range(i):
    #     if 'get_file_from_training' == inspect.stack()[i][3]:
    #         break
    #     if 'classify' == inspect.stack()[i][3]:
    #         break
    #     i -= 1
    # j = i + 2 # +1 for get_x_file, +1 for file of origin
    # frame = inspect.stack()[j]  # 0: this, 1: get_full_x_file, 2: file of origin
    #print(frame.filename)
    #frame = inspect.stack()[-6]
    cwd = os.getcwd()
    inside_dirs = ['report', 'borehole', 'textractor']
    try:
        #f_dir = frame.filename.split('/')[-2]
        f_dir = cwd.split('/')[-1]
    except IndexError:
        try:
            #f_dir = frame.filename.split('\\')[-2]
            f_dir = cwd.split('\\')[-1]
        except IndexError as e:
            print(e)
            exit()
    if f_dir in inside_dirs:  # if this isn't run by workflow, but a file inside a dir that needs to be gotten out of with a '../'
        return True
    return False


def get_dataset_path(dataset, mode=dataset_version):
    path = ''
    if run_from_inside():
        path = '../'
    path += dataset_path + mode + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    path += dataset + '_dataset.csv'
    return path

def get_s3_location(report_id, format='pdf', file_num=1):
    return 'QDEX/' + report_id + '/' + get_report_name(report_id, file_extension=format, file_num=file_num)


def get_s3_subdir(docid):
    return 'QDEX/' + docid + '/'


def get_report_name(report_id, local_path=False, file_extension=None, file_num=1):
    file = ''
    if local_path:
        if isinstance(local_path, str) and 'test' in local_path:
            file = test_local_path
        else:
            if run_from_inside():
                file += '../'
            file += report_local_path
        file += str(report_id) + '/'
        if not os.path.exists(file):
            os.makedirs(file)
    file += "cr_" + str(report_id) + "_" + str(file_num)
    if file_extension:
            file += file_extension
    return file


def get_file_from_training(folder, report_id, local_path, extension='.json', training=True, file_num=1, extrafolder=None):
    file = ''
    if local_path:
        fullfolder = folder
        if extrafolder:
            fullfolder = extrafolder + '/' + folder
        if training:
            if run_from_inside():
                file = '../'
            file += training_file_folder
        else:
            file += not_training_file_foder
        file += '/' + fullfolder + '/'
        if not os.path.exists(file):
            os.makedirs(file)
    if not 'cr_' in str(report_id):
        if not extension in str(report_id):
            file += get_report_name(report_id, file_num=file_num)
            return file + "_" + folder + extension
    return file + str(report_id) + "_" + folder + extension


def get_bookmarked_file(report_id, test=False, test_i='', filenum=1):
    path = ''
    if run_from_inside():
        path += '../'
    path += report_local_path
    if test:
        path = test_local_path
        files = path + str(report_id) + '/*'
        if not test_i:
            fpaths = glob.glob(files)
            test_i = len(fpaths)
    file = path + str(report_id) + "/"
    if not os.path.exists(file):
        os.makedirs(file)
    file += "cr_" + str(report_id) + "_" + str(filenum) + "_" + str(test_i) + "_bookmarked.pdf"
    return file


def get_restructpageinfo_file(report_id, local_path=True, training=True, file_num=1, extrafolder=None):
    return get_file_from_training('restructpageinfo', report_id, local_path, training=training, file_num=file_num, extrafolder=extrafolder)


def get_text_file(report_id, local_path=True, training=True, file_num=1, extrafolder=None):
    return get_file_from_training('text', report_id, local_path, training=training, file_num=file_num, extension='.txt', extrafolder=extrafolder)


def get_kvs_file(report_id, local_path=True, training=True, file_num=1, extrafolder=None):
    return get_file_from_training('kvs', report_id, local_path, extension='.csv', training=training, file_num=file_num, extrafolder=extrafolder)


def get_tables_file(report_id, local_path=True, training=True, file_num=1, bh=False, extrafolder=None):
    s = 'tables'
    if bh:
        s += '_bh'
    return get_file_from_training(s, report_id, local_path, extension='.csv', training=training, file_num=file_num, extrafolder=extrafolder)


def get_full_json_file(report_id, local_path=True, training=True, file_num=1, extrafolder=None):
    return get_file_from_training('fulljson', report_id, local_path, training=training, file_num=file_num, extrafolder=extrafolder)


def pad_num(num):
    if len(str(num)) == 2:
        num = '0' + str(num)
    elif len(str(num)) == 1:
        num = '00' + str(num)
    return num


def get_word_file(docid, file_num, service):
    base_path = not_training_file_foder + '/' + 'qutsample'
    id_path = base_path + '/' + service + '/texts/' + str(docid) +'/'
    fname = str(docid) + '_' + str(pad_num(file_num)) + '.docx'
    return id_path + fname

extensions = {'tables': 'csv', 'tables_bh': 'csv', 'kvs': 'csv',
              'fulljson': 'json', 'restructpageinfo': 'json'}


## Get files or docid, filenum pairs from a location
def get_files_from_path(type, get_file_paths=False, full_path=False, extension=None, extrafolders=None, training=True,
                        docid_only=False, one_docid=None, file_num_only=False):#, get_file_num=True):
    if full_path:
        files = glob.glob(type)
    else:
        path = ''
        if run_from_inside() and training:  # training=False uses an absolute path instead of relative
            path += '../'
        if training:
            path += training_file_folder  # don't have a training=False option for this, bc nottraining location isn't configured
        else:
            path += not_training_file_foder
        # can altenatively have option to search downloadedReports instead of trainingFiles
        if extrafolders:
            path += '/' + extrafolders
        path += '/' + type + '/'
        if not extension:
            extension = extensions.get(type)  # get file extension associated with type
        if 'json' in extension:  # files in json type folders will also have subdirs for docids
            if one_docid:
                path += one_docid
            else:
                path += '*'
            path += '/'
        if one_docid:
            path += 'cr_' + one_docid
        path += '*.' + extension
        #print(path)
        files = glob.glob(path)

    if get_file_paths:
        return files

    flines = [f.split('\\')[-1].replace('_' + type + '.' + extension, '').strip('cr_') for f in files]
    ids = [fline.split('_') for fline in flines]
    if file_num_only:
        return [id[1] for id in ids]
    if docid_only:
        return [id[0] for id in ids]
    return ids
