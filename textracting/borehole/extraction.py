## @file
# Borehole extraction from tables

import pandas as pd
from borehole.tables import get_tables
import string
from csv import writer
import os
import glob
import numpy as np


table_data_cols = ['BH', 'grid_loc_1', 'grid_loc_2', 'geo_loc_1', 'geo_loc_2',
                'BH source', 'grid_loc_1 source', 'grid_loc_2 source', 'geo_loc_1 source', 'geo_loc_2 source']
bh_data_cols = ['DocID', 'File']
bh_data_cols.extend(table_data_cols)
bh_col = []
bh_key = []
bh_loc_col = []
bh_loc_key = []
bhcsv_all = 'bh_refs_all_tables.csv'
bhcsv = 'bh_refs.csv'


## Initialise column and key terms to be extracted, from strings
def init():
    bh_col_str = 'Hole ID, borehole nu, drillhole, no. hole, hole no., Hole No. (Site No.), bore, hole, well, drill hole, ' \
                 'hole #, bore or well reference number, bore or well reference nu, borehole, uphole, holeid, bh_id, ' \
                 'well name, viell, bore no, borehole number, hole name, hole, bore site, hole number'
    bh_key_str = 'hole number, drill hole, well name and, well no, borehole no, well number, ' \
                 'Identifying name of the petroleum well, well name and number, well, well name, hole name'

    bh_geo_col_str = ''
    bh_grid_col_str = 'Survey ea, survey no, surveyed ea, surveyed no, collar (local easting, coordinates grid) northing, ' \
                      'northing, easting, Easting AMG (Local), Northing AMG (Local), Co-ords, collar, Local E & N, ' \
                      'Co-Ordinates North/East, collar east, collar north, AMG east, AMG north, East AGD66 Zone 54, ' \
                      'North AGD66 Zone 54, AMGEASTIN, AMGNORTH, east, north, Easting (m), Northing (m), ' \
                      'surveyed easting, surveyed northing, Easting AGD84 Zone 54, Northing AGD84 Zone 54, e m mga n m mga'

    bh_geo_key_str = 'Latitude, longitude, lat, long'
    bh_grid_key_str = 'northing, easting, grid (amg), grid location, surveyed location, location, field location'

    strs = [bh_col_str, bh_key_str, bh_geo_col_str, bh_grid_col_str, bh_geo_key_str, bh_grid_key_str]
    arrays = []

    for s in strs:
        a = s.split(',')
        a = [preprocess_str(w) for w in a]
        arrays.append(a)

    global bh_col
    bh_col = arrays[0]
    global bh_key
    bh_key = arrays[1]
    global bh_geo_col
    bh_geo_col = arrays[2]
    global bh_grid_col
    bh_grid_col = arrays[3]
    global bh_geo_key
    bh_geo_key = arrays[4]
    global bh_grid_key
    bh_grid_key = arrays[5]


## Save rows to csv
def save_rows(file_name: str, df: pd.DataFrame):
    # Open file in append mode
    write_cols = False
    if not os.path.exists(file_name):
        write_cols = True
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        if write_cols:
            csv_writer.writerow(df.columns.values)
        csv_writer.writerows(df.values)


## Pre-process string
def preprocess_str(s):
    s = str(s).lower()
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = s.strip(' ')
    s = s.replace('\n', '')
    return s


## Searches for a key value to downwards, and to the right, and returns it
def find_val_from_key(table, source_i, type='loc'):
    val = None
    val_down = search(table, [source_i[0] + 1, source_i[1]], dir='down', type=type)
    val_right = search(table, [source_i[0], source_i[1] + 1], dir='right', type=type)
    print('val down: ', val_down, 'val right: ', val_right)
    if val_down:
        val = val_down
    if val_right:
        val = val_right
    # if finds down and right values, returns right: can change this to find the better value
    return val


## Checks if a string has numerical characters
def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)


## Checks if a borehole name is valid (contains numbers)
# Can add more conditions: eg. check for a pattern
def validate_bh(bh):
    p_bh = str(bh).lower()
    has_nums = hasNumbers(p_bh)
    if not has_nums:
        return 'invalid_no_num'
    if 'unnamed' in p_bh:
        return 'invalid_nan'
    if p_bh == 'nan':
        return 'invalid_nan'
    return 'valid'


## Checks if a value is valid as a representation of location (numerical, with some other characters allowed)
def validate_loc(loc):
    p_loc = str(loc).lower()
    contains_letters = p_loc.islower()

    if contains_letters:
        p_loc = p_loc.strip()  # remove leading and trailing whitespace
        l_loc = p_loc.replace('amg', '')
        l_loc = l_loc.replace('s', '').replace('n', '').replace('e','').replace('w','').replace('m', '')
        l_loc = l_loc.replace('deg', '').replace('min', '').replace('sec', '')
        contains_letters = l_loc.islower()

        if contains_letters:
            print('loc ', loc, ' invalid')
            return 'invalid'
    if not p_loc:
        print('loc nan')
        return 'invalid_nan'

    print('loc ', loc, ' valid')
    return 'valid'


## Searches for a value in a table based on index of its key
def search(table, source_i, dir='right', type='loc'):
    try:
        val = table.iloc[source_i[0], source_i[1]]
    except IndexError:
        return False
    if type == 'loc':
        valid_val = validate_loc(val)
    elif type == 'bh':
        valid_val = validate_bh(val)
    elif type =='num':
        p_val = str(val).lower()
        valid_val = not p_val.islower()  # inverse of: contains_letters
        if not valid_val:
            valid_val = 'invalid'
        else:
            valid_val = 'valid'

    if 'invalid' in valid_val:
        if 'nan' in valid_val:
            if dir == 'right':
                 source_i[1] += 1
            elif dir == 'down':
                source_i[0] += 1
            return search(table, [source_i[0], source_i[1]], dir=dir, type=type)  # can run into index error
        if 'no_num' in valid_val:  # case: bh and num are in separate cells
            num = search(table, [source_i[0], source_i[1] + 1], dir='right', type='num')
            if num:
                val = val + ' ' + str(num)
                return val
        return False
    else:
        #print('search ', dir, ' for ', val, ', type: ', type, ' successful')
        return val


## Extracts values from table (with assumption that it is key-based)
def extract_from_keys(table):
    bh_source = None
    bh_source_i = None
    grid_source = []
    grid_source_i = []
    geo_source = []
    geo_source_i = []

    cols = pd.DataFrame([table.columns.values], columns=table.columns.values)
    alltable = pd.concat([cols, table], ignore_index=True)  # put "column name" values into the table to be worked with
    for i, row in alltable.iterrows():
        for j, cell in zip(range(len(row)), row):
            cvalue = preprocess_str(cell)
            if cvalue in bh_key:
                bh_source = cell
                bh_source_i = [i,j]  # +1, bc assuming value of the key will be next cell accross
            elif cvalue in bh_geo_key:
                geo_source.append(cell)
                geo_source_i.append([i,j])
            elif cvalue in bh_grid_key:
                grid_source.append(cell)
                grid_source_i.append([i,j])
    if bh_source:
        bh = find_val_from_key(alltable, bh_source_i, type='bh')
        if bh:
            grid_loc = []
            geo_loc = []
            for i in range(len(grid_source)):
                grid_loc.append([find_val_from_key(alltable, grid_source_i[i], type='loc')])
            for i in range(len(geo_source)):
                geo_loc.append([find_val_from_key(alltable, geo_source_i[i], type='loc')])

            return extracted_to_df(bh, grid_loc, geo_loc, bh_source, grid_source, geo_source)
    return None


## Gets extracted data into DataFrame format
def extracted_to_df(bhs, grid_loc, geo_loc, bh_source, grid_source, geo_source):
    if not isinstance(bhs, np.ndarray):
        bhs = [bhs]

    for i in range(2):
        if len(grid_loc) <= i:
            grid_loc.append(None)
            grid_source.append(None)
        if len(geo_loc) <= i:
            geo_loc.append(None)
            geo_source.append(None)

    rows = len(bhs)
    #print(rows)
    add_to_df = [bhs, grid_loc[0], grid_loc[1], geo_loc[0], geo_loc[1],
                 [bh_source], [grid_source[0]], [grid_source[1]], [geo_source[0]], [geo_source[1]]]
    for i in range(len(add_to_df)):
        if add_to_df[i] is None:
            add_to_df[i] = [None for e in range(rows)]
        elif len(add_to_df[i]) < rows:
            if len(add_to_df[i]) == 1:
                add_to_df[i] = [add_to_df[i][0] for e in range(rows)]

    dfdata = [pd.Series(a) for a in add_to_df]

    df_dict = {key: value for key, value in zip(table_data_cols, dfdata)}
    add_to_df = pd.DataFrame(df_dict, columns=table_data_cols)
    #print(add_to_df)
    return add_to_df


## Extracts values from table (with assumption that it is column-based)
def extract_from_columns(table):
    bh_source = None
    grid_source = []
    geo_source = []
    for name in table.columns:
        proc_name = preprocess_str(name)
        if proc_name in bh_col:
            #print('bh name: ', name)
            bh_source = name
        elif proc_name in bh_geo_col:
            #print('bh loc name: ', name)
            geo_source.append(name)
        elif proc_name in bh_grid_col:
            grid_source.append(name)

    if bh_source:
        bhs = table[bh_source].values
        bad_indices = []
        for j in range(len(bhs)):
            valid_val = validate_bh(bhs[j])
            if 'invalid' in valid_val:
                bad_indices.append(j)
        valid_bhs = np.delete(bhs, bad_indices)

        grid_loc = []
        geo_loc = []
        for i in range(len(grid_source)):
            locs = table[grid_source[i]].values
            valid_locs = np.delete(locs, bad_indices)
            grid_loc.append(valid_locs)
        for i in range(len(geo_source)):
            locs = table[geo_source[i]].values
            valid_locs = np.delete(locs, bad_indices)
            geo_loc.append(valid_locs)

        return extracted_to_df(valid_bhs, grid_loc, geo_loc, bh_source, grid_source, geo_source)
    return None


## Extract boreholes for a certain report
def extract_bh(docid, filenum=None, bh=True, training=True, extrafolder='', fname=bhcsv_all):
    sep = '`'
    if 'wondershare' in extrafolder:
        sep = ','
    if not filenum:
        fs = []
        if '_' not in docid:
            if not training:
                files = glob.glob('C:\\Users\\andraszeka\\OneDrive - ITP (Queensland Government)\\textract_result\\' + extrafolder + '/tables/cr_' + docid + '*.csv')
            else:
                files = glob.glob('training/tables/cr_' + docid + '*.csv')
            for file in files:
                f = file.split('\\')[-1].replace('_tables.csv', '').replace('cr_' + docid + '_', '')
                fs.append(f)
        else:
            docid, file = docid.split('_')
            fs = [file]
    else:
        fs = [filenum]

    for file in fs:
        try:
            bhtables = get_tables(docid, bh=bh, report_num=file, training=training, extrafolder=extrafolder, sep=sep)
        except FileNotFoundError:
            print('No file for ', str(docid), '_', file, ' bh: ', str(bh))
            return
        bh_data = pd.DataFrame(columns=bh_data_cols)

        for table in bhtables:
            res1 = extract_from_columns(table)
            res2 = extract_from_keys(table)
            # if isinstance(res1, pd.DataFrame) and isinstance(res2, pd.DataFrame):
            #     # compare the two and throw one out
            #     print('comparing')
            #     res = res1 # placeholder
            #     bh_data = bh_data.append(res, ignore_index=True)
            # else:
            if isinstance(res1, pd.DataFrame):
                bh_data = bh_data.append(res1, ignore_index=True)
            if isinstance(res2, pd.DataFrame):
                bh_data = bh_data.append(res2, ignore_index=True)

        bh_data['DocID'] = docid
        bh_data['File'] = file
        #if not bh:
        #    fname = bhcsv_all
        if bh:
            fname = bhcsv
        save_rows(fname, bh_data)


## Get report IDs of table files
def get_table_docids(bh=False, training=True, extrafolder=None):
    docids = []
    if not bh:
        folder = 'tables'
    else:
        folder = 'bh_tables'

    if extrafolder:
        folder = extrafolder + '/' + folder
    if training:
        lines_docs = glob.glob('training/' + folder + '/*.csv')
    else:
        lines_docs = glob.glob('C:\\Users\\andraszeka\\OneDrive - ITP (Queensland Government)\\textract_result/' + folder + '/*.csv')

    for lines_doc in lines_docs:
        docid = lines_doc.split('\\')[-1].replace('_tables.csv', '').strip('cr_')
        docids.append(docid)
    return docids


## Removes duplicates from csv
def manage_data(fname):
    df = pd.read_csv(fname, engine='python')
    df = df.drop_duplicates()
    df.to_csv(fname, index=False)


## Extracts boreholes for all report IDs (which have table files)
def extract_for_all_docids(training=True, extrafolder=None):
    init()
    docids = get_table_docids(training=training, extrafolder=extrafolder)
    for id in docids:
        extract_bh(id, bh=False, training=training)
        #extract_bh(id, bh=True)
    #manage_data(bhcsv)
    manage_data(bhcsv_all)


##Borehole extraction for a certain report (with init and data cleaning)
def extract_for_docid(docid, filenum=None, fname=bhcsv_all, training=False):
    init()
    extract_bh(docid, bh=False, filenum=filenum, fname=fname, training=training)
    manage_data(fname)


## Pads number to three digits [for getting files which represent file number as three digits]
def pad_num(num):
    if len(str(num)) == 2:
        num = '0' + str(num)
    elif len(str(num)) == 1:
        num = '00' + str(num)
    return num

if __name__ == "__main__":
    # init()
    # tx_extrafolder = 'qutsample/textract'
    # ws_extrafolder = 'qutsample/wondershare'

    # intersect = [['14142', '1'], ['14142', '2'], ['14142', '3'], ['14142', '4'], ['14142', '5'], ['14142', '6'], ['14142', '7'], ['14142', '8'], ['14142', '9'], ['14142', '10'], ['14142', '11'], ['14142', '12'], ['14142', '13'], ['14142', '14'], ['14142', '15'], ['14142', '19'], ['14142', '22'], ['14142', '23'], ['14142', '25'], ['14142', '39'], ['1664', '1'], ['1664', '2'], ['1664', '3'], ['1664', '4'], ['1664', '5'], ['1664', '6'], ['1664', '7'], ['1664', '8'], ['1664', '9'], ['1664', '10'], ['1664', '11'], ['1664', '12'], ['1664', '13'], ['1664', '14'], ['1664', '15'], ['1664', '16'], ['1664', '17'], ['1664', '18'], ['1664', '19'], ['1664', '20'], ['1664', '21'], ['1664', '23'], ['1664', '25'], ['1664', '27'], ['167', '1'], ['167', '2'], ['1799', '1'], ['1799', '2'], ['1799', '3'], ['1799', '4'], ['1799', '5'], ['1799', '6'], ['1799', '7'], ['1799', '8'], ['1799', '9'], ['1799', '10'], ['1799', '16'], ['1799', '18'], ['1799', '20'], ['1799', '21'], ['1799', '23'], ['1799', '29'], ['21166', '1'], ['21166', '2'], ['21166', '3'], ['21166', '4'], ['21166', '5'], ['21166', '6'], ['21166', '7'], ['21166', '15'], ['23455', '1'], ['23455', '2'], ['23455', '3'], ['23455', '4'], ['23455', '5'], ['23455', '8'], ['23455', '10'], ['23455', '11'], ['23455', '12'], ['23455', '13'], ['23455', '14'], ['23455', '15'], ['23455', '16'], ['23455', '17'], ['23455', '18'], ['23455', '19'], ['23455', '20'], ['23455', '21'], ['23455', '22'], ['23455', '23'], ['23455', '24'], ['27932', '1'], ['27932', '2'], ['27932', '3'], ['27932', '4'], ['27932', '5'], ['27932', '14'], ['27932', '18'], ['28822', '1'], ['28822', '2'], ['28822', '4'], ['28822', '5'], ['28822', '6'], ['28822', '7'], ['28822', '8'], ['28822', '9'], ['28822', '10'], ['28822', '11'], ['28822', '12'], ['28822', '15'], ['28822', '16'], ['28822', '18'], ['29695', '1'], ['29695', '4'], ['29695', '5'], ['29695', '16'], ['29695', '22'], ['29695', '23'], ['29695', '24'], ['29695', '25'], ['29695', '26'], ['30479', '1'], ['30479', '2'], ['30479', '3'], ['30479', '4'], ['30479', '5'], ['30479', '6'], ['30479', '7'], ['30479', '8'], ['30479', '9'], ['30479', '15'], ['31511', '1'], ['3354', '1'], ['3354', '2'], ['3354', '3'], ['3354', '4'], ['3354', '5'], ['3354', '6'], ['3354', '7'], ['3354', '8'], ['3354', '11'], ['3354', '12'], ['3354', '14'], ['3354', '17'], ['33931', '1'], ['33931', '4'], ['33931', '7'], ['33931', '8'], ['33931', '9'], ['33931', '10'], ['33931', '11'], ['3769', '1'], ['3769', '2'], ['3769', '3'], ['3769', '4'], ['3769', '5'], ['3769', '6'], ['3769', '7'], ['3769', '8'], ['3769', '9'], ['3769', '10'], ['3769', '11'], ['3769', '12'], ['3769', '13'], ['3769', '16'], ['3769', '21'], ['37802', '1'], ['42688', '1'], ['42688', '2'], ['46519', '1'], ['46519', '2'], ['46519', '3'], ['46519', '4'], ['46519', '5'], ['46519', '6'], ['504', '1'], ['504', '2'], ['504', '3'], ['504', '4'], ['504', '5'], ['504', '6'], ['504', '8'], ['504', '9'], ['504', '10'], ['504', '11'], ['504', '12'], ['504', '14'], ['504', '15'], ['51800', '2'], ['53382', '1'], ['53382', '2'], ['5992', '1'], ['5992', '2'], ['5992', '3'], ['5992', '4'], ['5992', '5'], ['5992', '6'], ['5992', '9'], ['63981', '1'], ['63981', '3'], ['63981', '5'], ['63981', '6'], ['64818', '1'], ['64818', '3'], ['64818', '4'], ['801', '1'], ['801', '2'], ['801', '3'], ['801', '4'], ['801', '6'], ['801', '7'], ['801', '8']]
    #
    # for i in intersect:
    #     docid, filenum = i[0], i[1]
    #     extract_bh(docid, filenum=pad_num(filenum), bh=False, training=False, extrafolder=ws_extrafolder, fname='ws_tables.csv')
    #     #extract_bh(docid, filenum=filenum, bh=False, training=False, extrafolder=tx_extrafolder, fname='tx_tables.csv')

    # extract_bh('37802', filenum=pad_num('1'), bh=False, training=False, extrafolder=ws_extrafolder,
    #            fname='ws_tables.csv')
    # manage_data('ws_tables.csv')
    #manage_data('tx_tables.csv')

    #coal = '2646 3050 25335 32730 33720 34372 35132 35152 35454 35500 36675 40923 41674 41720 41932 44638 47465 48384 48406 48777 49264 50481 55076 55636 64268 64479 65455 68354 76875 81735 85174 90461 99356 100291 10609200'
    #coal = '25335 34372 35500 36675 40923 41674 41720 41932 44638 48384 48406 48777 49264 55636 64479 68354 76875 81735 85174 90461'
    coal = '41932 '
    coals = coal.split()

    for i in coals:  # may not have all of these textracted
        extract_for_docid(i, fname='missing_coal_sample.csv')



    #manage_data(bhcsv)
    #manage_data(bhcsv_all)
    #docids = ['106092', '99356', '92099', '84329', '77290', '72095', '69365', '99419']
    #for id in docids:
    #    extract_for_docid(id)

    #extract_for_docid('44603')