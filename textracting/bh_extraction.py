import pandas as pd
from borehole_tables import get_tables
import string
from csv import writer
import os
import glob
import numpy as np
import re


table_data_cols = ['BH', 'location1', 'location2',
                'BH source', 'location1 source', 'location2 source']
bh_data_cols = ['DocID', 'File']
table_data_cols.extend(bh_data_cols)
bh_col = []
bh_key = []
bh_loc_col = []
bh_loc_key = []

bhcsv_all = 'bh_refs_all_tables.csv'
bhcsv = 'bh_refs.csv'


def init():
    bh_col_str = 'Hole ID, borehole nu, drillhole, no. hole, hole no., Hole No. (Site No.), bore, hole, well, drill hole, hole #, bore or well reference number, bore or well reference nu, borehole, uphole, holeid, bh_id, well name, viell, bore no'
    bh_key_str = ' hole number, drill hole, well name and, well no, borehole no, well number, Identifying name of the petroleum well, well name and number, well, well name, hole name'

    bh_loc_col_str = 'Survey ea, survey no, surveyed ea, surveyed no, collar (local easting, coordinates grid) northing, northing, easting, Easting AMG (Local), Northing AMG (Local), Co-ords, collar, Local E & N, Co-Ordinates North/East, collar east, collar north, AMG east, AMG north, East AGD66 Zone 54, North AGD66 Zone 54, AMGEASTIN, AMGNORTH, east, north, Easting (m), Northing (m) '
    bh_loc_key_str = 'Latitude, longitude, northing, easting, surveyed location, lat, long, location, grid (amg), grid location, field location'

    strs = [bh_col_str, bh_key_str, bh_loc_col_str, bh_loc_key_str]
    arrays = []

    for s in strs:
        a = s.split(',')
        a = [preprocess_str(w) for w in a]
        arrays.append(a)

    global bh_col
    bh_col = arrays[0]
    global bh_key
    bh_key = arrays[1]
    global bh_loc_col
    bh_loc_col = arrays[2]
    global bh_loc_key
    bh_loc_key = arrays[3]


def save_rows(file_name, df):
    # Open file in append mode
    write_cols = False
    if not os.path.exists(file_name):
        write_cols = True
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        if write_cols:
            csv_writer.writerow(df.columns.values)
        csv_writer.writerows(df.values)


def preprocess_str(s):
    s = str(s).lower()
    s = s.translate(str.maketrans('', '', string.punctuation))
    s = s.strip(' ')
    return s


def find_loc_from_key(table, loc_source_i):
    try:
        loc = table.iloc[loc_source_i[0], loc_source_i[1]+1]  # first look in the cell to the right
    except IndexError:
        loc = ''
    valid_loc = verify_loc(loc)
    if 'invalid' in valid_loc:
        loc_down = search_down(table, [loc_source_i[0] + 1, loc_source_i[1]])
        loc_right = search_right(table, [loc_source_i[0], loc_source_i[1] + 1])
        print('loc down: ', loc_down, 'loc right: ', loc_right)
        if loc_down:
            loc = loc_down
        if loc_right:
            loc = loc_right
    return loc


def verify_loc(loc):
    loc = str(loc).lower()
    contains_letters = loc.islower()

    if contains_letters:
        loc = loc[:-1].lower()
        contains_letters = loc.islower()

        if contains_letters:
            print('loc ', loc, ' invalid')
            return 'invalid'
    if not loc:
        print('loc nan')
        return 'invalid_nan'

    print('loc ', loc, ' valid')
    return 'valid'


def search_down(table, loc_source_i):
    try:
        loc = table.iloc[loc_source_i[0], loc_source_i[1]]
    except IndexError:
        return False
    valid_loc = verify_loc(loc)
    if 'invalid' in valid_loc:
        if 'nan' in valid_loc:
            return search_down(table, [loc_source_i[0] + 1, loc_source_i[1]])  # can run into index error
        return False
    else:
        print('search down for ', loc, ' successful')
        return loc


def search_right(table, loc_source_i):
    try:
        loc = table.iloc[loc_source_i[0], loc_source_i[1]]
    except IndexError:
        return False

    valid_loc = verify_loc(loc)
    if 'invalid' in valid_loc:
        if 'nan' in valid_loc:
            return search_right(table, [loc_source_i[0], loc_source_i[1]+1])  # can run into index error
        return False
    else:
        print('search right for ', loc, ' successful')
        return loc


def extract_from_keys(table):
    # for location: if the next cell's value after loc source is a float, (without the last char because it could be e or n)
    loc = 0
    bh_source = None
    bh_source_i = None
    loc1_source = None
    loc1_source_i = None
    loc2_source = None
    loc2_source_i = None
    cols = pd.DataFrame([table.columns.values], columns=table.columns.values)
    alltable = pd.concat([cols, table], ignore_index=True)  # put "column name" values into the table to be worked with
    for i, row in alltable.iterrows():
        for j, cell in zip(range(len(row)), row):
            cvalue = preprocess_str(cell)
            if cvalue in bh_key:
                bh_source = cell
                bh_source_i = [i,j]  # +1, bc assuming value of the key will be next cell accross
            elif cvalue in bh_loc_key:
                if loc == 0:
                    loc1_source = cell
                    loc1_source_i = [i,j]
                    loc += 1
                elif loc == 1:
                    loc2_source = cell
                    loc2_source_i = [i,j]
                else:
                    print('more than two loc? :', cell)
    if bh_source:
        try:
            bh = alltable.iloc[bh_source_i[0], bh_source_i[1]+1] # assuming BH name will be to the right, but need to write something as for loc
        except IndexError:
            print('bh index error, trying search down')
            try:
                bh = alltable.iloc[bh_source_i[0]+1, bh_source_i[1]]
            except IndexError:
                print("can't look right or down for bh")
                return None
        loc1 = None
        loc2 = None
        if loc1_source:
            loc1 = find_loc_from_key(alltable, loc1_source_i)
        if loc2_source_i:
            loc2 = find_loc_from_key(alltable, loc2_source_i)

        return extracted_to_df(bh, loc1, loc2, bh_source, loc1_source, loc2_source)
    return None


def extracted_to_df(bhs, loc1s, loc2s, bh_source, loc1_source, loc2_source):
    if not isinstance(bhs, np.ndarray):
        bhs = [bhs]
    if not isinstance(loc1s, np.ndarray):
        loc1s = [loc1s]
    if not isinstance(loc2s, np.ndarray):
        loc2s = [loc2s]

    rows = len(bhs)
    print(rows)
    add_to_df = [bhs, loc1s, loc2s, [bh_source], [loc1_source], [loc2_source]]
    for i in range(len(add_to_df)):
        if add_to_df[i] is None:
            add_to_df[i] = [None for e in range(rows)]
        elif len(add_to_df[i]) < rows:
            if len(add_to_df[i]) == 1:
                add_to_df[i] = [add_to_df[i][0] for e in range(rows)]

    dfdata = [pd.Series(a) for a in add_to_df]

    df_dict = {key: value for key, value in zip(table_data_cols, dfdata)}
    add_to_df = pd.DataFrame(df_dict, columns=table_data_cols)
    print(add_to_df)
    return add_to_df


def extract_from_columns(table):
    loc = 0
    bh_source = None
    loc1_source = None
    loc2_source = None
    for name in table.columns:
        proc_name = preprocess_str(name)
        if proc_name in bh_col:
            print('bh name: ', name)
            bh_source = name
        elif proc_name in bh_loc_col:
            print('bh loc name: ', name)
            if loc == 0:
                loc1_source = name
                loc += 1
            elif loc == 1:
                loc2_source = name
            else:
                print('more than two loc? :', name)
    if bh_source:
        bhs = table[bh_source].values
        loc1s = None
        loc2s = None
        if loc1_source:
            loc1s = table[loc1_source].values
        if loc2_source:
            loc2s = table[loc2_source].values

        return extracted_to_df(bhs, loc1s, loc2s, bh_source, loc1_source, loc2_source)
    return None


def extract_bh(docid, bh=True):
    fs = []
    if '_' not in docid:
        files = glob.glob('training/tables/cr_' + docid + '*.csv')
        for file in files:
            f = file.split('\\')[-1].replace('_tables.csv', '').replace('cr_' + docid + '_', '')
            fs.append(f)
    else:
        docid, file = docid.split('_')
        fs = [file]

    for file in fs:
        try:
            bhtables = get_tables(docid, bh=bh, report_num=file)
        except FileNotFoundError:
            print('No file for ', str(docid), '_', file, ' bh: ', str(bh))
            return
        bh_data = pd.DataFrame(columns=bh_data_cols)

        for table in bhtables:
            res1 = extract_from_columns(table)
            if isinstance(res1, pd.DataFrame):
                bh_data = bh_data.append(res1, ignore_index=True)
            res2 = extract_from_keys(table)
            if isinstance(res2, pd.DataFrame):
                bh_data = bh_data.append(res2, ignore_index=True)


        bh_data['DocID'] = docid
        bh_data['File'] = file
        if not bh:
            fname = bhcsv_all
        else:
            fname = bhcsv
        save_rows(fname, bh_data)


def get_table_docids(bh=False):
    docids = []
    if not bh:
        folder = 'tables'
    else:
        folder = 'bh_tables'

    lines_docs = glob.glob('training/' + folder + '/*.csv')
    for lines_doc in lines_docs:
        docid = lines_doc.split('\\')[-1].replace('_tables.csv', '').strip('cr_')
        docids.append(docid)
    return docids


def manage_data(fname):
    df = pd.read_csv(fname)
    df = df.drop_duplicates()
    df.to_csv(fname, index=False)


def extract_for_all_docids():
    init()
    docids = get_table_docids()
    for id in docids:
        extract_bh(id, bh=False)
        #extract_bh(id, bh=True)
    #manage_data(bhcsv)
    manage_data(bhcsv_all)


def extract_for_docid(docid):
    init()
    extract_bh(docid, bh=False)
    manage_data(bhcsv_all)


if __name__ == "__main__":
    extract_for_all_docids()
    #manage_data(bhcsv)
    #manage_data(bhcsv_all)

    #extract_for_docid('60237')