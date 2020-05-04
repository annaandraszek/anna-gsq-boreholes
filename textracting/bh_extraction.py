import pandas as pd
from borehole_tables import get_tables
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


def init():
    bh_col_str = 'Hole ID, borehole nu, drillhole, no. hole, hole no., Hole No. (Site No.), bore, hole, well, drill hole, ' \
                 'hole #, bore or well reference number, bore or well reference nu, borehole, uphole, holeid, bh_id, well name, viell, bore no, borehole number'
    bh_key_str = 'hole number, drill hole, well name and, well no, borehole no, well number, ' \
                 'Identifying name of the petroleum well, well name and number, well, well name, hole name'

    bh_geo_col_str = ''
    bh_grid_col_str = 'Survey ea, survey no, surveyed ea, surveyed no, collar (local easting, coordinates grid) northing, ' \
                      'northing, easting, Easting AMG (Local), Northing AMG (Local), Co-ords, collar, Local E & N, ' \
                      'Co-Ordinates North/East, collar east, collar north, AMG east, AMG north, East AGD66 Zone 54, ' \
                      'North AGD66 Zone 54, AMGEASTIN, AMGNORTH, east, north, Easting (m), Northing (m), ' \
                      'surveyed easting, surveyed northing, Easting AGD84 Zone 54, Northing AGD84 Zone 54 '

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


def find_val_from_key(table, source_i, type='loc'):
    val = None
    val_down = search(table, [source_i[0] + 1, source_i[1]], dir='down', type=type)
    val_right = search(table, [source_i[0], source_i[1] + 1], dir='right', type=type)
    print('val down: ', val_down, 'val right: ', val_right)
    if val_down:
        val = val_down
    if val_right:
        val = val_right

    return val


def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)


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
    print(rows)
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
    print(add_to_df)
    return add_to_df


def extract_from_columns(table):
    bh_source = None
    grid_source = []
    geo_source = []
    for name in table.columns:
        proc_name = preprocess_str(name)
        if proc_name in bh_col:
            print('bh name: ', name)
            bh_source = name
        elif proc_name in bh_geo_col:
            print('bh loc name: ', name)
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


def extract_bh(docid, bh=True, training=True):
    fs = []
    if '_' not in docid:
        if not training:
            files = glob.glob('C:\\Users\\andraszeka\\OneDrive - ITP (Queensland Government)\\textract_result\\tables/cr_' + docid + '*.csv')
        else:
            files = glob.glob('training/tables/cr_' + docid + '*.csv')
        for file in files:
            f = file.split('\\')[-1].replace('_tables.csv', '').replace('cr_' + docid + '_', '')
            fs.append(f)
    else:
        docid, file = docid.split('_')
        fs = [file]

    for file in fs:
        try:
            bhtables = get_tables(docid, bh=bh, report_num=file, training=training)
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
        if not bh:
            fname = bhcsv_all
        else:
            fname = bhcsv
        save_rows(fname, bh_data)


def get_table_docids(bh=False, training=True):
    docids = []
    if not bh:
        folder = 'tables'
    else:
        folder = 'bh_tables'
    if training:
        lines_docs = glob.glob('training/' + folder + '/*.csv')
    else:
        lines_docs = glob.glob('C:\\Users\\andraszeka\\OneDrive - ITP (Queensland Government)\\textract_result/' + folder + '/*.csv')

    for lines_doc in lines_docs:
        docid = lines_doc.split('\\')[-1].replace('_tables.csv', '').strip('cr_')
        docids.append(docid)
    return docids


def manage_data(fname):
    df = pd.read_csv(fname)
    df = df.drop_duplicates()
    df.to_csv(fname, index=False)


def extract_for_all_docids(training=True):
    init()
    docids = get_table_docids(training=training)
    for id in docids:
        extract_bh(id, bh=False, training=training)
        #extract_bh(id, bh=True)
    #manage_data(bhcsv)
    manage_data(bhcsv_all)


def extract_for_docid(docid):
    init()
    extract_bh(docid, bh=False)
    manage_data(bhcsv_all)


if __name__ == "__main__":
    #extract_for_all_docids(training=False)
    #manage_data(bhcsv)
    #manage_data(bhcsv_all)
    #docids = ['106092', '99356', '92099', '84329', '77290', '72095', '69365', '99419']
    #for id in docids:
    #    extract_for_docid(id)

    extract_for_docid('44603')