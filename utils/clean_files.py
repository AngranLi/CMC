"""Prepare raw CMR+TMS data files for analysis.

This script performs the following primary functions:

1. Reads an Excel spreadsheet containing the test/train matrix.
2. Uses filenames from TMS and CMR data to match files with test numbers.
3. Renames and moves files to a new directory in the correct format.
4. Uses file timestamps to pair TMS and CMR files, removing those files with no partner.

Notes:
To use this script, use should ensure that the test number column in the test matrix contains
only unique values.

Raises:
    Exception: When specified sheet in test matrix doesn't exist.
    Exception: When specified object columns don't exist.
    Exception: When specified test number column doesn't exist.
"""

import os
import shutil
import glob
import argparse
import re
import datetime as dt

import numpy as np
import pandas as pd


# Define names
# TEST_SHEET = 'Calibration Test'
TEST_SHEET = 'test_matrix'
# TEST_NUM_COL = 'Run number'
TEST_NUM_COL = 'ID'
OBJECT_COLS = [
    'Threat',
    'Non-threat 1',
    'Non-threat 2',
    'Non-threat 3',
    # 'Threat Objects',
    # 'Pocket Items',
    # 'Benign Other',
    # 'Bag',
    # 'A', 'B', 'C'
]

# Define temporal matching window
CMR_TO_TMS_MIN = 0
CMR_TO_TMS_MAX = 8

# Parse names
TEST_SHEET = TEST_SHEET.lower()
TEST_NUM_COL = TEST_NUM_COL.lower()
OBJECT_COLS = [s.lower() for s in OBJECT_COLS]


def clean_files(source, dest, matrix):

    # Read information from test matrix (excel file)
    test_matrix_file = pd.ExcelFile(matrix)
    for sheet_name in test_matrix_file.sheet_names:
        if sheet_name.lower().strip() == TEST_SHEET:
            test_sheet = sheet_name
            break
    else:
        raise Exception(f'Specified sheet name not found in {matrix}')

    test_matrix = test_matrix_file.parse(sheet_name=test_sheet)

    # Clean column names
    object_cols = []
    test_num_col = None
    for column in test_matrix.columns:
        if column.lower().strip() in OBJECT_COLS:
            object_cols.append(column)
        if column.lower().strip() == TEST_NUM_COL:
            test_num_col = column
    
    if len(object_cols) != len(OBJECT_COLS):
        print(object_cols)
        print(OBJECT_COLS)
        raise Exception(f'Not all specified object columns found in {matrix}')
    if test_num_col is None:
        raise Exception(f'Specified test number column not found in {matrix}')
        

    # Remove rows without a run number (i.e., heading rows)
    clean_nums = (
        test_matrix[test_num_col].astype(str).str.isdigit() |
        test_matrix[test_num_col].astype(str).str.strip('abcdefghi').str.isdigit()
    )
    test_matrix = test_matrix.loc[clean_nums]
    test_matrix[test_num_col] = test_matrix[test_num_col].astype(str).str.strip()

    # Drop unnecessary columns
    test_matrix = test_matrix[[test_num_col] + object_cols]

    # Construct file suffixes
    strip = re.compile(' |,|-|_|&')
    test_matrix['Suffix'] = 'test' + test_matrix[test_num_col]
    for column in object_cols:
        rows_to_add = test_matrix[column].notna()
        test_matrix['Suffix'].loc[rows_to_add] = (
            test_matrix['Suffix'].loc[rows_to_add] + '_' +
            test_matrix[column].loc[rows_to_add].replace(strip, '').str.lower()
        )
    
    # Set test number to index to enable easy seaching
    test_matrix.set_index(test_num_col, inplace=True)
    assert test_matrix.index.is_unique, "Test matrix run numbers are non-unique, exiting."
    
    # Get file names
    all_files = glob.glob(os.path.join(source, '**/*'), recursive=True)
    all_files = [f for f in all_files if os.path.isfile(f) and '.xlsx' not in f]
    tms_files = [f for f in all_files if '.pickle' not in f]
    cmr_files = [f for f in all_files if '.pickle' in f]

    os.makedirs(os.path.join(dest, 'tms-raw'), exist_ok=True)
    os.makedirs(os.path.join(dest, 'tms'), exist_ok=True)
    os.makedirs(os.path.join(dest, 'cmr'), exist_ok=True)
    missing_cmr_files = cmr_files
    for tms_file in tms_files:
        print('TMS file:', tms_file)
        # Get test number and subject
        clean_fname = os.path.basename(tms_file).lower().replace(' ', '_')
        test_timestamp, test_details = clean_fname.split('_test_')
        test_num = test_details.split('_')[0]
        test_subject = test_details.split('_')[1]

        # Get objects
        test_tags = test_matrix.loc[test_num]['Suffix']

        # Build TMS copy command
        tms_rename = os.path.join(f'{test_timestamp}-{test_subject}-{test_tags}')
        new_path = os.path.join(dest, 'tms-raw', tms_rename + ".txt")
        if not os.path.exists(new_path):
            shutil.copy(tms_file, new_path)

        # Make folder for CMR files
        os.makedirs(os.path.join(dest, 'cmr', tms_rename), exist_ok=True)

        # Build CMR copy commands
        cmr_file_matches = [f for f in cmr_files if f'test_{test_num}_' in os.path.basename(f).lower()]
        for f in cmr_file_matches:
            print('  CMR file:', f)
            new_path = os.path.join(
                dest,
                'cmr', 
                tms_rename, os.path.basename(f).split('_')[1].replace('-', '') + '.pickle'
            )
            if not os.path.exists(new_path):
                shutil.copy(f, new_path)
        
        missing_cmr_files = [f for f in missing_cmr_files if f not in cmr_file_matches]

    print('\nNon-matching CMR files:\n', '\n  '.join(missing_cmr_files), sep='')

    print('\nRunning split_csv.sh')
    os.system(f'bash ./utils/split_csv.sh {os.path.join(dest, "tms-raw")} {os.path.join(dest, "tms")}')

    # Loop through folders ensuring all files are in pairs
    print('\nChecking for non-paired CMR/TMS files and removing')
    folders = os.listdir(os.path.join(dest, 'tms'))
    for folder in folders:
        print(f'  Folder:', folder)
        tms_folder = os.path.join(dest, 'tms', folder)
        cmr_folder = os.path.join(dest, 'cmr', folder)
        tms_file_list = os.listdir(os.path.join(tms_folder))
        cmr_file_list = os.listdir(os.path.join(cmr_folder))

        tms_timestamps = np.array([os.path.splitext(f)[0] for f in tms_file_list])
        cmr_timestamps = np.array([os.path.splitext(f)[0] for f in cmr_file_list])

        tms_keep = []
        cmr_keep = []
        for i, tms_ts in enumerate(tms_timestamps):
            tms_ts_dt = dt.datetime(2000, 1, 1, int(tms_ts[:2]), int(tms_ts[2:4]), int(tms_ts[4:]))
            for j, cmr_ts in enumerate(cmr_timestamps):
                cmr_ts_dt = dt.datetime(2000, 1, 1, int(cmr_ts[:2]), int(cmr_ts[2:4]), int(cmr_ts[4:]))
                diff = (tms_ts_dt - cmr_ts_dt).seconds
                if (diff >= CMR_TO_TMS_MIN) and (diff <= CMR_TO_TMS_MAX):
                    tms_keep.append(i)
                    cmr_keep.append(j)
                    break

        for i, tms_file in enumerate(tms_file_list):
            if not i in tms_keep:
                rm_file = os.path.join(dest, 'tms', folder, tms_file)
                print(f'    Removing {rm_file}')
                os.remove(rm_file)

        for i, cmr_file in enumerate(cmr_file_list):
            if not i in cmr_keep:
                rm_file = os.path.join(dest, 'cmr', folder, cmr_file)
                print(f'    Removing {rm_file}')
                os.remove(rm_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rename and structure TMS+CMR raw data files.')
    parser.add_argument('source', type=str, help='Source directory')
    parser.add_argument('destination', type=str, help='Destination directory')
    parser.add_argument('matrix', type=str, help='Test matrix path')

    args = parser.parse_args()

    clean_files(args.source, args.destination, args.matrix)
