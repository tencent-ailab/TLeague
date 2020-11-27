#!/usr/bin/env python
# -*- coding:utf-8 -*-

import xlrd
import csv
import sys


def csv_from_excel(xlsx_path_list, csv_path, sheet_name='Sheet1'):
  is_py3 = True if sys.version_info.major > 2 else False
  sheet_list = []
  for xlsx_path in xlsx_path_list:
    wb = xlrd.open_workbook(xlsx_path)
    sheet_list.append(wb.sheet_by_name(sheet_name))
  with open(csv_path, 'w') as csv_file:
    wr = csv.writer(csv_file, delimiter=',', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
    starting_row = 0
    for sh in sheet_list:
      for rownum in range(starting_row, sh.nrows):
        if is_py3:
          row = [each for each in sh.row_values(rownum)]
        else:
          row = [each.encode('utf-8') if type(each) is unicode else each
                 for each in sh.row_values(rownum)]
        wr.writerow(row)
      starting_row = 1 # Skip header for the rest of xlsx files


if __name__ == '__main__':
  '''
  Usage: xlsx2csv [XLSX_FILES]
  '''
  import sys
  csv_path = '/dev/stdout'
  sheet_name = 'Sheet1'
  if len(sys.argv) == 1:
    print('''
    Combine/concatenate multiple *.xlsx and convert them to a single csv.
    Usage: %s [XLSX_FILES]
    ''' % sys.argv[0])
  else:
    csv_from_excel(sys.argv[1:], csv_path, sheet_name)
