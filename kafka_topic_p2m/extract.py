#extracting 100 lines from books ratings dataset to minimize running time and facilitate coding

import pandas as pd
import csv

def extract_lines(input_file, output_file, num_lines):
    with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        header = next(reader, None)
        if header:
            writer.writerow(header)

        for _ in range(num_lines):
            row = next(reader, None)
            if row is not None:
                writer.writerow(row)
            else:
                break

extract_lines('datasets/Books_rating.csv', 'datasets/BR.csv', 100)
df= pd.read_csv('datasets/BR.csv')