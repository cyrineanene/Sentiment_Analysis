#minimizing the running time
import csv
import pandas as pd

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
        return pd.read_csv(output_file)

extract_lines('./datasets/books_rating.csv','./BR4.csv',70000)