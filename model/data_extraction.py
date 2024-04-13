#minimizing the running time
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

#extract_lines('./IMDB_Dataset.csv', 'newbooks.csv', 10000)