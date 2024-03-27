import csv

def extract_lines(input_file, output_file, num_lines):
    with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Write header if the input file has one
        header = next(reader, None)
        if header:
            writer.writerow(header)

        # Extract specified number of lines
        for _ in range(num_lines):
            row = next(reader, None)
            if row is not None:
                writer.writerow(row)
            else:
                break

# Example usage: Extract 10 lines from input.csv to output.csv
extract_lines('Books_rating.csv', 'newbooks.csv', 70000)
