import csv
import sys

def extract_first_column(input_file, output_file):
    try:
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            reader = csv.reader(infile)
            # Skip the header if it exists
            next(reader, None)

            for index, row in enumerate(reader):
                outfile.write(f"A[{index}] = {row[0]};\n")

        print(f"Data successfully written to {output_file}")

    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <input_file> <output_file>")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        extract_first_column(input_file, output_file)