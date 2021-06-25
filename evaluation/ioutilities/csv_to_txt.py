import csv
import sys

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: {} <input csv> <output txt>'.format(sys.argv[0]))
        exit(1)
    csv_file = sys.argv[1]
    txt_file = sys.argv[2]
    with open(txt_file, "w") as my_output_file:
        with open(csv_file, "r") as my_input_file:
            [ my_output_file.write(" ".join(row)+'\n') for row in csv.reader(my_input_file)]
        my_output_file.close()
