import csv

with open('D:/Mess/Deeplearning/data_handle_all/181112-01_Z&R_Normal/Normal_0/temp_0_.csv', mode='r') as csv_file:
    temp_0 = csv.reader(csv_file)
    print(temp_0)
    line_count = 0
    for row in temp_0:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
    print(float(row["temp1"]))
    '''
        print(f'\t{row["name"]} works in the {row["department"]} department, and was born in {row["birthday month"]}.')
        line_count += 1
    print(f'Processed {line_count} lines.')
    '''


