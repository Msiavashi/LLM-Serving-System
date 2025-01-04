import csv

def write_csv_header(filename, header):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

def write_queue_states(filename, iteration, queue_sizes):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        row = [iteration] + [size for _, size in queue_sizes]
        writer.writerow(row)
