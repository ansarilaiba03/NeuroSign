import csv

input_file = "model/keypoint_classifier/keypoint.csv"
output_file = "model/keypoint_classifier/cleaned.csv"

with open(input_file, 'r') as f, open(output_file, 'w', newline='') as out:
    reader = csv.reader(f)
    writer = csv.writer(out)

    for row in reader:
        if len(row) == 85:  # correct rows only
            writer.writerow(row)

print("Done cleaning dataset")