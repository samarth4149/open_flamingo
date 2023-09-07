import csv

metadata_file = ''
test_file = ''

id2name = {}
# create the mapping between id and class names
with open(metadata_file, 'r') as file:
    reader = csv.reader(file)
    # Loop through each row in the CSV
    for row in reader:
        id2name[row[0]] = row[1]

with open(test_file, 'r') as file:
    reader = csv.reader(file)
    # Loop through each row in the CSV
    image_ann = {}
    for row in reader:
        image_id = row[0]
        labels = image_ann.get(image_id, [])
        confidence = int(row[-1])
        if confidence:
            label_id = row[2]
            labels.append(id2name)




