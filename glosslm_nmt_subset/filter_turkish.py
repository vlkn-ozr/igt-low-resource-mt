"""Filter Turkish entries from imtvault source in GlossLM dataset."""

import csv

input_file = 'glosslm_subset_100k.csv'
output_file = 'glosslm_subset_filtered_100k.csv'
removed_entries_file = 'turkish_imtvault_entries_100k.csv'

with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8', newline='') as outfile, \
     open(removed_entries_file, 'w', encoding='utf-8', newline='') as removed_file:
    
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    removed_writer = csv.writer(removed_file)
    
    header = next(reader)
    writer.writerow(header)
    removed_writer.writerow(header)
    
    for row in reader:
        if len(row) >= 5:
            if row[3] == 'imtvault' and row[4] == 'Turkish':
                removed_writer.writerow(row)
            else:
                writer.writerow(row)
        else:
            writer.writerow(row)

print("Filtering complete.")
print(f"Filtered dataset saved to: {output_file}")
print(f"Removed Turkish entries saved to: {removed_entries_file}") 