import os
import argparse
import hashlib
from collections import defaultdict

def read_file_lines(file_path):
    """Read lines from a file and return as a list."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def find_duplicates_in_file(file_path):
    """Find duplicates within a single file."""
    lines = read_file_lines(file_path)
    
    unique_lines = set()
    duplicates = []
    
    for i, line in enumerate(lines):
        if line in unique_lines:
            duplicates.append((line, i+1))
        else:
            unique_lines.add(line)
    
    return duplicates, len(lines)

def find_duplicates_across_files(file_paths):
    """Find duplicates across multiple files."""
    line_to_files = defaultdict(list)
    file_line_counts = {}
    
    for file_path in file_paths:
        lines = read_file_lines(file_path)
        file_line_counts[file_path] = len(lines)
        
        for i, line in enumerate(lines):
            line_to_files[line].append((file_path, i+1))
    
    duplicates = {line: occurrences for line, occurrences in line_to_files.items() 
                  if len(occurrences) > 1}
    
    return duplicates, file_line_counts

def find_duplicates_csv_column(file_path, column_index=0, delimiter=','):
    """Find duplicates within a specific column of a CSV file."""
    values = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            parts = line.strip().split(delimiter)
            if len(parts) > column_index:
                values.append((parts[column_index], i+1))
    
    value_to_lines = defaultdict(list)
    for value, line_num in values:
        value_to_lines[value].append(line_num)
    
    duplicates = {value: line_nums for value, line_nums in value_to_lines.items() 
                 if len(line_nums) > 1}
    
    return duplicates, len(values)

def find_duplicates_by_hash(file_paths):
    """Find duplicate files by content hash."""
    hash_to_files = defaultdict(list)
    
    for file_path in file_paths:
        if os.path.isfile(file_path):
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
                hash_to_files[file_hash].append(file_path)
    
    duplicates = {hash_val: files for hash_val, files in hash_to_files.items() 
                 if len(files) > 1}
    
    return duplicates

def find_duplicate_aligned_pairs(file_path, delimiter='\t'):
    """Find duplicate source-target pairs in an aligned_pairs file.
    Returns a dictionary of duplicate pairs with their line numbers."""
    pair_to_lines = defaultdict(list)
    total_pairs = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # Skip header
        header = f.readline()
        
        for i, line in enumerate(f, start=2):  # Start at 2 because we skipped header
            parts = line.strip().split(delimiter)
            if len(parts) >= 3:  # Expecting line_num, source, target
                # Create a unique key for the source-target pair
                source_target_pair = (parts[1], parts[2])
                pair_to_lines[source_target_pair].append(i)
                total_pairs += 1
    
    # Filter to keep only duplicate pairs
    duplicates = {pair: line_nums for pair, line_nums in pair_to_lines.items() 
                 if len(line_nums) > 1}
    
    return duplicates, total_pairs

def remove_duplicate_aligned_pairs(file_path, delimiter='\t', backup=True):
    """Remove duplicate source-target pairs from an aligned_pairs file.
    Preserves the header row and the first occurrence of each pair.
    Returns tuple of (removed_count, total_rows)."""
    # Read all lines with their original newlines
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    
    # Create backup if requested
    if backup:
        backup_path = file_path + '.bak'
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
    
    # Keep header row
    header = lines[0]
    data_lines = lines[1:]
    
    # Keep track of seen pairs while preserving order
    seen_pairs = set()
    unique_lines = [header]
    
    for line in data_lines:
        parts = line.strip().split(delimiter)
        if len(parts) >= 3:  # Expecting line_num, source, target
            source_target_pair = (parts[1], parts[2])
            if source_target_pair not in seen_pairs:
                seen_pairs.add(source_target_pair)
                unique_lines.append(line)
    
    # Write the unique lines back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(unique_lines)
    
    removed_count = total_lines - len(unique_lines)
    return removed_count, total_lines

def remove_duplicates_from_file(file_path, backup=True):
    """Remove duplicate lines from a file while preserving order.
    Returns tuple of (removed_count, total_lines)."""
    # Read all lines with their original newlines
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    
    # Create backup if requested
    if backup:
        backup_path = file_path + '.bak'
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
    
    # Keep track of seen lines while preserving order
    seen = set()
    unique_lines = []
    
    for line in lines:
        line_content = line.strip()
        if line_content not in seen:
            seen.add(line_content)
            unique_lines.append(line)
    
    # Write the unique lines back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(unique_lines)
    
    removed_count = total_lines - len(unique_lines)
    return removed_count, total_lines

def remove_duplicates_csv_column(file_path, column_index=0, delimiter=',', backup=True):
    """Remove rows with duplicate values in a specific column of a CSV file.
    Preserves header row. Returns tuple of (removed_count, total_rows)."""
    # Read all lines with their original newlines
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    
    # Create backup if requested
    if backup:
        backup_path = file_path + '.bak'
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
    
    # Keep header row if exists
    header = lines[0] if lines else None
    
    # Process data rows (everything after header)
    data_lines = lines[1:] if header else lines
    
    # Keep track of seen values while preserving order
    seen_values = set()
    unique_lines = [header] if header else []
    
    for line in data_lines:
        parts = line.strip().split(delimiter)
        if len(parts) > column_index:
            column_value = parts[column_index]
            if column_value not in seen_values:
                seen_values.add(column_value)
                unique_lines.append(line)
    
    # Write the unique lines back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(unique_lines)
    
    removed_count = total_lines - len(unique_lines)
    return removed_count, total_lines

def save_duplicates_to_file(output_file, duplicates_data, mode):
    """Save duplicate results to a file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        if mode == 'within':
            for file_path, (duplicates, total_lines) in duplicates_data.items():
                f.write(f"File: {file_path}\n")
                f.write(f"Total lines: {total_lines}\n")
                
                if duplicates:
                    f.write(f"Found {len(duplicates)} duplicate lines:\n")
                    for line, line_num in duplicates:
                        f.write(f"  Line {line_num}: {line}\n")
                else:
                    f.write("No duplicates found.\n")
                f.write("\n")
                
        elif mode == 'across':
            duplicates, file_line_counts = duplicates_data
            
            f.write(f"Checked {len(file_line_counts)} files:\n")
            for file_path, count in file_line_counts.items():
                f.write(f"  {file_path}: {count} lines\n")
            
            if duplicates:
                f.write(f"\nFound {len(duplicates)} duplicated content across files:\n")
                for line, occurrences in duplicates.items():
                    if len(line) > 50:
                        line_display = line[:47] + "..."
                    else:
                        line_display = line
                    f.write(f"\nContent: {line_display}\n")
                    f.write("  Appears in:\n")
                    for file_path, line_num in occurrences:
                        f.write(f"    {file_path}:{line_num}\n")
            else:
                f.write("\nNo duplicates found across files.\n")
                
        elif mode == 'csv':
            for file_path, (duplicates, total_values) in duplicates_data.items():
                f.write(f"File: {file_path}\n")
                f.write(f"Total values: {total_values}\n")
                
                if duplicates:
                    f.write(f"Found {len(duplicates)} duplicate values:\n")
                    for value, line_nums in duplicates.items():
                        f.write(f"  Value '{value}' appears at lines: {', '.join(map(str, line_nums))}\n")
                else:
                    f.write("No duplicates found.\n")
                f.write("\n")
        
        elif mode == 'aligned':
            duplicates, total_pairs = duplicates_data
            
            # Calculate total duplicates (total occurrences beyond the first one)
            total_duplicates_removed = sum(len(line_nums) - 1 for line_nums in duplicates.values())
            
            f.write(f"Total pairs analyzed: {total_pairs}\n")
            f.write(f"Found {len(duplicates)} unique duplicate source-target pairs\n")
            f.write(f"Total duplicate rows that would be removed: {total_duplicates_removed}\n\n")
            
            if duplicates:
                f.write(f"Duplicate source-target pairs:\n\n")
                for (source, target), line_nums in duplicates.items():
                    f.write(f"Source: {source}\n")
                    f.write(f"Target: {target}\n")
                    f.write(f"Appears at lines: {', '.join(map(str, line_nums))}\n")
                    f.write(f"Occurrences: {len(line_nums)}\n\n")
            else:
                f.write("No duplicate source-target pairs found.\n")
                
        elif mode == 'files':
            duplicate_files = duplicates_data
            
            if duplicate_files:
                f.write(f"Found {len(duplicate_files)} sets of duplicate files:\n")
                for hash_val, files in duplicate_files.items():
                    f.write(f"\nFiles with hash {hash_val}:\n")
                    for file_path in files:
                        f.write(f"  {file_path}\n")
            else:
                f.write("No duplicate files found.\n")

def main():
    parser = argparse.ArgumentParser(description='Find or remove duplicates in files')
    parser.add_argument('files', nargs='+', help='Files to check for duplicates')
    parser.add_argument('--mode', choices=['within', 'across', 'csv', 'files', 'aligned'], 
                        default='within', help='Mode of duplicate checking')
    parser.add_argument('--column', type=int, default=0, 
                        help='Column index for CSV mode (0-based)')
    parser.add_argument('--delimiter', default=',', 
                        help='Delimiter for CSV/TSV mode')
    parser.add_argument('--output', '-o', help='Output file to save results')
    parser.add_argument('--remove', action='store_true',
                        help='Remove duplicates from the original files')
    parser.add_argument('--no-backup', action='store_true',
                        help='Do not create backup files when removing duplicates')
    
    args = parser.parse_args()
    
    if args.mode == 'aligned':
        # Handle aligned pairs TSV files (source-target pairs)
        results = {}
        for file_path in args.files:
            delimiter = args.delimiter if args.delimiter != ',' else '\t'  # Default to tab for aligned mode
            
            if args.remove:
                # First find duplicates to get stats before removing
                duplicates, total_pairs = find_duplicate_aligned_pairs(file_path, delimiter)
                total_duplicates_to_remove = sum(len(line_nums) - 1 for line_nums in duplicates.values())
                
                # Now remove duplicates
                removed_count, total_lines = remove_duplicate_aligned_pairs(
                    file_path, delimiter, backup=not args.no_backup)
                
                print(f"\nFile: {file_path}")
                print(f"Found {len(duplicates)} unique duplicate source-target pairs")
                print(f"Removed {removed_count} rows with duplicate source-target pairs")
                print(f"Out of {total_lines} total rows")
                
                if args.output:
                    # Still save the detailed duplicate info to the output file
                    save_duplicates_to_file(args.output, (duplicates, total_pairs), 'aligned')
                    print(f"\nDetailed results saved to {args.output}")
                
                if not args.no_backup:
                    print(f"Original file backed up to {file_path}.bak")
            else:
                duplicates, total_pairs = find_duplicate_aligned_pairs(file_path, delimiter)
                total_duplicates_to_remove = sum(len(line_nums) - 1 for line_nums in duplicates.values())
                
                print(f"\nFile: {file_path}")
                print(f"Total pairs analyzed: {total_pairs}")
                
                if duplicates:
                    print(f"Found {len(duplicates)} unique duplicate source-target pairs")
                    print(f"Total duplicate rows that would be removed: {total_duplicates_to_remove}")
                    
                    pair_count = 0
                    for (source, target), line_nums in duplicates.items():
                        pair_count += 1
                        if pair_count <= 5:  # Only show first 5 pairs to avoid flooding console
                            print(f"\nSource: {source}")
                            print(f"Target: {target}")
                            print(f"Appears at lines: {', '.join(map(str, line_nums))}")
                            print(f"Occurrences: {len(line_nums)}")
                        
                    if pair_count > 5:
                        print(f"\n... and {pair_count - 5} more duplicate pairs")
                        print("Check the output file for full details.")
                else:
                    print("No duplicate source-target pairs found.")
                
                # Save results to output file if requested
                if args.output:
                    save_duplicates_to_file(args.output, (duplicates, total_pairs), 'aligned')
                    print(f"\nDetailed results saved to {args.output}")
    
    elif args.mode == 'within':
        # Check for duplicates within each file
        results = {}
        for file_path in args.files:
            if args.remove:
                removed_count, total_lines = remove_duplicates_from_file(
                    file_path, backup=not args.no_backup)
                print(f"\nFile: {file_path}")
                print(f"Removed {removed_count} duplicate lines out of {total_lines} total lines")
                if not args.no_backup:
                    print(f"Original file backed up to {file_path}.bak")
            else:
                duplicates, total_lines = find_duplicates_in_file(file_path)
                results[file_path] = (duplicates, total_lines)
                
                print(f"\nFile: {file_path}")
                print(f"Total lines: {total_lines}")
                
                if duplicates:
                    print(f"Found {len(duplicates)} duplicate lines:")
                    for line, line_num in duplicates:
                        print(f"  Line {line_num}: {line}")
                else:
                    print("No duplicates found.")
                
        if args.output and not args.remove:
            save_duplicates_to_file(args.output, results, args.mode)
            print(f"\nResults saved to {args.output}")
    
    elif args.mode == 'across':
        # Check for duplicates across files
        if args.remove:
            print("Cannot remove duplicates in 'across' mode. Use 'within' mode for each file instead.")
        else:
            results = find_duplicates_across_files(args.files)
            duplicates, file_line_counts = results
            
            print(f"\nChecked {len(args.files)} files:")
            for file_path, count in file_line_counts.items():
                print(f"  {file_path}: {count} lines")
            
            if duplicates:
                print(f"\nFound {len(duplicates)} duplicated content across files:")
                for line, occurrences in duplicates.items():
                    if len(line) > 50:
                        line_display = line[:47] + "..."
                    else:
                        line_display = line
                    print(f"\nContent: {line_display}")
                    print("  Appears in:")
                    for file_path, line_num in occurrences:
                        print(f"    {file_path}:{line_num}")
            else:
                print("\nNo duplicates found across files.")
                
            if args.output:
                save_duplicates_to_file(args.output, results, args.mode)
                print(f"\nResults saved to {args.output}")
    
    elif args.mode == 'csv':
        # Check for duplicates in CSV column
        results = {}
        for file_path in args.files:
            if args.remove:
                removed_count, total_lines = remove_duplicates_csv_column(
                    file_path, args.column, args.delimiter, backup=not args.no_backup)
                print(f"\nFile: {file_path}")
                print(f"Removed {removed_count} rows with duplicate values in column {args.column}")
                print(f"Out of {total_lines} total rows")
                if not args.no_backup:
                    print(f"Original file backed up to {file_path}.bak")
            else:
                duplicates, total_values = find_duplicates_csv_column(
                    file_path, args.column, args.delimiter)
                results[file_path] = (duplicates, total_values)
                
                print(f"\nFile: {file_path}")
                print(f"Total values in column {args.column}: {total_values}")
                
                if duplicates:
                    print(f"Found {len(duplicates)} duplicate values:")
                    for value, line_nums in duplicates.items():
                        print(f"  Value '{value}' appears at lines: {', '.join(map(str, line_nums))}")
                else:
                    print("No duplicates found.")
                
        if args.output and not args.remove:
            save_duplicates_to_file(args.output, results, args.mode)
            print(f"\nResults saved to {args.output}")
    
    elif args.mode == 'files':
        # Check for duplicate files
        if args.remove:
            print("Removing duplicate files is not supported. Please use manual removal.")
        else:
            duplicate_files = find_duplicates_by_hash(args.files)
            
            if duplicate_files:
                print(f"\nFound {len(duplicate_files)} sets of duplicate files:")
                for hash_val, files in duplicate_files.items():
                    print(f"\nFiles with hash {hash_val}:")
                    for file_path in files:
                        print(f"  {file_path}")
            else:
                print("\nNo duplicate files found.")
                
            if args.output:
                save_duplicates_to_file(args.output, duplicate_files, args.mode)
                print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main() 