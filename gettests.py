import argparse
import glob
import os
import re
import csv

#!/usr/bin/env python3

def find_transformed_in_file(path):
    results = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if 'Transformed values' in line:
            block = ''.join(lines[i:i+6])  # this line + next 5 lines (or fewer if EOF)
            m = re.search(r'\[(.*?)\]', block, re.S)
            if not m:
                continue
            inside = re.sub(r'\s+', '', m.group(1))  # remove whitespace
            if inside == '':
                continue
            parts = [p for p in inside.split(',') if p != '']
            results.append(parts)
    return results

def collect_input_files(patterns):
    files = []
    for p in patterns:
        # if p is a directory, add all files in it
        if os.path.isdir(p):
            for entry in os.listdir(p):
                files.append(os.path.join(p, entry))
        else:
            expanded = glob.glob(p)
            if expanded:
                files.extend(expanded)
            else:
                # treat as literal filename even if glob didn't match
                files.append(p)
    # remove duplicates, keep order
    seen = set()
    out = []
    for f in files:
        if f not in seen and os.path.isfile(f):
            seen.add(f)
            out.append(f)
    return out

def write_csv(rows, outpath):
    if not rows:
        print("No data found. No CSV written.")
        return
    maxlen = max(len(r) for r in rows)
    with open(outpath, 'w', newline='', encoding='utf-8') as csvf:
        writer = csv.writer(csvf)
        for r in rows:
            row = r + [''] * (maxlen - len(r))
            writer.writerow(row)
    print(f"Wrote {len(rows)} rows to {outpath}")

def trim_to_random(rows, n=120):
    import random
    if len(rows) >+ n:
        return random.sample(rows, n)
    print(f"Only {len(rows)} rows found.")
    raise ValueError("Not enough rows to trim to random sample.")

def main():
    parser = argparse.ArgumentParser(description="Extract 'Transformed values' blocks and save to CSV.")
    parser.add_argument('inputs', nargs='+', help='Files, glob patterns, or directories to scan (e.g. *.txt)')
    parser.add_argument('-o', '--output', default='out.csv', help='Output CSV file (default out.csv)')
    args = parser.parse_args()

    files = collect_input_files(args.inputs)
    if not files:
        print("No input files found.")
        return

    all_rows = []
    for f in files:
        print('Parsing file ', f)
        rows = find_transformed_in_file(f)
        if rows:
            print(f'  Found {len(rows)} blocks in {f}')
            all_rows.extend(rows)

    trimmed_rows = trim_to_random(all_rows)

    write_csv(trimmed_rows, args.output)

if __name__ == '__main__':
    main()