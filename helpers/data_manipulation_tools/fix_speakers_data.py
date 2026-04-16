import csv

input_file = "speakers_info.csv"
output_file = "cleaned_output.csv"

# This script enforces the rule that every speaker that represents the homan voice 
# is have to have one F/M/A value set to 1.0 and other to 0.0
# the speakers that are represent the silence are must have all values set to 0.0

multi_one_speakers = []
fixed_speakers = []

with open(input_file, newline='') as infile, open(output_file, "w", newline='') as outfile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)

    writer.writeheader()

    for row in reader:
        speaker = row["speaker_name"]

        values = {
            "femininity": float(row["femininity"]),
            "masculinity": float(row["masculinity"]),
            "atypicality": float(row["atypicality"])
        }

        val_list = list(values.values())
        ones_count = val_list.count(1.0)
        zeros_count = val_list.count(0.0)

        # Detect invalid values
        has_invalid = any(v not in (0.0, 1.0) for v in val_list)

        if ones_count > 1:
            # Leave unchanged, just report
            multi_one_speakers.append((speaker, val_list))

        elif has_invalid:
            # Fix: pick max value → set to 1.0, others to 0.0
            max_key = max(values, key=values.get)

            for key in values:
                values[key] = 1.0 if key == max_key else 0.0

            fixed_speakers.append(speaker)

        # Silence (all zeros) or already valid → keep as is

        # Write updated row
        row.update(values)
        writer.writerow(row)


print("\nSpeakers with multiple 1.0 values (unchanged):")
for speaker, vals in multi_one_speakers:
    print(f"{speaker}: {vals}")

print("\nSpeakers that were fixed:")
for speaker in fixed_speakers:
    print(speaker)

print(f"\nTotal fixed: {len(fixed_speakers)}")
