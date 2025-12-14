"""Add language code prefix to each line in a text file."""

with open("gloss_lang_code.txt", "r", encoding="utf-8") as file:
    text = file.readlines()

clean_text = ["tur " + line for line in text]

with open("gloss_lang_code.txt", "w", encoding="utf-8") as file:
    file.writelines(clean_text)

print("File processed successfully.")
