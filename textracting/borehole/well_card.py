## @file
# Reading text from a well card

import docx
import paths


if __name__ == "__main__":
    fname = paths.get_word_file('38276', '1', 'wondershare')
    doc = docx.Document(fname)

    for p in doc.paragraphs:
        print(p.text)

    #f = open(fname, 'r', encoding="utf8", )
    #doc = f.readlines()

    print()