import re
import difflib
import json

def cleaning_extracted_text(text):
    text = text.replace("“", "'").replace("”", "'").replace("’", "'").replace("–", "-")
    text = text.replace("\\u201c", "'").replace("\\u201d", "'").replace("\\u2019", "'").replace("\\u2013", "-")
    text = text.replace('"', "'").replace(",", "")
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def sentence_splitting(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if s.strip()]

def get_word_diff(source, target):
    diff = list(difflib.ndiff(source.split(), target.split()))
    missing = [w[2:] for w in diff if w.startswith('- ')]
    added = [w[2:] for w in diff if w.startswith('+ ')]
    return missing, added

def align_sentences(sentences_1, sentences_2):
    sm = difflib.SequenceMatcher(None, sentences_1, sentences_2)
    return sm.get_opcodes()

def extract_modification(template, extracted):
    template_clean = cleaning_extracted_text(template)
    extracted_clean = cleaning_extracted_text(extracted)

    sents_template = sentence_splitting(template_clean)
    sents_extracted = sentence_splitting(extracted_clean)

    alignments = align_sentences(sents_template, sents_extracted)

    result = {}
    line_number = 1

    for tag, i1, i2, j1, j2 in alignments:
        if tag == 'equal':
            continue

        max_len = max(i2 - i1, j2 - j1)
        for k in range(max_len):
            temp_sent = sents_template[i1 + k] if i1 + k < i2 else ''
            ext_sent = sents_extracted[j1 + k] if j1 + k < j2 else ''

            missing, added = get_word_diff(temp_sent, ext_sent)

            result[f"line_{line_number}"] = {
                "Modified": True,
                "Original Sentence": temp_sent,
                "Compared Sentence": ext_sent,
                "Missing": missing,
                "Addition/Modification": added
            }
            line_number += 1

    return result
