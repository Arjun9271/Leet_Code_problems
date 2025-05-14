import difflib
import json
from nltk.tokenize import sent_tokenize

def clean_text(text):
    replacements = {
        '\u201c': "'", '\u201d': "'", '\u2019': "'", '\u2013': "-",
        '"': "'",  # Normalize quotes
        '\\u201c': "'", '\\u201d': "'",  # Handle escaped unicode
        '  ': ' ',  # Replace double spaces
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.strip().strip("'").lower()

def normalize_sentences(text):
    return [s.strip().rstrip('.;,') for s in sent_tokenize(text)]

def word_diff(old_sentence, new_sentence):
    diff = difflib.ndiff(old_sentence.split(), new_sentence.split())
    return {
        'additions': [word[2:] for word in diff if word.startswith('+')],
        'missing': [word[2:] for word in diff if word.startswith('-')]
    }

def process_changes(template_sents, extracted_sents):
    matcher = difflib.SequenceMatcher(None, template_sents, extracted_sents)
    changes = []
    line_counter = 1

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            line_counter += (i2 - i1)
        elif tag == 'replace':
            for i, j in zip(range(i1, i2), range(j1, j2)):
                diff = word_diff(template_sents[i], extracted_sents[j])
                changes.append((
                    line_counter,
                    {'additions': diff['additions'], 'missing': diff['missing']}
                ))
                line_counter += 1
        elif tag == 'insert':
            for j in range(j1, j2):
                changes.append((
                    line_counter,
                    {'additions': extracted_sents[j].split(), 'missing': []}
                ))
                line_counter += 1
        elif tag == 'delete':
            for i in range(i1, i2):
                changes.append((
                    line_counter,
                    {'additions': [], 'missing': template_sents[i].split()}
                ))
                line_counter += 1

    return changes

def extract_modifications(template, extracted):
    template_clean = clean_text(template)
    extracted_clean = clean_text(extracted)
    
    template_sents = normalize_sentences(template_clean)
    extracted_sents = normalize_sentences(extracted_clean)
    
    changes = process_changes(template_sents, extracted_sents)
    
    if not changes:
        return {"Modified": False}

    result = {
        "Modified": True,
        "Changes": {
            f"line_{num}": {
                "Addition/Modification": diff['additions'],
                "Missing": diff['missing']
            } for num, diff in changes
        }
    }
    
    # Ensure proper line order
    result["Changes"] = dict(sorted(
        result["Changes"].items(),
        key=lambda x: int(x[0].split('_')[1])
    ))
    
    return result

# Example usage
template = """During the Subscription Term, Customer may purchase units of PROD15028 ServiceNow@ Business Stake holder (“Required Product”) ordered at minimum quantity of 10 in each order at the subscription fee rate in accordance with the order quantity as set forth in the pricing matrix below, and the subscription fee for the units shall be prorated to the end of the Subscription Term; provided that (i) the Required Product continues to be made commercially available by ServiceNow and, if not, then the order shall be for ServiceNow’s then-available subscription product that is substantially equivalent to the Required Product; and (ii) the pricing model for the Required Product continues to be made commercially available by ServiceNow at the time of the subsequent order. Customer agrees that the pricing for the additional units is limited to the additional units only and shall not affect units of the Subscription Service that are already purchased. Number of [Fulfiller Users] Monthly price per [Fulfiller User] [x]-[y] [$] [a]-[b] [$]"""

extracted = """During the Subscription Term, Customer may purchase units of PROD15028 ServiceNow@ Business Stake holder ("'Required Product') ordered at minimum quantity of 9 in each order at the subscription fee rate In accordance with the order quantity as Is et forth in the pricing matrix below and the subscription fee for the units shall be prorated to the end of the Subscription Term; provided that i the Required Product continues to be made commercially available by Service Now and if not then the order shall be for ServiceNow's then available subscription product that is substantially equivalent to the Required Product; and [(ii) the pricing model for the Required Product continues to be made commercially available by Service Now at the time of the subsequent order . Customer agrees that the pricing for the additional units is limited to the additional units only and shall not affect units of the Subscription Service that are already purchased"""

result = extract_modifications(template, extracted)
print(json.dumps(result, indent=2))


{
  "Modified": true,
  "Changes": {
    "line_1": {
      "Addition/Modification": [
        "(''required",
        "9",
        "is",
        "et",
        "below",
        "i",
        "service",
        "now",
        "and",
        "not",
        "then",
        "available",
        "[(ii)",
        "service",
        "now"
      ],
      "Missing": []
    },
    "line_3": {
      "Addition/Modification": [],
      "Missing": [
        "number",
        "of",
        "[fulfiller",
        "users]",
        "monthly",
        "price",
        "per",
        "[fulfiller",
        "user]",
        "[x]-[y]",
        "[$]",
        "[a]-[b]",
        "[$]"
      ]
    }
  }
}
