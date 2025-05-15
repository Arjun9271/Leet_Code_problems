import re
import difflib
import json

def cleaning_extracted_text(sentence):
   
    text = sentence
    # Remove literal backslashes and double slashes
    text = text.replace('\\', '')
    text = text.replace('//', '')
    # Normalize curly quotes and dashes
    for src, repl in [("\u201c", "'"), ("\u201d", "'"), ("\u2019", "'"), ("\u2013", "-")]:
        text = text.replace(src, repl)
    text = text.replace('“', "'").replace('”', "'")
    text = text.replace('"', '').replace("'", '')
    # Remove commas
    text = text.replace(',', '')
    # text = re.sub(r"\bthen available\b", 'then-available', text)
    text = re.sub(r"[\[\]\(\)]", '', text)
    text = re.sub(r"\s+", ' ', text).strip()
    text = text.rstrip('.;')
   
    text = re.sub(r"\bservice\s+now\b", "servicenow", text, flags=re.IGNORECASE)
    text = re.sub(r"\bservicenow['’]s\b", "servicenow", text, flags=re.IGNORECASE)
    text = re.sub(r"\bservicenow[-_]\b", "servicenow", text, flags=re.IGNORECASE)
    text = re.sub(r"\bservicenow\b", "servicenow", text, flags=re.IGNORECASE)
    return text.lower()


def sentence_splitting(text):
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip().rstrip('.!?') for p in parts if p]


def extract_modification(template: str, extracted: str) -> dict:
    """
    Compare cleaned and split sentences from template and extracted, reporting additions and missing tokens.
    """
   

    # Clean and split
    lst1 = sentence_splitting(cleaning_extracted_text(template))
    lst2 = sentence_splitting(cleaning_extracted_text(extracted))

    # Log cleaned lists
    print("Cleaned Template Sentences:", lst1)
    print("="*100)
    print("Cleaned Extracted Sentences:", lst2)
    print("="*100)

    result = {"Modified": True}
    matcher = difflib.SequenceMatcher(a=lst1, b=lst2)
    idx = 1
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for _ in range(i2 - i1):
                result[f"line_{idx}"] = {"Addition/Modification": [], "Missing": []}
                idx += 1
        else:
            # Handle replace, delete, insert uniformly
            maxlen = max(i2 - i1, j2 - j1)
            for k in range(maxlen):
                s1 = lst1[i1 + k] if i1 + k < i2 else ''
                s2 = lst2[j1 + k] if j1 + k < j2 else ''
                diff = difflib.ndiff(s1.split(), s2.split())
                adds, misses = [], []
                for d in diff:
                    if d.startswith('+ '): adds.append(d[2:])
                    elif d.startswith('- '): misses.append(d[2:])
                result[f"line_{idx}"] = {"Addition/Modification": adds, "Missing": misses}
                idx += 1
    return result






if __name__ == "__main__":
    template = """During the Subscription Term, Customer may purchase units of PROD15028 ServiceNow@ Business Stake holder ("Required Product") ordered at minimum quantity of 10 in each order at the subscription fee rate in accordance with the order quantity as set forth in the pricing matrix below, and the subscription fee for the units shall be prorated to the end of the Subscription Term; provided that (i) the Required Product continues to be made commercially available by ServiceNow and, if not, then the order shall be for ServiceNow's then-available subscription product that is substantially equivalent to the Required Product; and (ii) the pricing model for the Required Product continues to be made commercially available by ServiceNow at the time of the subsequent order. Customer agrees that the pricing for the additional units is limited to the additional units only and shall not affect units of the Subscription Service that are already purchased. Number of [Fulfiller Users] Monthly price per [Fulfiller User] [x]-[y] [$] [a]-[b] [$]"""

    extracted = """During the Subscription Term, Customer may purchase units of PROD15028 ServiceNow@ Business Stake holder ("'Required Product') ordered at minimum quantity of 10 in each order at the subscription fee rate In accordance with the order quantity as Is et forth in the pricing matrix below and the subscription fee for the units shall be prorated to the end of the Subscription Term; provided that i the Required Product continues to be made commercially available by Service Now and if not then the order shall be for ServiceNow's then available subscription product that is substantially equivalent to the Required Product; and [(ii) the pricing model for the Required Product continues to be made commercially available by Service Now at the time of the subsequent order . Customer agrees that the pricing for the additional units is limited to the additional units only and shall not affect units of the Subscription Service that are already purchased"""

    result = extract_modification(template, extracted)
    print(json.dumps(result, indent=4))
