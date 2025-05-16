import re
import difflib
import json
import nltk
# nltk.download('punkt_tab')  
from nltk.tokenize import sent_tokenize

"""
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
"""



def cleaning_extracted_text(sentence):
    text = str(json.dumps(sentence)).replace("\\u201c","'").replace("\\u201d","'").replace("\\u2019","'").replace("\\u2013", "'")
    text = text.replace('"',"'").replace(",","")
    text = text.strip("'")
    text = text.lower().strip('.')
    return text
    

   


def sentence_splitting(sentence):    
    
    return sent_tokenize(sentence)




def extract_modification(template, extracted):
    lst_1 = sentence_splitting(cleaning_extracted_text(template.lower()))
    lst_2 = sentence_splitting(cleaning_extracted_text(extracted.lower()))
    rest_dict = {}
    print("="*50)
    print(lst_1)
    print("="*50)
    print(lst_2)
    
    if len(lst_1) == len(lst_2):  
        for i in range(1,len(lst_1)+1):
            diff = difflib.ndiff(lst_1[i-1].split(), lst_2[i-1].split())
            missing_words,additional_words = [],[]
            for line in diff:  
                if line.startswith('+'):
                    additional_words.append(line[2:])
                elif line.startswith('-'):
                    missing_words.append(line[2:])
            dic = {"Modified": True, f"line_{i}":{
                'Addition/Modification': additional_words,
                'Missing': ' '.join(missing_words)
            }}
            rest_dict.update(dic)
        return rest_dict
    


    elif len(lst_1) != len(lst_2):
        for i in range(1,len(lst_1)+1):
            for j in range(1,len(lst_2)+1):
                missing_words,additional_words = [],[]
                if i == j:
                    diff = difflib.ndiff(lst_1[i-1].split(), lst_2[j-1].split())
                    for line in diff: 
                        if line.startswith('+'):
                            additional_words.append(line[2:])
                        elif line.startswith('-'):
                            missing_words.append(line[2:])
                    dic_1 = {"Modified": True, f"line_{i}":{
                        'Addition/Modification': additional_words,
                        'Missing': missing_words
                    }}
                    rest_dict.update(dic_1)
                    continue
                elif i >len(lst_2):
                    additional_words =[]
                    missing_words.append(lst_1[i-1])
                    dic_2 = {"Modified": True, f"line_{i}":{
                        'Addition/Modification': additional_words,
                        'Missing': missing_words
                    }}
                    rest_dict.update(dic_2)
                    continue
                elif j > len(lst_1):
                    additional_words.append(lst_2[j-1])
                    missing_words = []
                    dic = {"Modified": True, f"line_{j}":{
                        'Addition/Modification': additional_words,
                        'Missing': missing_words
                    }}
                    rest_dict.update(dic)
                    final_dict = sorted(rest_dict.keys())
                    rest_dict = {key: rest_dict[key] for key in final_dict if key in rest_dict}
        return rest_dict
    

template = """During the Subscription Term, Customer may purchase units of PROD15028 ServiceNow@ Business Stake holder (“Required Product”) ordered at minimum quantity of 10 in each order at the subscription fee rate in accordance with the order quantity as set forth in the pricing matrix below, and the subscription fee for the units shall be prorated to the end of the Subscription Term; provided that (i) the Required Product continues to be made commercially available by ServiceNow and, if not, then the order shall be for ServiceNow’s then-available subscription product that is substantially equivalent to the Required Product; and (ii) the pricing model for the Required Product continues to be made commercially available by ServiceNow at the time of the subsequent order. Customer agrees that the pricing for the additional units is limited to the additional units only and shall not affect units of the Subscription Service that are already purchased. Number of [Fulfiller Users] Monthly price per [Fulfiller User] [x]-[y] [$] [a]-[b] [$]"""

extracted = """During the Subscription Term, Customer may purchase units of PROD15028 ServiceNow@ Business Stake holder ("'Required Product') ordered at minimum quantity of 9 in each order at the subscription fee rate In accordance with the order quantity as Is et forth in the pricing matrix below and the subscription fee for the units shall be prorated to the end of the Subscription Term; provided that i the Required Product continues to be made commercially available by Service Now and if not then the order shall be for ServiceNow's then available subscription product that is substantially equivalent to the Required Product; and [(ii) the pricing model for the Required Product continues to be made commercially available by Service Now at the time of the subsequent order . Customer agrees that the pricing for the additional units is limited to the additional units only and shall not affect units of the Subscription Service that are already purchased"""


result = extract_modification(template, extracted)
print(json.dumps(result, indent=5))
