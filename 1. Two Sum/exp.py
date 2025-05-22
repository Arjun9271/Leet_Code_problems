import difflib
import re
import pandas as pd

def tokenize(s):
    return re.split(r'\s+', s)
def untokenize(ts):
    return ' '.join(ts)
        
def equalize(s1, s2):
    l1 = tokenize(s1)
    l2 = tokenize(s2)
    res1 = []
    res2 = []
    prev = difflib.Match(0,0,0)
    for match in difflib.SequenceMatcher(a=l1, b=l2).get_matching_blocks():
        if (prev.a + prev.size != match.a):
            for i in range(prev.a + prev.size, match.a):
                res2 += ['_' * len(l1[i])]
            res1 += l1[prev.a + prev.size:match.a]
        if (prev.b + prev.size != match.b):
            for i in range(prev.b + prev.size, match.b):
                res1 += ['_' * len(l2[i])]
            res2 += l2[prev.b + prev.size:match.b]
        res1 += l1[match.a:match.a+match.size]
        res2 += l2[match.b:match.b+match.size]
        prev = match
    return untokenize(res1), untokenize(res2)

def insert_newlines(string, every=64, window=10):
    result = []
    from_string = string
    while len(from_string) > 0:
        cut_off = every
        if len(from_string) > every:
            while (from_string[cut_off-1] != ' ') and (cut_off > (every-window)):
                cut_off -= 1
        else:
            cut_off = len(from_string)
        part = from_string[:cut_off]
        result += [part]
        from_string = from_string[cut_off:]
    return result

def show_comparison(s1_input, s2_input, width=40, margin=10, sidebyside=True, compact=False):
    s1, s2 = equalize(s1_input.replace('"',"'").replace("“", "'").replace("’", "'") ,s2_input.replace('"',"'").replace("“", "'").replace("’", "'"))
    if sidebyside:
        s1 = insert_newlines(s1, width, margin)
        s2 = insert_newlines(s2, width, margin)
        if compact:
            for i in range(0, len(s1)):
                lft = re.sub(' +', ' ', s1[i].replace('_', '')).ljust(width)
                rgt = re.sub(' +', ' ', s2[i].replace('_', '')).ljust(width) 
                print(lft + ' | ' + rgt + ' | ')        
        else:
            for i in range(0, len(s1)):
                lft = s1[i].ljust(width)
                rgt = s2[i].ljust(width)
                print(lft + ' | ' + rgt + ' | ')
    else:
        print(s1)
        print("-------------------------------------------------------------------------------------")
        print(s2)
    return s1,s2, s1_input, s2_input

def display_delta(new_s1, new_s2, s1_input, s2_input):
    print("Length check",len(new_s1),len(new_s2))
    flag=0
    final_list1=[]
    for i in range(0,len(new_s1)):
        if i==0:
            if new_s1[i]=="_":
                temp_start=i
                flag=1
        if i>0:
            if new_s1[i]=="_" and new_s1[i-1]==" " and flag==0:
                temp_start=i
                flag=1
            if i<len(new_s1)-1:
                if new_s1[i]=="_" and new_s1[i+1]==" " and new_s1[i+2]!="_" and flag==1:
                    temp_end=i
                    flag=2
            else:
                if new_s1[i]=="_" and flag==1:
                    temp_end=i
                    flag=2
            if flag==2:
                temp_list = []
                temp_list.append(temp_start)
                temp_list.append(temp_end)
                flag=0
                final_list1.append(temp_list)
    flag=0
    final_list2=[]
    for i in range(0,len(new_s2)):
        if i==0:
            if new_s2[i]=="_":
                temp_start=i
                flag=1
        if i>0:
            if new_s2[i]=="_" and new_s2[i-1]==" " and flag==0:
                temp_start=i
                flag=1
            if i<len(new_s2)-1:
                if new_s2[i]=="_" and new_s2[i+1]==" " and new_s2[i+2]!="_" and flag==1:
                    temp_end=i
                    flag=2
            else:
                if new_s2[i]=="_" and flag==1:
                    temp_end=i
                    flag=2
            if flag==2:
                temp_list = []
                temp_list.append(temp_start)
                temp_list.append(temp_end)
                flag=0
                final_list2.append(temp_list)

    if len(final_list1)!=len(final_list2):
        if len(final_list1)>len(final_list2):
            for i in range(0,len(final_list1)):
                if final_list1[i][0]-final_list2[i][1]==2:
                    pass
                else:
                    final_list2.insert(i,final_list1[i])
        else:
            for i in range(0,len(final_list2)):
                if final_list1[i][0]-final_list2[i][1]==2:
                    pass
                else:
                    final_list1.insert(i,final_list1[i])

    print(final_list1)
    print("----------------------------")
    print(final_list2)
    print("--------------------------")

    ph_dict = {}
    replacement_dict = {}
    for j, pair in enumerate(final_list1):
        template_match = new_s2[final_list1[j][0]:final_list1[j][1]+1]
        extracted_match = new_s1[final_list2[j][0]:final_list2[j][1]+1]
        reg_pattern = r"(.*?)(\[<<.*?>>])(.*?)(?=.\[<<)"
        ph_match = re.findall(reg_pattern,template_match+" [<<")
        if len(ph_match) > 0:
            for match in ph_match:
                ph_dict[match[1]] = extracted_match.replace(match[0], "").replace(match[2], "")
        temp1 = template_match
        temp2 = extracted_match
        for char in extracted_match:
            if char in temp1 and char in temp2:
                temp1 = temp1.replace(char, "", 1)
                temp2 = temp2.replace(char, "", 1)
        if min(len(temp1), len(temp2))== 0:
            if len(temp1) > len(temp2):
                for t in temp1:
                    if t.isalnum():
                        break
                else:
                    extracted_match = template_match
            if len(temp1) < len(temp2):
                for t in temp2:
                    if t.isalnum():
                        break
                else:
                    template_match = extracted_match
        # if len(new_s1[final_list2[j][0]:final_list2[j][1]+1])-len(new_s2[final_list1[j][0]:final_list1[j][1]+1])==1:
        #     print("Check")
        #     if new_s2[final_list1[j][0]:final_list1[j][1]+1]==new_s1[final_list2[j][0]:final_list2[j][1]]:
        #         print("------------",new_s1[final_list2[j][1]:final_list2[j][1]+1])
        #         new_s3=''.join([new_s3[:final_list1[j][1]+1],new_s1[final_list2[j][1]:final_list2[j][1]+1],new_s3[:final_list1[j][1]:]])
        print("============================")

        # print(new_s2[final_list2[j][0]:final_list2[j][1]+1])
        # print(new_s1[final_list2[j][0]:final_list2[j][1]+1])
        # print("*-----------------------------*")
        #new_s3=new_s3[:final_list1[j][0]]+"|"+new_s3[final_list1[j][0]:final_list1[j][1]+1]+"|"+new_s3[final_list1[j][1]+1:]
        #new_s3=new_s3.replace(new_s2[final_list2[j][0]:final_list2[j][1]+1],"*/*"+new_s1[final_list2[j][0]:final_list2[j][1]+1]+"*/*",1)
        
        # print(new_s2[final_list1[j][0]:final_list1[j][1]])
        # print(new_s1[final_list2[j][0]:final_list2[j][1]])

    print("---------------------------------")

s1111 = "Template: During the Subscription Term as set forth on this Order Form, Customer may purchase units of [<<PRODUCT>>] (“Required Product”) at the monthly net price of [<<NET PRICE>>] per [<<USER TYPE>>]. The purchase of additional units shall be pursuant to a mutually agreed Order Form and co-termed to the Term End Date set forth in this Order Form. Customer shall be entitled to the specified price, provided that (i) the Required Product continues to be made commercially available by ServiceNow and, if not, then the order shall be for ServiceNow’s then available subscription product that is substantially equivalent to the Required Product; and (ii) the pricing model for the Required Product continues to be made commercially available by ServiceNow at the time of the subsequent order. Customer agrees that the pricing for the additional units is limited to the additional units only and shall not affect units of the Subscription Service that are already purchased."
s2222 = "Extracted: During the Subscription Term as set forth on this Order Form, Customer may purchase units of Software Asset Management (Required Product') at the monthly net price of $1.10 per Computer at a minimum quantity of 10,000 Computers. The purchase of additional units shall be pursuant to a mutually agreed Order Form and co-termed to the Term End Date set forth in this Order Form. Customer shall be entitled to the specified price provided that i the Required Pra duct continues to be made commercially available by ServiceNow and if not then the order shall be for ServiceNow's then available subscription product that is substantially equivalent to the Required Product; and wii the pricing model for the Required Product continues to be made commercially available by ServiceNow at the time of the subsequent order Customer agrees that the pricing for the additional units is limited to the additional units only and shall not affect units of the Subscription Service that are already purchased"
s111 = "Template: Customer acknowledges that it previously procured subscriptions and access to the instance of its Subscription Service the 'Original Instance') through a reseller ('Original Reseller'). By directly purchasing additional subscriptions and/or users for the Original Instance from ServiceNow on this Order Form, Customer acknowledges that Customer may need to file warranty or service credit claims with both the Original Reseller and ServiceNow. ServiceNow will use commercially reasonable efforts to assist Customer in allocating the claims among the Original Reseller and ServiceNow."
s222 = "Extracted: Customer acknowledges that it previously procured subscriptions and access to the instance of its Subscription Service (the “Original Instance”) through a reseller (“Original Reseller”). By directly purchasing additional subscriptions and/or users for the Original Instance from ServiceNow on this Order Form, Customer acknowledges that Customer may need to file warranty or service credit claims with both the Original Reseller and ServiceNow. ServiceNow will use commercially reasonable efforts to assist Customer in allocating the claims among the Original Reseller and ServiceNow."
s11 = "Template: ServiceNow shall determined in its sole discretion a whether and when to develop release and apply any Update or Upgrade to Customer's instances of the Subscription Service; and be whether a particular release is an Update, Upgrade or new service offering that is available separately for purchase."
s222222= "Extracted: During the Subscription Term, if Customer’s instances are on the then-current Release Family or the immediately preceding Release Family, Customer may determine when to apply an Upgrade unless, in the reasonable judgment of ServiceNow an Upgrade is necessary to: (a) maintain the availability, security, or performance of the Subscription Service; (b) comply with Law; or (c) avoid infringement or misappropriation of third-party Intellectual Property Right. If at any time during the Subscription Term Customer’s instances are not on the then-current Release Family or the immediately preceding Release Family, ServiceNow shall apply an Upgrade to the Customer’s instances upon notice in accordance with the Upgrade and Update exhibit to the Agreement."
s111111="Template: During the Subscription Term, if Customer's instance(s) is on the then-current Release Family or the immediately preceding Release Family, Customer may determine when to apply an Upgrade unless, in the reasonable judgment of ServiceNow an Upgrade is necessary to: (a) maintain the availability, security, or performance of the Subscription Service; (b) comply with Law; or (c) avoid infringement or misappropriation of third-party Intellectual Property Right. If at any time during the Subscription Term Customerâ€™s instance(s) is not on the then-current Release Family or the immediately preceding Release Family, ServiceNow shall apply an Upgrade to the Customer's instance(s) upon notice in accordance with the Upgrade and Update exhibit to the Agreement."
s2211="Extracted: transfer Customer agrees to provide transitional services to the Divested Entity in connection with the sale or transfer of such Divested Entity, including transitional use of the Subscription Service for such Divested Entity, then ServiceNow agrees that upon prior written notice to ServiceNow describing the transaction in reasonable details Customer may continue to use the Subscription Service, solely during the Subscription Term, to process the data of the Divested Entity as provided here inform a period not to exceed twelve (12) months after the completion of any such sale or transfer ('Transition Period') at no additional charge (i.e., no charge other than fees otherwise due to ServiceNow hereunder as if the Divested Entity were a part of Customer), provided that Customer is and remains current on the payment of all fees due to ServiceNow hereunder During the Transition Period, Customer and its successors and assigns shall be responsible for such Divested Entity's compliance with the terms of this Agreement. After the Transition Period, the Divested Entity shall have no right to access or use the Subscription Service under this Agreement."
s11222="Template: If, during the Subscription Term, Customer sells or otherwise transfers ownership of the assets or equity of any entity that constitutes Customer (each a “Divested Entity”), and if, as part of such sale or transfer, Customer agrees to provide transitional use of the Subscription Service for such Divested Entity, then ServiceNow agrees that, upon sixty (60) days’ prior written notice to ServiceNow describing the transaction in reasonable detail, Customer may continue to use the Subscription Service, solely during the Subscription Term, to process the data of the Divested Entity as provided herein for a period not to exceed the lesser of the remaining Subscription Term (including any renewal Subscription Term thereto) or [<<MONTHS>>] after the completion of any such sale or transfer (“Transition Period”) at no additional charge (i.e., no charge other than fees otherwise due to ServiceNow hereunder as if the Divested Entity were a part of Customer), provided that Customer is and remains current on the payment of all fees due to ServiceNow hereunder. During the Transition Period, Customer and its successors and assigns shall be wholly responsible for the acts and omissions of the Divested Entity and the Divested Entity’s compliance with the terms of the Agreement. No Divested Entity shall have the right to take any legal action against ServiceNow under the Agreement or enforce any provision of the Agreement directly against ServiceNow. After the Transition Period, the Divested Entity shall have no right to access or use the Subscription Service under the Agreement. The foregoing right to provide transitional services shall not apply in the event that the Divested Entity is acquired by a ServiceNow customer as of the date of the acquisition."
s1="During the Subscription Term as set forth on this Order Form, Customer may purchase units of [<<PRODUCT>>] (“Required Product”) at the monthly net price of [<<NET PRICE>>] per [<<USER TYPE>>]. The purchase of additional units shall be pursuant to a mutually agreed Order Form and co-termed to the Term End Date set forth in this Order Form. Customer shall be entitled to the specified price, provided that (i) the Required Product continues to be made commercially available by ServiceNow and, if not, then the order shall be for ServiceNow’s then available subscription product that is substantially equivalent to the Required Product; and (ii) the pricing model for the Required Product continues to be made commercially available by ServiceNow at the time of the subsequent order. Customer agrees that the pricing for the additional units is limited to the additional units only and shall not affect units of the Subscription Service that are already purchased."
s2="During the Subscription Term as set forth on this Order Form, Customer may purchase units of SericeNow® Additional 1TB Storage ('Required Product') at the monthly net price of 1667.0 per Production or Non Production Instance. The purchase of additional units shall be pursuant o a mutually agreed Order Form and co-termed to the Term End Date set forth in this Order t Form. Customer shall be entitled to the specified price provided that i the Required Product continues to be made commercially available by ServiceNow and if not then the order shall be for ServiceNow's then available subscription product that's substantially equivalent to the Required Product; and wii the pricing model for the Required Product continues to be made commercially available by ServiceNow at the time of the subsequent order Customer agrees that the pricing for the additional units is limited to the additional units only and shall not affect units of the Subscription Service that are already purchased"
# print(s1)
# print(s2)
print()
# print('Full side-by-side')
# print('-------------------------------------------------------------------------------------')
# show_comparison(s1, s2, width=40, sidebyside=True, compact=False)
# print()
# print('Compact side-by-side')
# print('-------------------------------------------------------------------------------------')
# show_comparison(s1, s2, width=40, sidebyside=True, compact=True)
# print()
# print('Above-below comparison')
print('-------------------------------------------------------------------------------------')
new_s1,new_s2 = show_comparison(s1, s2, sidebyside=False)


display_delta(new_s1, new_s2)
