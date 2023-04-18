def combine_lists(list_a, list_b):
    combined_list_a = []
    combined_list_b = []
    
    start = 0
    for i in range(len(list_b)):
        if i == len(list_b) - 1 or list_b[i] != list_b[i+1] or list_b[i] == "MISC":
            combined_list_a.append(" ".join(list_a[start:i+1]))
            combined_list_b.append(list_b[i])
            start = i+1
    
    return combined_list_a, combined_list_b
    
    
 def combine_lists(list_a, list_b):
    combined_list_a = []
    combined_list_b = []
    
    temp_a = ""
    temp_b = ""
    for i in range(len(list_b)):
        if list_b[i] == "MISC":
            if temp_a != "":
                combined_list_a.append(temp_a)
                combined_list_b.append(temp_b)
                temp_a = ""
                temp_b = ""
            combined_list_a.append(list_a[i])
            combined_list_b.append(list_b[i])
        else:
            if temp_b == "":
                temp_a = list_a[i]
                temp_b = list_b[i]
            elif temp_b == list_b[i]:
                temp_a += " " + list_a[i]
            else:
                combined_list_a.append(temp_a)
                combined_list_b.append(temp_b)
                temp_a = list_a[i]
                temp_b = list_b[i]
    
    if temp_a != "":
        combined_list_a.append(temp_a)
        combined_list_b.append(temp_b)
    
    return combined_list_a, combined_list_b

# Example usage:
a = ["hello", "My", "Name", "is", "Prabhat", "Kumar"]
b = ["MISC", "MISC", "MISC", "MISC", "PERSON", "PERSON"]
combined_a, combined_b = combine_lists(a, b)
print(combined_a)
print(combined_b)


import spacy

def get_tokenized_sentence_and_entity_list(sentence, spans):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(sentence)
    entity_dict = {}
    for span in spans:
        for i in range(span['start'], span['end']):
            entity_dict[i] = span['entity_type']
    tokenized_sentence = []
    entity_list = []
    for token in doc:
        tokenized_sentence.append(token.text)
        if token.i in entity_dict:
            entity_list.append(entity_dict[token.i])
        else:
            entity_list.append('misc')
    return tokenized_sentence, entity_list

