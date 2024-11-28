from datasets import load_dataset
from sentence_transformers import SentenceTransformer, models
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import datetime
import random
import numpy as np
import pandas as pd

class BertForSTS(torch.nn.Module):

    def __init__(self):
        super(BertForSTS, self).__init__()
        self.bert = models.Transformer('bert-base-uncased', max_seq_length=128)
        self.pooling_layer = models.Pooling(self.bert.get_word_embedding_dimension())
        self.sts_bert = SentenceTransformer(modules=[self.bert, self.pooling_layer])

    def forward(self, input_data):
        output = self.sts_bert(input_data)['sentence_embedding']
        return output
    
    class STSBDataset(torch.utils.data.Dataset):

    def __init__(self, dataset):

        # Normalize the similarity scores in the dataset
        similarity_scores = [i['labels'] for i in dataset]
        self.normalized_similarity_scores = [i/5.0 for i in similarity_scores]
        self.first_sentences = [i['name1'] for i in dataset]
        self.second_sentences = [i['name2'] for i in dataset]
        self.concatenated_sentences = [[str(x), str(y)] for x,y in zip(self.first_sentences, self.second_sentences)]

    def __len__(self):
        return len(self.concatenated_sentences)

    def get_batch_labels(self, idx):
        return torch.tensor(self.normalized_similarity_scores[idx])

    def get_batch_texts(self, idx):
        return tokenizer(self.concatenated_sentences[idx], padding='max_length', max_length=128, truncation=True, return_tensors="pt")

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y


def collate_fn(texts):
    input_ids = texts['input_ids']
    attention_masks = texts['attention_mask']
    features = [{'input_ids': input_id, 'attention_mask': attention_mask}
                for input_id, attention_mask in zip(input_ids, attention_masks)]
    return features

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
    
def predict_similarity_embedding_model(sentence_pair):
    """
    Predict similarity between a pair of sentences
    """
    test_input = tokenizer(sentence_pair, padding='max_length', max_length=128, truncation=True, return_tensors="pt").to(device)
    test_input['input_ids'] = test_input['input_ids']
    test_input['attention_mask'] = test_input['attention_mask']
    del test_input['token_type_ids']
    output = model(test_input)
    sim = torch.nn.functional.cosine_similarity(output[0], output[1], dim=0).item()
    return sim

import pickle
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
    
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)
    
    
## embedding model fine tuned     
KEYWORDS = [
    "traders","trading", "enterprise", "garments", "collection", "food", "clothes", 
    "glass", "fittings", "digital", "kirana", "medical", "agency","tex",
    "security", "systems", "badges", "hospitality", "jewellers",'lic','agent',
    "ready-made", "store", "hospital", "restaurant", "auto", "center", 
    "dairy", "home", "products", "services", "furniture", "hardware", 
    "pharmacy", "stationery", "treatments", "nutrition", "wellness", 
    "sweets", "resort", "kitchen", "clothing", "market",'workshop','agency','consumer','amale' 
    "poultry", "seeds", "pesticides", "sales", "cafe", "clinic", 'project'
    "supermart", "distributors", "automobiles", "electricity", 
    "electronics", "general", "provision", "fertilizers", "agriculture", 
    "beverages", "textiles", "plumbing", "supplies", "handicrafts", 
    "construction", "medical", "bakery", "tissue", "cleaning", 
    "appliances", "homecare", "kitchenware", "decor", "glass and fittings",
    "interiors", "shopping", "crafts", "tools", "wholesale", 
    "retail", "outlet", "merchants", "trade", "distribution", 
    "solutions", "innovation", "consultancy", "services", "equipment", 
    "manufacturing", "exports", "imports", "packaging", "network", 
    "consultants", "transport", "moving", "storage", "logistics", 
    "construction", "real estate",'distributor','wines','hardware',
    'plywood','company','craft','soda','station','mobile'
    "brokerage", "management", 'handloom','co.','tvs',
    "finance", "investment", "funding", "support", "technology", 
    "software", "applications", "digital marketing", "advertising", 
    "communication", "entertainment", "events", "tourism", "travel", 
    "transportation", "automotive", "services", "supply chain", 
    "fashion", "cosmetics", "beauty", "spa", "wellness", "glass and fitting",
    "personal care", "gifts", "custom", "specialty", "craftsmanship", 
    "fashions", "motors", "enterprises", "garment", "cloth centre", "mart", 
    "foods", "silk and readymade", "wool centre", "jewellery", "mill", 
    "farms", "farm", "electrical", "egg centre", "centre", 
    "vegetable and fruits", "vegetables", "fruits", "pvt", "pvt ltd", 
    "limited", "solutions", "energies", "photo", "studio", "works", 
    "associates", "medico", "agencies", "diagnosis", "cool drinks", 
    "drinks", "care", "liquor", "automobiles", "materials", "diagnostics", 
    "provision", "trader", "farms", "farm", "stations", "restaurant", 
    "creations", "travels", "hardware", "printers", "graphics", 
    "fertilisers", "house", "studio", "private", "appliances", "steels", 
    "shop", "metals", "international", "jwellers", "corporation", 
    "dresses", "industries", "electricals", "company", "lim", "colddrinks", 
    "electron", "medicines", "llc", "computers", "hotel", "spa", 
    "cosmetics", "telecom", "sarees", "petroleums", "bhandar",'store','stores', 
    "surgical", "wines", "constructions", "shoppy", "lab", "builders", 
    "footwear", "wear", "shoe", "repair", "ventures", "paint", "depot",'cake','chinies',
    "tent", "decorators", "communications", "pharmacy", "products",'textile','CERAMIC','Pharmaceuticals','stores','sons',
]

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from fuzzywuzzy import fuzz
import re
import numpy as np
import pickle
# from BB import BertForSTS

# threshold = 0.80
# threshold1 = 0.65



##------------------------------------Fuzzy Wuzzy Layer-------------------------------------------------------------------------------------------------------


# def calculate_fuzzy_similarity(name1, name2):
#     fuzzy_ratio = fuzz.ratio(name1, name2) / 100.0
#     fuzzy_partial_ratio = fuzz.partial_ratio(name1, name2) / 100.0
#     fuzzy_token_sort_ratio = fuzz.token_sort_ratio(name1, name2) / 100.0
#     fuzzy_token_set_ratio = fuzz.token_set_ratio(name1, name2) / 100.0

#     fuzzy_similarity = (fuzzy_ratio + fuzzy_partial_ratio + fuzzy_token_sort_ratio + fuzzy_token_set_ratio) / 4.0
#     return fuzzy_similarity
def calculate_fuzzy_similarity(name1, name2):
    # Convert non-string types (e.g., float, None) to string
    if not isinstance(name1, str):
        name1 = str(name1) if name1 is not None else ""
    if not isinstance(name2, str):
        name2 = str(name2) if name2 is not None else ""

    # Perform fuzzy matching calculations
    fuzzy_ratio = fuzz.ratio(name1, name2) / 100.0
    fuzzy_partial_ratio = fuzz.partial_ratio(name1, name2) / 100.0
    fuzzy_token_sort_ratio = fuzz.token_sort_ratio(name1, name2) / 100.0
    fuzzy_token_set_ratio = fuzz.token_set_ratio(name1, name2) / 100.0

    # Calculate average fuzzy similarity
    fuzzy_similarity = (fuzzy_ratio + fuzzy_partial_ratio + fuzzy_token_sort_ratio + fuzzy_token_set_ratio) / 4.0
    return fuzzy_similarity


#-----------------------------------------Data Preprocessing anf Framework--------------------------------------------------------------------------------------------------


from transformers import AutoModel, AutoTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import distance as levenshtein_distance
import jellyfish

SPECIAL_CHAR_DOT_REGEX = r"[.]"
SPECIAL_CHARS_REGEX = r"[-+.^:,_/\s]+" 
SALUTATION_REGEX = r"^(shree|shri|miss|smt|mrs|mr|ms|dr|master|hon|sir|madam|prof|capt|major|rev|fr|br)\s*"
PARENT_SPOUSE_NAME_REGEX = r"(?:\s*(?:s/o|d/o|w/o|so|do|wo|daughter of|son of|wife of|husband of)\s*)"
COMMON_MUSLIM_SALUTATIONS_MOHAMMAD_REGEX = r"\b(mohammad|mohammed|muhamed|mohd|mohamed|mohamad|muhamad|muhammad|muhammed|muhammet|mohamud|mohummad|mohummed|mouhamed|muhamaad|mohammod|mouhamad|mo|md|mahmood|mahmud|ahmad|ahmed|hameed|hamid|hammed|mahd|mahmod|mohd|mouhammed|mohamad|muhmood|mohhammed|muhmamed|mohmed|mohmat|muhmat|mu|m|shaikh|mo)\b"
LAST_NAMES_AGARWAL_VARIANTS_REGEX = r"\b(aggarwal|agrawal|agarwal|aggrawal|agarwalla|agarwal)\b"


def convert_to_lower(name):
    return name.lower()

def replace_adjacent_duplicates(value):
    if isinstance(value, str):
        return re.sub(r'(.)\1+', r'\1', value)
    return value

def replace_characters(name):
    replacements = {'e': 'i', 'j': 'z', 'v': 'w', 'q': 'k'}
    for old, new in replacements.items():
        name = name.replace(old, new)
    return name

def replace_bigrams(name):
    replacements = {'ph': 'f', 'gh': 'g', 'th': 't', 'kh': 'k', 'dh': 'd', 'ch': 'c', 'sh': 's', 'au': 'o',
                    'bh': 'b', 'ks': 'x', 'ck': 'k', 'ah': 'h', 'wh': 'w', 'wr': 'r'}
    for old, new in replacements.items():
        name = name.replace(old, new)
    return name

def remove_extra_spaces(name):
    return re.sub(r'\s+', ' ', name).strip()

def remove_consonant_a(name):
    consonants = 'bcdfghjklmnpqrstvwxyz'
    new_name = ''.join([name[i] for i in range(len(name)) if not (i > 0 and name[i] == 'a' and name[i - 1].lower() in consonants)])
    return new_name

def remove_special_characters(text):
    text = re.sub(SPECIAL_CHAR_DOT_REGEX, '', text)
    text = re.sub(SPECIAL_CHARS_REGEX, '', text)
    return text.strip()

def remove_salutations(text):
    return re.sub(SALUTATION_REGEX, '', text, flags=re.IGNORECASE).strip()

def remove_parent_spouse_name(text):
    return re.sub(r'\s*(?:s/o|d/o|w/o|so|do|wo|daughter of|son of|wife of|husband of|daughter|son|child of)\s*[\w\s,.]*$', '', text, flags=re.IGNORECASE).strip()

def remove_common_muslim_variations(text):
    return re.sub(COMMON_MUSLIM_SALUTATIONS_MOHAMMAD_REGEX, '', text, flags=re.IGNORECASE).strip()

def remove_agarwal_variants(text):
    return re.sub(LAST_NAMES_AGARWAL_VARIANTS_REGEX, '', text, flags=re.IGNORECASE).strip()

def remove_stop_words(text):
    stop_words = ['devi', 'dei', 'debi', 'kumar', 'kumaar', 'kumari', 'kumaari', 'kmr', 'kumr', 'bhai', 'bhau', 'bai', 'ben', 'singh', 'kaur', 'Md', 'Mohd', 'Mohammad', 'Mohamad']
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def preprocess(name):
    name = remove_salutations(name)
    name = remove_parent_spouse_name(name)
    name = remove_common_muslim_variations(name)
    name = remove_agarwal_variants(name)
    name = convert_to_lower(name)
    name = replace_adjacent_duplicates(name)
    name = replace_characters(name)
    name = replace_bigrams(name)
    name = remove_consonant_a(name)
    name = remove_special_characters(name)
    name = remove_extra_spaces(name)
    name = remove_stop_words(name)
    return name


import pickle
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
    
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)
    

def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings


# Similarity functions
def calculate_cosine_similarity(embedding1, embedding2):
    return cosine_similarity(embedding1, embedding2).item()

def calculate_levenshtein_similarity(name1, name2):
    lev_distance = levenshtein_distance(name1, name2)
    max_len = max(len(name1), len(name2))
    return (max_len - lev_distance) / max_len if max_len > 0 else 1.0

def calculate_phonetic_similarity(name1, name2):
    soundex1 = jellyfish.soundex(name1)
    soundex2 = jellyfish.soundex(name2)
    return jellyfish.jaro_winkler_similarity(soundex1, soundex2)

def calculate_jaccard_similarity(name1, name2):
    set1, set2 = set(name1), set(name2)
    intersection, union = set1.intersection(set2), set1.union(set2)
    return len(intersection) / len(union) if union else 1.0

##------------------------------------Calling Name_Match------------------------------------------------------------------------------------------------------
 

def name_match(name1, name2):
    name1_processed = preprocess(name1)
    name2_processed = preprocess(name2)

    embedding_similarity = predict_similarity_embedding_model([name1_processed, name2_processed])

    levenshtein_similarity = calculate_levenshtein_similarity(name1_processed, name2_processed)
    phonetic_similarity = calculate_phonetic_similarity(name1_processed, name2_processed)
    jaccard_similarity = calculate_jaccard_similarity(name1_processed, name2_processed)

    fuzzy_ratio = fuzz.ratio(name1_processed, name2_processed) / 100.0
    fuzzy_partial_ratio = fuzz.partial_ratio(name1_processed, name2_processed) / 100.0
    fuzzy_token_sort_ratio = fuzz.token_sort_ratio(name1_processed, name2_processed) / 100.0
    fuzzy_token_set_ratio = fuzz.token_set_ratio(name1_processed, name2_processed) / 100.0

    fuzzy_similarity = (fuzzy_ratio + fuzzy_partial_ratio + fuzzy_token_sort_ratio + fuzzy_token_set_ratio) / 4.0
    
    
#     return {
#         "embedding_similarity": embedding_similarity,
#         "levenshtein_similarity": levenshtein_similarity,
#         "phonetic_similarity": phonetic_similarity,
#         "jaccard_similarity": jaccard_similarity,
#         "fuzzy_similarity": fuzzy_similarity
#     }
  

    final_score = (
        embedding_similarity * 0.121630 +
        levenshtein_similarity * 0.205104 +
        phonetic_similarity * 0.022287 +
        jaccard_similarity * 0.262775 +
        fuzzy_similarity * 0.388205
    )
    return final_score

    if final_score >= 0.65:
        return check_keywords_in_names(name1, name2, KEYWORDS)  # Check keywords after threshold pass

    return 0   


##------------------------------------Fuzzy With Data Preprocessing and keyword matching-------------------------------------------------------------------------------------------------------

SPECIAL_CHAR_DOT_REGEX = r"[.]"
SPECIAL_CHARS_REGEX = r"[-+.^:,_/\s]+" 
SALUTATION_REGEX = r"^(shree|shri|miss|smt|mrs|mr|ms|dr|master|hon|sir|madam|prof|capt|major|rev|fr|br)\s*"
PARENT_SPOUSE_NAME_REGEX = r"(?:\s*(?:s/o|d/o|w/o|so|do|wo|daughter of|son of|wife of|husband of)\s*)"
COMMON_MUSLIM_SALUTATIONS_MOHAMMAD_REGEX = r"\b(mohammad|mohammed|muhamed|mohd|mohamed|mohamad|muhamad|muhammad|muhammed|muhammet|mohamud|mohummad|mohummed|mouhamed|muhamaad|mohammod|mouhamad|mo|md|mahmood|mahmud|ahmad|ahmed|hameed|hamid|hammed|mahd|mahmod|mohd|mouhammed|mohamad|muhmood|mohhammed|muhmamed|mohmed|mohmat|muhmat|mu|m|shaikh|mo)\b"
LAST_NAMES_AGARWAL_VARIANTS_REGEX = r"\b(aggarwal|agrawal|agarwal|aggrawal|agarwalla|agarwal)\b"


def convert_to_lower(name):
    return name.lower()

def remove_extra_spaces(name):
    return re.sub(r'\s+', ' ', name).strip()

def remove_common_muslim_variations(text):
    return re.sub(COMMON_MUSLIM_SALUTATIONS_MOHAMMAD_REGEX, '', text, flags=re.IGNORECASE).strip()

def remove_agarwal_variants(text):
    return re.sub(LAST_NAMES_AGARWAL_VARIANTS_REGEX, '', text, flags=re.IGNORECASE).strip()

def remove_stop_words(text):
    stop_words = ['devi', 'dei', 'debi', 'kumar', 'kumaar', 'kumari', 'kumaari', 'kmr', 'kumr', 'bhai', 'bhau', 'bai', 'ben', 'singh', 'kaur', 'Md', 'Mohd', 'Mohammad', 'Mohamad','alam','shekh']
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)


def preprocess_FuzzyWuzzy(name):
    """Process a name through all the defined normalization steps."""
    name = convert_to_lower(name)
    name = remove_stop_words(name)
    name = remove_common_muslim_variations(name)
    name = remove_agarwal_variants(name)
    return name

def calculate_fuzzy_similarity_processed(name1, name2):
    name1 = preprocess_FuzzyWuzzy(name1)
    name2 = preprocess_FuzzyWuzzy(name2)
    
    fuzzy_ratio = fuzz.ratio(name1, name2) / 100.0
    fuzzy_partial_ratio = fuzz.partial_ratio(name1, name2) / 100.0
    fuzzy_token_sort_ratio = fuzz.token_sort_ratio(name1, name2) / 100.0
    fuzzy_token_set_ratio = fuzz.token_set_ratio(name1, name2) / 100.0

    fuzzy_similarity = (fuzzy_ratio + fuzzy_partial_ratio + fuzzy_token_sort_ratio + fuzzy_token_set_ratio) / 4.0
    return fuzzy_similarity


def check_keywords_in_names(name1, name2):
    """Check for presence of keywords in either or both names."""
    found_in_name1 = any(keyword in name1.lower() for keyword in KEYWORDS)
    found_in_name2 = any(keyword in name2.lower() for keyword in KEYWORDS)
    
    if found_in_name1 and found_in_name2:
        return 1  
    elif found_in_name1 or found_in_name2:
        return 0  
    return 1


def process_false_cases(row):
    name1 = row['name1']
    name2 = row['name2']

    fuzzy_SS = calculate_fuzzy_similarity_processed(name1, name2)

    fuzzy_flag = True if fuzzy_SS >= 0.75 else False

    if fuzzy_flag:
        fuzzy_flag = check_keywords_in_names(name1, name2)

    return True if fuzzy_flag else False

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def calculate_metrics(df):
    y_true = df['labels']  # True labels
    y_pred = df['Prediction']  # Predicted labels
    y_scores = df['Prediction'].astype(float)  # Convert boolean to float for AUC calculation
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc_score": roc_auc_score(y_true, y_scores),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }
    
    return metrics

def process_name_matching(file_path):
    # Load data for first layer
    df_layer1 = pd.read_csv(file_path)
    threshold = 0.75
    threshold1 = 0.65

    # First Layer
    df_layer1["First_Layer_Score"] = df_layer1.apply(lambda x: calculate_fuzzy_similarity(x['name1'], x['name2']), axis=1)
    df_layer1["First_Layer_Pass"] = df_layer1["First_Layer_Score"] >= threshold
    df_layer1["Prediction"] = df_layer1["First_Layer_Pass"]
    
    # Save first layer result
    df_layer1.to_csv("first_audit.csv", index=False)
    first_layer_metrics = calculate_metrics(df_layer1)

    # Second Layer
    df_layer2 = df_layer1[df_layer1["Prediction"] == False]  # Filter rows where Prediction == False
    df_layer2["Second_Layer_Score"] = df_layer2.apply(lambda x: name_match(x['name1'], x['name2']), axis=1)
    df_layer2["Second_Layer_Pass"] = df_layer2["Second_Layer_Score"] >= threshold1
    df_layer2["Prediction"] = df_layer2["Second_Layer_Pass"]
    
    # Save second layer result
    df_layer2.to_csv("second_audit.csv", index=False)
    second_layer_metrics = calculate_metrics(df_layer2)

    # Third Layer
    df_layer3 = df_layer2[df_layer2["Prediction"] == False]  # Filter rows where Prediction == False
    df_layer3["Third_Layer_Pass"] = df_layer3.apply(process_false_cases, axis=1)
    df_layer3["Prediction"] = df_layer3["Third_Layer_Pass"]
    
    # Save third layer result
    df_layer3.to_csv("third_audit.csv", index=False)
    third_layer_metrics = calculate_metrics(df_layer3)
    
    # Combine results
    df_combined = pd.concat([
        df_layer1[df_layer1["Prediction"]],
        df_layer2[df_layer2["Prediction"]],
        df_layer3[df_layer3["Prediction"]]
    ])

    # Final output
    df_layer1["Prediction"] = False
    df_layer1.loc[df_layer1[df_layer1["Prediction"]].index, "Prediction"] = True
    df_layer1.loc[df_layer2[df_layer2["Prediction"]].index, "Prediction"] = True
    df_layer1.loc[df_layer3[df_layer3["Prediction"]].index, "Prediction"] = True

    return first_layer_metrics, second_layer_metrics, third_layer_metrics, df_combined

if __name__ == "__main__":
    first_metrics, second_metrics, third_metrics, combined_data = process_name_matching("new_data.csv")

    print("\nConfusion Matrix for Each Layer:")

    # First Layer Confusion Matrix
    print("\nFirst Layer Confusion Matrix:")
    first_cm = first_metrics["confusion_matrix"]
    print(pd.DataFrame(first_cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]))
    print(f"Accuracy: {first_metrics['accuracy']}")
    print(f"Precision: {first_metrics['precision']}")
    print(f"Recall: {first_metrics['recall']}")
    print(f"F1 Score: {first_metrics['f1_score']}")
    print(f"AUC: {first_metrics['roc_auc_score']}")

    # Second Layer Confusion Matrix
    print("\nSecond Layer Confusion Matrix:")
    second_cm = second_metrics["confusion_matrix"]
    print(pd.DataFrame(second_cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]))
    print(f"Accuracy: {second_metrics['accuracy']}")
    print(f"Precision: {second_metrics['precision']}")
    print(f"Recall: {second_metrics['recall']}")
    print(f"F1 Score: {second_metrics['f1_score']}")
    print(f"AUC: {second_metrics['roc_auc_score']}")

    # Third Layer Confusion Matrix
    print("\nThird Layer Confusion Matrix:")
    third_cm = third_metrics["confusion_matrix"]
    print(pd.DataFrame(third_cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"]))
    print(f"Accuracy: {third_metrics['accuracy']}")
    print(f"Precision: {third_metrics['precision']}")
    print(f"Recall: {third_metrics['recall']}")
    print(f"F1 Score: {third_metrics['f1_score']}")
    print(f"AUC: {third_metrics['roc_auc_score']}")
