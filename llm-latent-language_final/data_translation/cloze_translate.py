import pandas as pd
import transformers
import sys
import os
from dataclasses import dataclass
import json
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import argparse
# from llamawrapper import load_unemb_only, LlamaHelper


# target_lang = 'te'

os.environ["HF_TOKEN"] = "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"
prefix = "./data/langs/"


# input_lang = 'en'
# model_id = 'meta-llama/Llama-2-7b-hf'
# model_id = "google/gemma-2-9b-it"
# model_id = "CohereForAI/aya-23-8B"



# def format_messages(en_word):
#     examples_prompt = " English: mars - മലയാളം: ചൊവ്വ \n English: bottle - മലയാളം: കുപ്പി \n English: tongue - മലയാളം: നാവ് \n English: silver - മലയാളം: വെള്ളി \n "

    
#     new_prompt = f"\n English: {en_word} - മലയാളം: "
    
#     return  examples_prompt + new_prompt

# def format1_messages(en_word):
#     examples_prompt = " English: mars - हिन्दी: मंगल ग्रह \n English: bottle - हिन्दी: बोतल \n English: tongue - हिन्दी: जीभ \n English: silver - हिन्दी: चाँदी \n "

    
#     new_prompt = f"\n English: {en_word} - हिन्दी: "
    
#     return [
#     {"role": "user", "content":  examples_prompt +'Just translate the english word to hindi dont generate anythiing else' +new_prompt},
#     ]


def format_he_messages(blank_prompt_translation_masked):
    examples_prompt = ' English ->  "The ""___"" is seen as a tiny spot in the sky from earth . Answer: ""star""." - עברית  -> "ה""___"" נראה כנקודה קטנה בשמים מהארץ. תשובה: ""כוכב""." \n English -> "I have a new ""___"" . Answer: ""idea""." - עברית ->  "יש לי ""___"" חדש. תשובה: ""רעיון""." \n English -> "Loud ""___""  was played in the auditorium. Answer: ""music""." - עברית -> "בוצע ""___"" רועש באולם. תשובה: ""מוזיקה""." \n English -> "People love watching a ""___"" in the theater. Answer: ""movie""." - עברית -> "אנשים אוהבים לצפות ב""___"" בתיאטרון. תשובה: ""סרט""."  \n  English -> "India is the most populated ""___"" in the world . Answer: ""country""." - עברית -> "הודו היא ה""___"" המאוכלסת ביותר בעולם. תשובה: ""מדינה""."  \n '

    prompt_suggestion = "Just translate the English masked sentence and answer to Hebrew. Don't generate synonyms of the answer."
    new_prompt = f"\n English -> {blank_prompt_translation_masked} - עברית -> "
    
    return [
    {"role": "user", "content":  examples_prompt + prompt_suggestion + new_prompt},
    ]


def format_th_messages(blank_prompt_translation_masked):
    examples_prompt = ' English -> "The ""___"" is seen as a tiny spot in the sky from earth. Answer: ""star""." - ไทย -> ""___"" ถูกมองว่าเป็นจุดเล็ก ๆ บนท้องฟ้าจากโลก คำตอบ: ""ดาว""。\n English -> "I have a new ""___"" . Answer: ""idea""." - ไทย -> "ฉันมี ""___"" ใหม่ คำตอบ: ""ความคิด""。\n English -> "Loud ""___"" was played in the auditorium. Answer: ""music""." - ไทย -> "มีการเล่น ""___"" ดังในหอประชุม คำตอบ: ""ดนตรี""。\n English -> "People love watching a ""___"" in the theater. Answer: ""movie""." - ไทย -> "ผู้คนชอบดู ""___"" ในโรงภาพยนตร์ คำตอบ: ""ภาพยนตร์""。\n English -> "India is the most populated ""___"" in the world . Answer: ""country""." - ไทย -> "อินเดียเป็น ""___"" ที่มีประชากรมากที่สุดในโลก คำตอบ: ""ประเทศ""。\n '
    
    prompt_suggestion = "Just translate the English masked sentence and answer to Thai. Don't generate synonyms of the answer. Generate the answer in Thai."
    new_prompt = f"\n English -> {blank_prompt_translation_masked} - ไทย -> "

    return [
    {"role": "user", "content":  examples_prompt + prompt_suggestion + new_prompt},
    ]


def format_ka_messages(blank_prompt_translation_masked):
    examples_prompt = ' English -> "The ""___"" is seen as a tiny spot in the sky from earth. Answer: ""star""." - ქართული -> ""___"" დედამიწიდან ცაზე პატარა წერტილად ჩანს. პასუხი: ""ვარსკვლავი""." \n English -> "I have a new ""___"". Answer: ""idea""." - ქართული -> "მე მაქვს ახალი ""___"". პასუხი: ""იდეა""." \n English -> "Loud ""___"" was played in the auditorium. Answer: ""music""." - ქართული -> "აუდიტორიაში ხმამაღალი ""___"" აჟღერდა. პასუხი: ""მუსიკა""." \n English -> "People love watching a ""___"" in the theater. Answer: ""movie""." - ქართული -> "ხალხს თეატრში ""___"" ყურება უყვარს. პასუხი: ""ფილმი""." \n English -> "India is the most populated ""___"" in the world. Answer: ""country""." - ქართული -> "ინდოეთი მსოფლიოში ყველაზე მრავალრიცხოვანი ""___"" არის. პასუხი: ""ქვეყანა""." \n '

    prompt_suggestion = "Just translate the English masked sentence and answer to Georgian. Don't generate synonyms of the answer."
    new_prompt = f"\n English -> {blank_prompt_translation_masked} - ქართული -> "
    
    return [
    {"role": "user", "content":  examples_prompt + prompt_suggestion + new_prompt},
    ]

def format_am_messages(blank_prompt_translation_masked):
    examples_prompt = ' English -> "The ""___"" is seen as a tiny spot in the sky from earth. Answer: ""star""." - አማርኛ -> ""___"" ከምድር በሰማይ እንደ ትንሽ ነጥብ ይታያል። መልስ፡ ""ኮከብ""." \n English -> "I have a new ""___"". Answer: ""idea""." - አማርኛ -> "አዲስ ""___"" አለኝ። መልስ፡ ""ሃሳብ""." \n English -> "Loud ""___"" was played in the auditorium. Answer: ""music""." - አማርኛ -> "በአዳራሽ ውስጥ ""___"" በታላቅ ድምፅ ተነፈሰ። መልስ፡ ""ሙዚቃ""." \n English -> "People love watching a ""___"" in the theater. Answer: ""movie""." - አማርኛ -> "ሰዎች ""___"" በቲያትር ማየት ይወዳሉ። መልስ፡ ""ፊልም""." \n English -> "India is the most populated ""___"" in the world. Answer: ""country""." - አማርኛ -> "ኢንዲያ በዓለም ላይ በጣም ብዙ ሰዎች ያሉበት ""___"" ናት። መልስ፡ ""አገር""." \n '

    prompt_suggestion = "Just translate the English masked sentence and answer to Amharic. Don't generate synonyms of the answer."
    new_prompt = f"\n English -> {blank_prompt_translation_masked} - አማርኛ -> "

    return [
    {"role": "user", "content":  examples_prompt + prompt_suggestion + new_prompt},
    ]
    

def format_ta_messages(blank_prompt_translation_masked):
    examples_prompt =  ' English ->  "The ""___"" is seen as a tiny spot in the sky from earth . Answer: ""star""." - தமிழ் -> ""___"" பூமியிலிருந்து வானத்தில் ஒரு சிறிய புள்ளியாகக் காணப்படுகிறது. பதில்: ""நட்சத்திரம்""." \n English -> "I have a new ""___"" . Answer: ""idea""." - தமிழ் -> "என்னிடம் புதிய ""___"" உள்ளது. பதில்: ""யோசனை""." \n English -> "Loud ""___""  was played in the auditorium. Answer: ""music""." - தமிழ் -> "ஆடிட்டோரியத்தில் ""___" சத்தமாக ஒலித்தது. பதில்: ""இசை""." \n English -> "People love watching a ""___"" in the theater. Answer: ""movie""." - தமிழ் -> "மக்கள் தியேட்டரில் ""___"" பார்க்க விரும்புகிறார்கள். பதில்: ""திரைப்படம்""."  \n  English -> "India is the most populated ""___"" in the world . Answer: ""country""." - தமிழ் -> "உலகில் அதிக மக்கள்தொகை கொண்ட ""___"" இந்தியா. பதில்: ""நாடு""." \n '

    prompt_suggestion = "Just translate the english masked sentence and answer to Tamil. Dont generate synonyms of the answer."
    new_prompt = f"\n English -> {blank_prompt_translation_masked} - தமிழ் -> "
    
    return [
    {"role": "user", "content":  examples_prompt + prompt_suggestion + new_prompt},
    ]

def format_te_messages(blank_prompt_translation_masked):
    examples_prompt =  ' English ->  "The ""___"" is seen as a tiny spot in the sky from earth . Answer: ""star""." - తెలుగు -> ""___"" భూమి నుండి ఆకాశంలో ఒక చిన్న ప్రదేశంగా కనిపిస్తుంది. సమాధానం: ""నక్షత్రం""." \n English -> "I have a new ""___"" . Answer: ""idea""." - తెలుగు -> "నా దగ్గర కొత్త ""___"" ఉంది. సమాధానం: ""ఆలోచన""." \n English -> "Loud ""___""  was played in the auditorium. Answer: ""music""." - తెలుగు -> ఆడిటోరియంలో ""___"" బిగ్గరగా ప్లే చేయబడింది. సమాధానం: ""సంగీతం""." \n English -> "People love watching a ""___"" in the theater. Answer: ""movie""." - తెలుగు -> "ప్రజలు థియేటర్‌లో ""___""ని చూడటానికి ఇష్టపడతారు. సమాధానం: ""సినిమా""." \n  English -> "India is the most populated ""___"" in the world . Answer: ""country""." - తెలుగు -> "భారతదేశం ప్రపంచంలో అత్యధిక జనాభా కలిగిన ""___"". సమాధానం: ""దేశం""." \n '
    prompt_suggestion = "Just translate the english masked sentence and answer to Telugu. Dont generate synonyms of the answer."
    
    new_prompt = f"\n English -> {blank_prompt_translation_masked} - తెలుగు -> "
    
    return [
    {"role": "user", "content":  examples_prompt + prompt_suggestion + new_prompt},
    ]

def format_gu_messages(blank_prompt_translation_masked):
    examples_prompt =  ' English ->  "The ""___"" is seen as a tiny spot in the sky from earth . Answer: ""star""." - ગુજરાતી -> ""___"" ને પૃથ્વી પરથી આકાશમાં એક નાનકડા સ્થળ તરીકે જોવામાં આવે છે. જવાબ: ""તારો""." \n English -> "I have a new ""___"" . Answer: ""idea""." - ગુજરાતી -> "મારી પાસે એક નવું ""___"" છે. જવાબ: ""વિચાર""." \n English -> "Loud ""___""  was played in the auditorium. Answer: ""music""." - ગુજરાતી -> ઓડિટોરિયમમાં ""___"" મોટેથી વગાડવામાં આવ્યું હતું. જવાબ: ""સંગીત""." \n English -> "People love watching a ""___"" in the theater. Answer: ""movie""." - ગુજરાતી -> "લોકોને થિયેટરમાં ""___" જોવાનું પસંદ છે. જવાબ: ""મૂવી""."  \n  English -> "India is the most populated ""___"" in the world . Answer: ""country""." - ગુજરાતી -> "ભારત વિશ્વમાં સૌથી વધુ વસ્તી ધરાવતો ""___"" છે. જવાબ: ""દેશ""." \n '

    prompt_suggestion = "Just translate the english masked sentence and answer to Gujarati. Dont generate synonyms of the answer."
    new_prompt = f"\n English -> {blank_prompt_translation_masked} - ગુજરાતી -> "
    
    return [
    {"role": "user", "content":  examples_prompt + prompt_suggestion + new_prompt},
    ]


def format_hi_messages(blank_prompt_translation_masked):
    examples_prompt = ' English ->  "The ""___"" is seen as a tiny spot in the sky from earth . Answer: ""star""." - हिन्दी  -> ""___"" को पृथ्वी से आकाश में एक छोटे से स्थान के रूप में देखा जाता है। उत्तर: ""तारा""। \n English -> "I have a new ""___"" . Answer: ""idea""." - हिन्दी ->  "मेरे पास एक नया ""___"" है। उत्तर: ""विचार""।." \n English -> "Loud ""___""  was played in the auditorium. Answer: ""music""." - हिन्दी -> सभागार में "जोर से ""___"" बजाया गया। उत्तर: ""संगीत""। \n English -> "People love watching a ""___"" in the theater. Answer: ""movie""." - हिन्दी -> "लोग थिएटर में ""___"" देखना पसंद करते हैं। उत्तर: ""मूवी""।  \n  English -> "India is the most populated ""___"" in the world . Answer: ""country""." - हिन्दी -> "भारत दुनिया में सबसे अधिक आबादी वाला ""___"" देश है। उत्तर: ""देश""।  \n '

    prompt_suggestion ="Just translate the english masked sentence and answer to Hindi. Dont generate synonyms of the answer."
    new_prompt = f"\n English -> {blank_prompt_translation_masked} - हिन्दी -> "
    
    return [
    {"role": "user", "content":  examples_prompt + prompt_suggestion + new_prompt},
    ]

def format_ml_messages(blank_prompt_translation_masked):
    examples_prompt = ' English ->  "The ""___"" is seen as a tiny spot in the sky from earth . Answer: ""star""." - മലയാളം -> ""___"" ഭൂമിയിൽ നിന്ന് ആകാശത്തിലെ ഒരു ചെറിയ പൊട്ടായാണ് കാണുന്നത്. ഉത്തരം: ""നക്ഷത്രം"".  \n English -> "I have a new ""___"" . Answer: ""idea""." - മലയാളം -> "എനിക്ക് ഒരു പുതിയ ""___"" ഉണ്ട്. ഉത്തരം: ""ആശയം, ആലോചന""." \n English -> "Loud ""___""  was played in the auditorium. Answer: ""music""." - മലയാളം -> "ഉച്ചത്തിൽ ""___"" ഓഡിറ്റോറിയത്തിൽ പ്ലേ ചെയ്തു. ഉത്തരം: ""സംഗീതം, പാട്ട്, ഗാനം""." \n English -> "People love watching a ""___"" in the theater. Answer: ""movie""." - മലയാളം -> "തീയറ്ററിൽ ""___"" കാണാൻ ആളുകൾ ഇഷ്ടപ്പെടുന്നു. ഉത്തരം: ""സിനിമ, ചലച്ചിത്രം, പടം""."  \n  English -> "India is the most populated ""___"" in the world . Answer: ""country""." - മലയാളം -> "ലോകത്തിലെ ഏറ്റവും ജനസംഖ്യയുള്ള ""___"" ഇന്ത്യയാണ്. ഉത്തരം: ""രാജ്യം, നാട്, രാജം, ദേശം""."  \n '
    prompt_suggestion = "Just translate the english masked sentence and answer to Malayalam. Dont generate synonyms of the answer."
    
    new_prompt = f"\n English -> {blank_prompt_translation_masked} - മലയാളം -> "
    
    return [
    {"role": "user", "content":  examples_prompt + prompt_suggestion + new_prompt},
    ]
# 'Just translate the english masked sentence and answer to Malayalam dont generate anythiing else. Gernerate all the synonyms of the answer in Malayalam but number of synonyms generated should not be more than 5. Separate the synonyms with a comma' +

def format_es_messages(en_word):
    examples_prompt = " English: star - español: estrella \n English: crime - español: delito \n English: music - español: música \n English: movie - español: película \n  English: country - español: país \n "

    
    new_prompt = f"\n English: {en_word} - español: "
    
    return [
    {"role": "user", "content":  examples_prompt +'Just translate the english word to Spanish dont generate anythiing else.' +new_prompt},
    ]
def format_it_messages(en_word):
    examples_prompt = " English: star - Italiano: stella \n English: crime - Italiano: crimine \n English: music - Italiano: musica \n English: movie - Italiano: film \n  English: country - Italiano: paese \n "

    
    new_prompt = f"\n English: {en_word} - Italiano: "
    
    return [
    {"role": "user", "content":  examples_prompt +'Just translate the english word to Italian dont generate anythiing else.' +new_prompt},
    ]


def format_da_messages(en_word):
    examples_prompt = " English: star - dansk: stjerne \n English: crime - dansk: forbrydelse \n English: music - dansk: musik \n English: sea - dansk: hav \n  English: beautiful - dansk: smuk \n "

    
    new_prompt = f"\n English: {en_word} - dansk: "
    
    return [
    {"role": "user", "content":  examples_prompt +'Just translate the english word to Danish dont generate anythiing else.' +new_prompt},
    ]









        

def main(input_lang = 'en', target_lang = 'hi'):

    

    model_id = "meta-llama/Llama-3.3-70B-Instruct"
    df_en_fr = pd.read_csv(f'{prefix}{input_lang}/clean.csv').reindex()
    df = df_en_fr
    df1 = pd.read_csv(f'{prefix}{target_lang}/clean6.csv').reindex()


    df1["blank_prompt_translation_masked"] = ""


    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    for idx, row in df.iterrows():
        if row["word_original"] != df1.at[idx, "word_original"]:
            print('skipping...')
            continue

        if target_lang == 'hi':
            messages = format_hi_messages(row["blank_prompt_translation_masked"])
        if target_lang == 'ml':
            messages = format_ml_messages(row["blank_prompt_translation_masked"])
        if target_lang == 'gu':
            messages = format_gu_messages(row["blank_prompt_translation_masked"])
        if target_lang == 'ta':
            messages = format_ta_messages(row["blank_prompt_translation_masked"])
        if target_lang == 'te':
            messages = format_te_messages(row["blank_prompt_translation_masked"])
        if target_lang == 'ka':
            messages = format_ka_messages(row["blank_prompt_translation_masked"])
        if target_lang == 'am':
            messages = format_am_messages(row["blank_prompt_translation_masked"])
        if target_lang == 'he':
            messages = format_he_messages(row["blank_prompt_translation_masked"])
        if target_lang == 'th':
            messages = format_th_messages(row["blank_prompt_translation_masked"])
      

        outputs = pipeline(
            messages,
            max_new_tokens= 500,

        )
        print(outputs)
        outputs1 = outputs[0]["generated_text"][-1]['content']
        
    
        # print(outputs1)
        stop_sequence = '\n'
        if stop_sequence in outputs1:
            outputs1 = outputs1.split(stop_sequence)[0]
        
        
        print(outputs1)
        
        df1.at[idx, "blank_prompt_translation_masked"] = outputs1.strip()
        
    print('yu')

    df1[["lang","word_original","word_translation","blank_prompt_translation_masked"]].to_csv(f'./data/langs/{target_lang}/clean_cloze.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject parameters into a Python script.")
    parser.add_argument('--input_lang', type=str, required=True, help='Input lang')
    parser.add_argument('--target_lang', type=str, required=True, help='Target lang')

    args = parser.parse_args()
    main(args.input_lang, args.target_lang)