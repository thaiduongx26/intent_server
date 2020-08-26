import pandas as pd
from tqdm import tqdm
import random, json
import os
import unidecode

excel_path = "D:\\flair_classification\\data\\Final_Create_Extend_Disable_T24.xlsx"
train_json = "D:\\flair_classification\\data/train.json"
test_json = "D:\\flair_classification\\data/test.json"
data_json = "D:\\flair_classification\\data/data.json"

sheet_dict = {'Cap moi': False,
              'Gia han': True,
              'Thu hoi': True,
              'Unlock': True,
              'Reset-dang lam': True}

data = {'others': [],
        'Cấp mới T24_Không xác định ý định\nint_create_t24_need_confirm': [],
        'Cấp mới T24_Thủ tục': [],
        'Cấp mới T24_Cấp phê duyệt': [],
        'Cấp mới T24_Chế độ tiếp nhận\nint_create_t24_onboarding': [],
        'Cấp mới T24_Mẫu biểu T24': [],
        'Cấp mới T24_Đối tượng\nint_create_t24_subject': [],
        'Cấp mới T24_Thời gian cam kết': [],
        'Cấp mới T24_Cấp mới mặc định': [],
        'Cấp mới T24_File đề nghị cấp T24 (int_create_t24_attached)': [],
        'Cấp mới T24_FAQ': []}

label_list = ['others',
              'Cấp mới T24_Không xác định ý định\nint_create_t24_need_confirm',
              'Cấp mới T24_Thủ tục',
              'Cấp mới T24_Cấp phê duyệt',
              'Cấp mới T24_Chế độ tiếp nhận\nint_create_t24_onboarding',
              'Cấp mới T24_Mẫu biểu T24',
              'Cấp mới T24_Đối tượng\nint_create_t24_subject',
              'Cấp mới T24_Thời gian cam kết',
              'Cấp mới T24_Cấp mới mặc định',
              'Cấp mới T24_File đề nghị cấp T24 (int_create_t24_attached)',
              'Cấp mới T24_FAQ']

label_map = {label_list[i]: str(i) for i in range(len(label_list))}
label_map_reversed = {str(i): label_list[i] for i in range(len(label_list))}


def parse_sheet(sheet_name, is_other=False):
    excel = pd.read_excel(excel_path, sheet_name=sheet_name)
    intents = excel['INTENT'].values
    if is_other:
        for intent in intents:
            if isinstance(intent, str):
                data['others'].append(intent)
    else:
        current_intent = None
        for text in intents:
            if not isinstance(text, str):
                continue
            else:
                if text in data:
                    current_intent = text
                elif current_intent:
                    data[current_intent].append(text)
                else:
                    continue
    pass


def split_data(data, ratio=0.3):
    train_data = {tag: [] for tag in data}
    test_data = {tag: [] for tag in data}
    for tag in data:
        _len = len(data[tag])
        test_idx = random.sample(range(_len), max(1, int(ratio * _len)))
        train_idx = [idx for idx in range(_len) if idx not in test_idx]
        for i in range(_len):
            if i in train_idx:
                train_data[tag].append(data[tag][i])
            else:
                test_data[tag].append(data[tag][i])
    with open(data_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    with open(train_json, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False)
    with open(test_json, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False)


prefixes = ["em ơi cho chị hỏi ", "em ơi chị hỏi chút ", "bạn cho mình hỏi ", "thế anh muốn hỏi ", "thế ",
            "cho anh hỏi chút ", "chị muốn hỏi một chút ", "thắc mắc một tẹo ", "em xinh đẹp ơi, ", "em bot xinh đẹp, ",
            "em gái xinh đẹp ", "bot ơi"]
subfixes = ["ạ", "ơi", "với"]

typos = {'ạ': 'aj', "nhé": "nhes", "ơi": "owi", "thế": "thees", "hỏi": "hoir"}

# intents_allow_augment = ['change_schedule', 'ask_admin_booking', 'salary_related', 'mirror_process',
#                          'update_self_review',
#                          'ask_review_result', 'ask_check_meeting', 'ask_self_G_result', 'ask_admin_computer',
#                          'ask_other_feedback', 'update_self_review', 'ask_admin_onboard', 'ask_leave',
#                          'ask_reiew_late', 'what_can_do', 'ask_name_bot', 'ask_howold']
intents_not_augment = ['others']


def augment_text(text: str = None, augment_ratio=0.5):
    all_text = [text]

    def get_augment_factor(all_augment_factors):
        number_of_factor = random.randint(0, len(all_augment_factors) - 1)
        if isinstance(all_augment_factors, list):
            return random.sample(all_augment_factors, number_of_factor)
        else:
            keys = random.sample(list(all_augment_factors.keys()), number_of_factor)
            return {k: all_augment_factors[k] for k in keys}

    def add_prefix(text, augment_factor):
        return augment_factor + text

    def add_subfix(text, augment_factor):
        text = text.strip()
        if text[-1] in ['?']:
            text = text[:-1]
        return text + ' ' + augment_factor

    def lower_text(text, augment_factor=None):
        return text.lower()

    def unidecode_text(text, augment_factor=None):
        return unidecode.unidecode(text)

    def do_augment(text_list, augment_fuction, all_augment_factors):
        augmented_text = []
        for text in text_list:
            ratio = random.uniform(0, 1)
            if ratio < augment_ratio:
                continue
            if len(all_augment_factors) > 0:
                augment_factors = get_augment_factor(all_augment_factors)
            else:
                augment_factors = [""]
            for factor in augment_factors:
                augmented_text.append(augment_fuction(text, factor))
        return augmented_text

    all_text.extend(do_augment(all_text, add_prefix, prefixes))
    all_text.extend(do_augment(all_text, add_subfix, subfixes))
    all_text.extend(do_augment(all_text, lower_text, []))
    all_text.extend(do_augment(all_text, unidecode_text, []))
    # all_text.extend(do_augment(all_text, replace_typos, typ))

    return list(set(all_text))


def convert_to_fastext(data_file, do_augment=True, output_file=None):
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    train_data = []
    for tag in data:
        if tag == 'others':
            continue
        for text in data[tag]:
            text = text.lower()
            a_sent = ''
            a_sent += '__label__' + label_map[tag] + ' '
            if not do_augment or tag in intents_not_augment:
                a_sent = a_sent[:len(a_sent) - 1] + '\t' + text
                # print(a_sent)
                train_data.append(a_sent)
            else:
                augmented_text_list = augment_text(text)
                for text in augmented_text_list:
                    curr_sent = a_sent[:len(a_sent) - 1] + '\t' + text
                    train_data.append(curr_sent)

    with open(output_file, 'w', encoding='utf-8') as myfile:
        [myfile.write(sent + '\n') for sent in train_data]
        myfile.close()


if __name__ == '__main__':
    # sheet_name = 'Cap moi'
    # for sheet in tqdm(sheet_dict):
    #     parse_sheet(sheet, sheet_dict[sheet])
    # split_data(data)
    # print(data)
    convert_to_fastext("D:\\flair_classification\\data/train.json", False, "D:\\flair_classification\\data/train.csv")
    convert_to_fastext("D:\\flair_classification\\data/test.json", False, "D:\\flair_classification\\data/test.csv")
