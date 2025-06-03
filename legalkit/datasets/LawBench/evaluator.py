from typing import List, Dict
from legalkit.datasets.base import BaseEvaluator
from legalkit.datasets.LawBench.other_utils.rc_f1 import CJRCEvaluator
from legalkit.datasets.LawBench.utils import compute_rouge, multi_choice_judge, compute_f1_two_sets, compute_rc_f1, compute_ie_f1
from legalkit.datasets.utils import clean_prediction

import re
import math
import cn2an
import os
import subprocess
import tempfile

# ---------------------- score computer -----------------------

def compute_cjft(data_dict):
    """
    Compute the ROUGE-L score between the prediction and the reference
    """
    references, predictions = [], []
    for example in data_dict:
        question, prediction, answer = example["origin_prompt"], example["prediction"], example["refr"]
        predictions.append(prediction)
        references.append(answer)

    # compute the accuracy of score_list
    rouge_scores = compute_rouge(predictions, references)
    rouge_ls = [score["rouge-l"]["f"] for score in rouge_scores]
    average_rouge_l = sum(rouge_ls) / len(rouge_ls)
    return {"score": average_rouge_l}

def compute_flzx(data_dict):
    """
    Compute the ROUGE-L score between the prediction and the reference
    """
    references, predictions = [], []
    for example in data_dict:
        question, prediction, answer = example["origin_prompt"], example["prediction"], example["refr"]
        predictions.append(prediction)
        references.append(answer)

    # compute the accuracy of score_list
    rouge_scores = compute_rouge(predictions, references)
    rouge_ls = [score["rouge-l"]["f"] for score in rouge_scores]
    average_rouge_l = sum(rouge_ls) / len(rouge_ls)
    return {"score": average_rouge_l}

def compute_ftcs(data_dict):
    """
    Compute the ROUGE-L score between the prediction and the reference
    """
    references, predictions = [], []
    for example in data_dict:
        question, prediction, answer = example["origin_prompt"], example["prediction"], example["refr"]
        answer = answer.replace("答案:", "")
        predictions.append(prediction)
        references.append(answer)

    # compute the accuracy of score_list
    rouge_scores = compute_rouge(predictions, references)
    rouge_ls = [score["rouge-l"]["f"] for score in rouge_scores]
    average_rouge_l = sum(rouge_ls) / len(rouge_ls)
    return {"score": average_rouge_l}

def compute_jdzy(data_dict):
    """
    Compute the Accuracy
    The JEC dataset has 16 possible answers for each question, stored in the option_list
    A prediction is correct if
    1. The correct answer appears in the prediction, and
    2. Options other than the answer do not appear in the prediction.
    """

    score_list, abstentions = [], 0
    option_list = ["诉讼主体", "租金情况", "利息", "本金争议", "责任认定", "责任划分", "损失认定及处理",
                   "原审判决是否适当", "合同效力", "财产分割", "责任承担", "鉴定结论采信问题", "诉讼时效", "违约", "合同解除", "肇事逃逸"]
    for example in data_dict:
        question, prediction, answer = example["origin_prompt"], example["prediction"], example["refr"]
        if answer[7:-1] == "赔偿":
            # todo: dataset imperfection
            continue
        assert answer.startswith("争议焦点类别：") and answer[7:-1] in option_list, \
            f"answer: {answer} \n question: {question}"

        answer_letter = answer[7:-1]
        judge = multi_choice_judge(prediction, option_list, answer_letter)
        score_list.append(judge["score"])
        abstentions += judge["abstention"]

    # compute the accuracy of score_list
    accuracy = sum(score_list) / len(score_list)
    return {"score": accuracy, "abstention_rate": abstentions / len(data_dict)}

def compute_jec_ac(data_dict):
    """
    Compute the Accuracy
    The JEC dataset has 4 options for each question: A, B, C, D
    A prediction is correct if
    1. The correct answer appears in the prediction, and
    2. Options other than the answer do not appear in the prediction.
    """
    score_list, abstentions = [], 0
    option_list = ["A", "B", "C", "D"]
    for example in data_dict:
        question, prediction, answer = example["origin_prompt"], example["prediction"], example["refr"]
        assert answer.startswith("正确答案:") and answer[5] in option_list, f"answer[5]: {answer}, question: {question}"

        answer_letter = answer[5]
        judge = multi_choice_judge(prediction, option_list, answer_letter)
        score_list.append(judge["score"])
        abstentions += judge["abstention"]

    # compute the accuracy of score_list
    accuracy = sum(score_list) / len(score_list)
    return {"score": accuracy, "abstention_rate": abstentions / len(data_dict)}

def compute_jec_kd(data_dict):
    """
    Compute the Accuracy
    The JEC_KD dataset has 4 options for each question: A, B, C, D
    A prediction is correct if
    1. The correct answer appears in the prediction, and
    2. Options other than the answer do not appear in the prediction.
    """
    score_list, abstentions = [], 0
    option_list = ["A", "B", "C", "D"]
    for example in data_dict:
        question, prediction, answer = example["origin_prompt"], example["prediction"], example["refr"]
        assert answer.startswith("正确答案：") and answer[5] in option_list, f"answer[5]: {answer}, question: {question}"

        answer_letter = answer[5]
        judge = multi_choice_judge(prediction, option_list, answer_letter)
        score_list.append(judge["score"])
        abstentions += judge["abstention"]

    # compute the accuracy of score_list
    accuracy = sum(score_list) / len(score_list)
    return {"score": accuracy, "abstention_rate": abstentions / len(data_dict)}

def compute_jetq(data_dict):
    """
    Compute the Accuracy
    we extract the total amount of cost involved in the crime from the prediction and compare it with the reference
    The prediction is correct if
    the total amount of cost provided in the reference, appears in the prediction.
    """
    score_list, abstentions = [], 0

    for example in data_dict:
        question, prediction, answer = example["origin_prompt"], example["prediction"], example["refr"]
        assert answer.startswith("上文涉及到的犯罪金额:"), f"answer: {answer}, question: {question}"
        assert answer.endswith("元。"), f"answer: {answer}, question: {question}"
        answer = answer.replace("上文涉及到的犯罪金额:", "")

        assert "千元" not in answer, f"answer: {answer}, question: {question}"
        assert "万" not in answer, f"answer: {answer}, question: {question}"

        # remove "元"
        answer = answer.replace("元。", "")
        answer = float(answer)

        prediction_digits = re.findall(r"\d+\.?\d*", prediction)
        prediction_digits = [float(digit) for digit in prediction_digits]

        if len(prediction_digits) == 0:
            abstentions += 1
        if answer in prediction_digits:
            score_list.append(1)
        else:
            score_list.append(0)


    # compute the accuracy of score_list
    accuracy = sum(score_list) / len(score_list)
    return {"score": accuracy, "abstention_rate": abstentions/len(data_dict)}

def compute_lblj(data_dict):
    """
    Compute the Accuracy
    The LBLJ dataset has 5 options for each question: A, B, C, D, E
    A prediction is correct if
    1. The correct answer appears in the prediction, and
    2. Options other than the answer do not appear in the prediction.
    """
    score_list, abstentions = [], 0
    option_list = ["A", "B", "C", "D", "E"]
    for example in data_dict:
        question, prediction, answer = example["origin_prompt"], example["prediction"], example["refr"]
        assert answer.startswith("[正确答案]") and answer[6] in option_list, f"answer[6]: {answer}, question: {question}"

        answer_letter = answer[6]
        judge = multi_choice_judge(prediction, option_list, answer_letter)
        score_list.append(judge["score"])
        abstentions += judge["abstention"]

    # compute the accuracy of score_list
    accuracy = sum(score_list) / len(score_list)
    return {"score": accuracy, "abstention_rate": abstentions / len(data_dict)}

def compute_ljp_accusation(data_dict):
    option_list = ["侮辱", "违法发放贷款", "失火", "票据诈骗", "帮助犯罪分子逃避处罚", "重大责任事故", "对非国家工作人员行贿",
                   "非法制造、销售非法制造的注册商标标识", "非法制造、买卖、运输、邮寄、储存枪支、弹药、爆炸物", "非法获取公民个人信息",
                   "扰乱无线电通讯管理秩序", "非法持有、私藏枪支、弹药", "拒不执行判决、裁定", "虚开发票", "巨额财产来源不明",
                   "组织、领导、参加黑社会性质组织", "非法获取国家秘密", "以危险方法危害公共安全", "非法持有毒品",
                   "聚众扰乱公共场所秩序、交通秩序", "包庇毒品犯罪分子", "滥伐林木", "伪造公司、企业、事业单位、人民团体印章",
                   "非法占用农用地", "走私废物", "串通投标", "非法采伐、毁坏国家重点保护植物", "冒充军人招摇撞骗", "玩忽职守",
                   "重婚", "招收公务员、学生徇私舞弊", "组织、领导传销活动", "非法猎捕、杀害珍贵、濒危野生动物", "侵犯著作权",
                   "非法种植毒品原植物", "伪造、变造、买卖武装部队公文、证件、印章", "倒卖文物", "伪造、变造居民身份证", "滥用职权",
                   "诽谤", "猥亵儿童", "非法转让、倒卖土地使用权", "挪用公款", "污染环境", "出售、购买、运输假币", "敲诈勒索",
                   "高利转贷", "故意伤害", "持有、使用假币", "单位受贿", "强奸", "引诱、容留、介绍卖淫", "虐待",
                   "生产、销售伪劣农药、兽药、化肥、种子", "妨害公务", "容留他人吸毒", "拐骗儿童", "强制猥亵、侮辱妇女",
                   "非法处置查封、扣押、冻结的财产", "骗取贷款、票据承兑、金融票证", "强迫他人吸毒", "非法拘禁",
                   "非法携带枪支、弹药、管制刀具、危险物品危及公共安全", "绑架", "聚众斗殴", "破坏计算机信息系统",
                   "制造、贩卖、传播淫秽物品", "虐待被监管人", "贷款诈骗", "赌博", "徇私舞弊不征、少征税款",
                   "盗窃、抢夺枪支、弹药、爆炸物、危险物质", "故意杀人", "介绍贿赂", "提供侵入、非法控制计算机信息系统程序、工具",
                   "编造、故意传播虚假恐怖信息", "妨害作证", "强迫卖淫", "走私、贩卖、运输、制造毒品", "伪证", "拐卖妇女、儿童",
                   "过失损坏武器装备、军事设施、军事通信", "破坏广播电视设施、公用电信设施", "洗钱", "职务侵占", "倒卖车票、船票",
                   "抢劫", "侵占", "掩饰、隐瞒犯罪所得、犯罪所得收益", "徇私舞弊不移交刑事案件", "引诱、教唆、欺骗他人吸毒", "遗弃",
                   "生产、销售伪劣产品", "放火", "非法采矿", "对单位行贿", "盗窃、抢夺枪支、弹药、爆炸物", "破坏易燃易爆设备",
                   "妨害信用卡管理", "制作、复制、出版、贩卖、传播淫秽物品牟利", "金融凭证诈骗", "私分国有资产",
                   "走私国家禁止进出口的货物、物品", "假冒注册商标", "危险物品肇事", "走私普通货物、物品", "经济犯", "虚报注册资本",
                   "盗掘古文化遗址、古墓葬", "传播淫秽物品", "窝藏、包庇", "拒不支付劳动报酬", "行贿", "开设赌场", "传授犯罪方法",
                   "协助组织卖淫", "保险诈骗", "破坏生产经营", "破坏交通设施", "打击报复证人", "非法侵入住宅", "非国家工作人员受贿",
                   "过失致人重伤", "伪造、变造金融票证", "窝藏、转移、隐瞒毒品、毒赃", "帮助毁灭、伪造证据", "走私珍贵动物、珍贵动物制品",
                   "生产、销售假药", "逃税", "挪用特定款物", "聚众扰乱社会秩序", "组织、强迫、引诱、容留、介绍卖淫", "合同诈骗",
                   "非法生产、销售间谍专用器材", "破坏交通工具", "传播性病", "强迫交易", "隐匿、故意销毁会计凭证、会计帐簿、财务会计报告",
                   "非法组织卖血", "强迫劳动", "破坏电力设备", "销售假冒注册商标的商品", "收买被拐卖的妇女、儿童", "诬告陷害", "脱逃",
                   "非法经营", "徇私枉法", "信用卡诈骗", "生产、销售不符合安全标准的食品", "非法行医", "伪造货币", "动植物检疫徇私舞弊",
                   "单位行贿", "破坏监管秩序", "盗窃", "盗伐林木", "重大劳动安全事故", "非法吸收公众存款",
                   "非法制造、出售非法制造的发票", "非法狩猎", "组织卖淫", "非法买卖、运输、携带、持有毒品原植物种子、幼苗", "挪用资金",
                   "诈骗", "伪造、变造、买卖国家机关公文、证件、印章", "持有伪造的发票", "贪污", "非法生产、买卖警用装备",
                   "投放危险物质", "伪造、倒卖伪造的有价票证", "集资诈骗", "抢夺", "生产、销售有毒、有害食品", "非法捕捞水产品",
                   "过失致人死亡", "非法买卖制毒物品", "虚开增值税专用发票、用于骗取出口退税、抵扣税款发票", "寻衅滋事", "危险驾驶",
                   "故意毁坏财物", "招摇撞骗", "盗窃、侮辱尸体", "走私武器、弹药",
                   "非法收购、运输、加工、出售国家重点保护植物、国家重点保护植物制品", "非法出售发票", "劫持船只、汽车",
                   "受贿", "聚众哄抢", "交通肇事"]
    
    score_list, abstentions = [], 0

    for example in data_dict:
        question, prediction, answer = example["origin_prompt"], example["prediction"], example["refr"]

        assert answer.startswith("罪名:"), f"answer: {answer} \n question: {question}"
        answer = answer.replace("罪名:", "")
        answers = answer.split(";")

        prediction_list =[]
        for option in option_list:
            if option in prediction:
                prediction_list.append(option)

        if len(prediction_list) == 0:
            abstentions += 1
        gt_set = set(answers)
        pred_set = set(prediction_list)
        score = compute_f1_two_sets(gt_set, pred_set)
        score_list.append(score)

    f1_score_average = sum(score_list) / len(score_list)
    return {"score": f1_score_average, "abstention_rate": abstentions/len(data_dict)}


def compute_ljp_article(data_dict):
    def replace_match(match):
        return match.group(1)
    
    score_list, abstentions = [], 0

    for example in data_dict:
        question, prediction, answer = example["origin_prompt"], example["prediction"], example["refr"]
        assert answer.startswith("法条:刑法第"), f"answer: {answer}"
        assert answer.endswith("条"), f"answer: {answer}"

        answer = answer.replace("法条:刑法第", "")
        answer = answer.replace("条", "")

        answer_law_indices = answer.split("、")
        answer_law_index_digit_list = []
        for answer_law_index in answer_law_indices:
            assert answer_law_index.isdigit(), f"answer_law_index: {answer_law_index}"
            answer_law_index_digit = int(answer_law_index)
            assert answer_law_index_digit <= 490, "刑法总共只有490条"
            answer_law_index_digit_list.append(answer_law_index_digit)

        prediction_law_chunks = prediction.split("、")
        prediction_law_index_digit_list = []

        for prediction_law_chunk in prediction_law_chunks:
            prediction_law_chunk = prediction_law_chunk.replace("万元", "元")

            # delete phrase starts with "第" and ends with "款", we don't care about it in the answer
            prediction_law_chunk = re.sub(r'第(.*?)款', "", prediction_law_chunk)
            # keep only the digits in the phrase starts with "第" and ends with "条", otherwise cn may fail to convert
            prediction_law_chunk = re.sub(r'第(.*?)条', replace_match, prediction_law_chunk)
            prediction_law_chunk = cn2an.transform(prediction_law_chunk, "cn2an")
            # find digtis in prediction_law_chunk
            prediction_law_section_numbers = re.findall(r"\d+", prediction_law_chunk)
            if len(prediction_law_section_numbers) == 0:
                continue
            if len(prediction_law_section_numbers) != 1:
                # in this case, we only take the first number, and reject the others
                pass

            prediction_law_index_digit = int(prediction_law_section_numbers[0])
            prediction_law_index_digit_list.append(prediction_law_index_digit)

        gt_set = set(answer_law_index_digit_list)
        pred_set = set(prediction_law_index_digit_list)
        if len(pred_set) == 0:
            abstentions += 1
        precision = len(gt_set.intersection(pred_set)) / len(pred_set) if len(pred_set) != 0 else 0
        recall = len(gt_set.intersection(pred_set)) / len(gt_set) if len(gt_set) != 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
        score_list.append(f1_score)

    # compute the accuracy of score_list
    average_f1 = sum(score_list) / len(score_list)
    return {'score': average_f1, 'abstention_rate': abstentions/len(data_dict)}

def compute_ljp_imprison(data_dict):
    score_list, abstentions = [], 0

    for example in data_dict:
        question, prediction, answer = example["origin_prompt"], example["prediction"], example["refr"]
        # get answer digit, which is the number between "刑期:" and "个月"
        if "死刑" in answer or "无期" in answer:
            # TODO: data imperfection
            continue

        assert answer.startswith("刑期:") and answer.endswith("个月"), f"answer: {answer}, question: {question}"
        answer = answer.replace("刑期:", "")
        answer = answer.replace("个月", "")
        answer_digit = int(answer)
        prediction = cn2an.transform(prediction, "cn2an")

        # use regular expression to extract the digits from prediction, only consider digits before "个月" or "月"
        prediction_digit_month_list = re.findall(r"\d+个月", prediction)
        prediction_digit_month_list = [int(digit.replace("个月", "")) for digit in prediction_digit_month_list]
        prediction_digit_month_list2 = re.findall(r"\d+月", prediction)
        prediction_digit_month_list2 = [int(digit.replace("月", "")) for digit in prediction_digit_month_list2]
        prediction_digit_month_list.extend(prediction_digit_month_list2)
        # catches the digits before "年"
        prediction_digit_year_list = re.findall(r"\d+年", prediction)
        prediction_digit_year_list = [int(digit.replace("年", "")) for digit in prediction_digit_year_list]

        if len(prediction_digit_month_list) > 0:
            prediction_digit_month = int(prediction_digit_month_list[0])
        elif len(prediction_digit_year_list) > 0:
            prediction_digit_month = int(prediction_digit_year_list[0]) * 12
        else:
            abstentions += 1
            prediction_digit_month = -1

        if prediction_digit_month != -1:
            score_list.append(abs(math.log(answer_digit + 1) - math.log(prediction_digit_month + 1)))
        else:
            score_list.append(math.log(216))
    
    if abstentions == len(score_list):
        log_distance = 0
    else:
        # compute the average of score_list (log distance)
        log_distance = sum(score_list) / len(score_list)
        # normalize the score to between 0 and 1
        log_distance = (math.log(216) - log_distance)/math.log(216)
    return {"score": log_distance, "abstention_rate": abstentions/len(data_dict)}


def compute_sjjc(data_dict):
    option_list = ["支付/给付", "欺骗", "搜查/扣押", "要求/请求", "卖出", "买入", "获利", "拘捕", "鉴定", "同意/接受", "供述", "联络", "帮助/救助", "租用/借用", "受伤", "伪造", "卖淫", "伤害人身", "赔偿", "归还/偿还"]
    score_list, abstentions = [], 0

    for example in data_dict:
        question, prediction, answer = example["origin_prompt"], example["prediction"], example["refr"]

        answers = answer.split(";")

        prediction_list =[]
        for option in option_list:
            if option in prediction:
                prediction_list.append(option)

        if len(prediction_list) == 0:
            abstentions += 1
        gt_set = set(answers)
        pred_set = set(prediction_list)
        score = compute_f1_two_sets(gt_set, pred_set)
        score_list.append(score)

    f1_score_average = sum(score_list) / len(score_list)
    return {"score": f1_score_average, "abstention_rate": abstentions/len(data_dict)}

def compute_cfcy(data_dict):

    scores = 0

    for example in data_dict:
        question, prediction, answer = example["origin_prompt"], example["prediction"], example["refr"]

        answers = answer.split(";")
        predictions = prediction.split(";")
        intersected = [CJRCEvaluator.compute_f1(r, h) for r, h in zip(answers, predictions)]

        prec = sum(intersected) / len(predictions) if len(predictions) > 0 else 0
        rec = sum(intersected) / len(answers) if len(answers) > 0 else 0
        # print(prec, rec, intersected)
        scores += 2 * prec * rec / (prec + rec + 1e-10)

    f1_score_average = scores / len(data_dict)
    return {"score": f1_score_average}

def compute_wbfl(data_dict):
    """
    A reference (R) contains a list of options, each option is from the option_list.
    We will extract the options appearing in the prediction and convert them into a set (P).
    We compute the F1 score between the prediction (P) and the reference (R).
    """


    score_list, abstentions = [], 0
    option_list = ["婚后有子女", "限制行为能力子女抚养", "有夫妻共同财产", "支付抚养费", "不动产分割", "婚后分局",
                   "二次起诉离婚", "按月给付抚养费", "准予离婚", "有夫妻共同债务", "婚前个人财产", "法定离婚", "不履行家庭义务",
                   "存在非婚生子", "适当帮助", "不履行离婚协议", "损害赔偿", "感情不和分居满二年", "子女随非抚养权人生活", "婚后个人财产"]
    for example in data_dict:
        question, prediction, answer = example["origin_prompt"], example["prediction"], example["refr"]
        assert answer.startswith("类别:") and answer.endswith("。"), f"answer: {answer}, question: {question}"

        gt_list = (answer[3:-1].split("、"))
        for gt in gt_list:
            assert gt in option_list, f"gt: {gt}, question: {question}"
        gt_set = set(gt_list)

        prediction_list = []
        for option in option_list:
            if option in prediction:
                prediction_list.append(option)
        if len(prediction_list) == 0:
            abstentions += 1
        predict_set = set(prediction_list)
        precision = len(gt_set.intersection(predict_set)) / len(predict_set) if len(predict_set) != 0 else 0
        recall = len(gt_set.intersection(predict_set)) / len(gt_set) if len(gt_set) != 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
        score_list.append(f1_score)

    # compute the accuracy of score_list
    final_f1_score = sum(score_list) / len(score_list)
    return {'score': final_f1_score, 'abstention_rate': abstentions / len(data_dict)}

def compute_wsjd(data_dict):
    origins, references, predictions = [], [], []
    for example in data_dict:
        question, prediction, answer = example["origin_prompt"], example["prediction"], example["refr"]
        if isinstance(question, list):
            question = question[0]['prompt']
        start = question.index('句子：\n') + 4
        origins.append(re.sub(r'\n|\t', '', question[start:].split('\n')[0]))
        # truncate predictions >5 tokens longer than the reference
        prediction = re.sub(r'\n|\t', '', prediction)
        if len(prediction) - len(answer) > 5:
            prediction = prediction[:len(answer) + 5]
        if len(prediction) == 0:
            prediction = "无内容"
        predictions.append(prediction)
        references.append(re.sub(r'\n|\t', '', answer))

    #generate input files for ChERRANT
    preds = [f'{i} \t {origin} \t {prediction} \n' for i, (origin, prediction) in enumerate(zip(origins, predictions))]
    golds = [f'{i} \t {origin} \t {reference} \n' for i, (origin, reference) in enumerate(zip(origins, references))]

    now_path = os.path.abspath(os.getcwd())
    utils_path = os.path.abspath(os.path.join(__file__, '..', 'other_utils'))
    os.chdir(utils_path)
    with tempfile.NamedTemporaryFile(delete=False, mode='w') as tmp_pred_file, \
            tempfile.NamedTemporaryFile(delete=False, mode='w') as tmp_gold_file:
        tmp_pred_file.writelines(preds)
        tmp_gold_file.writelines(golds)

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.system(f'python3 parallel_to_m2.py -f {tmp_pred_file.name} -o {tmp_pred_file.name}.m2 -g char')
    os.system(f'python3 parallel_to_m2.py -f {tmp_gold_file.name} -o {tmp_gold_file.name}.m2 -g char')
    output = subprocess.check_output(
        f"python3 compare_m2_for_evaluation.py -hyp {tmp_pred_file.name}.m2 -ref {tmp_gold_file.name}.m2", shell=True)
    score = float(output.decode().split('\t')[-1].split('\n')[0])
    #remove prediction files
    os.remove(tmp_pred_file.name)
    os.remove(tmp_gold_file.name)
    os.remove(f"{tmp_pred_file.name}.m2")
    os.remove(f"{tmp_gold_file.name}.m2")
    os.chdir(now_path)
    return {"score": score}

def compute_xxcq(data_dict):
    references, predictions = [], []
    for example in data_dict:
        question, prediction, answer = example["origin_prompt"], example["prediction"], example["refr"]
        predictions.append(prediction)
        references.append(answer)

    return compute_ie_f1(predictions, references, {"犯罪嫌疑人", "受害人", "被盗货币", "物品价值", "盗窃获利",
                                                   "被盗物品", "作案工具", "时间", "地点", "组织机构"})

def compute_ydlj(data_dict):
    references, predictions = [], []
    for example in data_dict:
        question, prediction, answer = example["origin_prompt"], example["prediction"], example["refr"]
        answer = answer.replace("回答:", "")
        predictions.append(prediction)
        references.append(answer)

    f1_score = compute_rc_f1(predictions, references)
    return f1_score

def compute_yqzy(data_dict):
    """
    Compute the ROUGE-L score between the prediction and the reference
    """
    references, predictions = [], []
    for example in data_dict:
        question, prediction, answer = example["origin_prompt"], example["prediction"], example["refr"]
        predictions.append(prediction)
        references.append(answer)

    # compute the accuracy of score_list
    rouge_scores = compute_rouge(predictions, references)
    rouge_ls = [score["rouge-l"]["f"] for score in rouge_scores]
    average_rouge_l = sum(rouge_ls) / len(rouge_ls)
    return {"score": average_rouge_l}

def compute_zxfl(data_dict):
    score_list, abstentions = [], 0
    option_list = ['婚姻家庭', '劳动纠纷', '交通事故', '债权债务', '刑事辩护', '合同纠纷', '房产纠纷', '侵权', '公司法', '医疗纠纷', '拆迁安置', '行政诉讼', '建设工程', '知识产权', '综合咨询', '人身损害', '涉外法律', '海事海商', '消费权益', '抵押担保']
    for example in data_dict:
        question, prediction, answer = example["origin_prompt"], example["prediction"], example["refr"]
        judge = multi_choice_judge(prediction, option_list, answer)
        score_list.append(judge["score"])
        abstentions += judge["abstention"]

    # compute the accuracy of score_list
    final_accuracy_score = sum(score_list) / len(score_list)
    return {'score': final_accuracy_score, 'abstention_rate': abstentions / len(data_dict)}


# ---------------------- evaluator -----------------------
# map task_id to compute function
funct_dict = {
    '1-1': compute_ftcs,
    '1-2': compute_jec_kd,
    '2-1': compute_wsjd,
    '2-2': compute_jdzy,
    '2-3': compute_wbfl,
    '2-4': compute_zxfl,
    '2-5': compute_ydlj,
    '2-6': compute_xxcq,
    '2-7': compute_yqzy,
    '2-8': compute_lblj,
    '2-9': compute_sjjc,
    '2-10': compute_cfcy,
    '3-1': compute_ljp_article,
    '3-2': compute_cjft,
    '3-3': compute_ljp_accusation,
    '3-4': compute_ljp_imprison,
    '3-5': compute_ljp_imprison,
    '3-6': compute_jec_ac,
    '3-7': compute_jetq,
    '3-8': compute_flzx,
}

class Evaluator(BaseEvaluator):
    def evaluate(
        self,
        task_id: str,
        records: List[Dict],
        predictions: Dict[int, str],
        origin_prompts: List[str] = None
    ) -> Dict[str, float]:
        scorer = funct_dict.get(task_id)
        if not scorer:
            return {'error': f"Unsupported task '{task_id}'"}

        data_dict = []
        for i, rec in enumerate(records):
            orig = origin_prompts[i] if origin_prompts else f"{rec['instruction']}\n{rec['question']}"
            data_dict.append({
                'origin_prompt': orig,
                'prediction': clean_prediction(predictions.get(rec['id'], '')),
                'refr': rec['answer']
            })

        score_result = scorer(data_dict)

        # normalize float scores
        return {k: (v * 100 if 0.0 <= v <= 1.0 else v) for k, v in score_result.items()}

