# -*- coding:utf-8 -*-
import pandas as pd
import pkuseg
import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
# from chemdataextractor import Document
# import logging
# import gensim
# from gensim.models import word2vec, KeyedVectors
from openccpy.opencc import Opencc
# import pynlpir
import re
from tqdm import tqdm
from mat2vec import MaterialsTextProcessor
text_processor = MaterialsTextProcessor()
text_processor.process("LiCoO2 is a battery cathode material.")


'''
1.去空格
'''
# with open(r'WF-CNKI.txt', 'r', encoding="utf-8") as f1,open(r"空格.txt",'w',encoding="utf-8") as f2:
#     for line in f1.readlines():
#         line1 = line.replace(" ","")
#         f2.write(line1)

'''
2.全角转半角
'''

# def full_to_half(sentence):  # 输入为一个句子
#     change_sentence = ""
#     for word in sentence:
#         inside_code = ord(word)
#         if inside_code == 12288:  # 全角空格直接转换
#             inside_code = 32
#         elif inside_code >= 65281 and inside_code <= 65374:  # 全角字符（除空格）根据关系转化
#             inside_code -= 65248
#         change_sentence += chr(inside_code)
#     return change_sentence
#
# with open(r"空格.txt", 'r', encoding="utf-8") as f1,open(r"空格-半角.txt",'w',encoding="utf-8") as f2:
#     for line in f1.readlines():
#         change_sentence = full_to_half(line)
#         f2.write(change_sentence)

'''
3.去除标点
'''

# def clear_character(sentence):
#     pattern = re.compile("[,'\".?!_`*:@^#&;:。，；：“？！、\[\]【】《》]")  #去掉中英文标点
#     #若只保留中英文和数字，则替换为[^\u4e00-\u9fa5^a-z^A-Z^0-9]
#     line=re.sub(pattern,'',sentence)  #把文本中匹配到的字符替换成空字符
#     new_sentence=''.join(line.split())    #去除空白
#     return new_sentence
#
# with open(r"空格-半角.txt", 'r', encoding="utf-8") as f1,open(r"空格-半角-标点.txt",'w',encoding="utf-8") as f2:
#     for line in f1.readlines():
#         change_sentence = clear_character(line)
#         change_sentence = change_sentence + '\n'
#         f2.write(change_sentence)

'''
4.繁体中文转简体中文
'''

# def Simplified(sentence):
#     new_sentence = Opencc.to_simple(sentence)   # 繁体转为简体
#     return new_sentence
#
# with open(r"空格-半角-标点.txt", 'r', encoding="utf-8") as f1,open(r"空格-半角-标点-简体.txt",'w',encoding="utf-8") as f2:
#     for line in f1.readlines():
#         change_sentence = Simplified(line)
#         f2.write(change_sentence)


'''
5.分词、停止词、自定义
'''
# stopwords = [line.strip() for line in open(r'stopwords.txt', encoding='UTF-8').readlines()] #加载自定义停止词
#
# sentence=str()
#
# with open(r"空格-半角-标点-简体.txt", encoding='utf-8') as f: #加载原始数据库并分词
#     document = f.read()
#     seg = pkuseg.pkuseg(user_dict = r"userdict.txt")
#     text=seg.cut(document)
#
#     result = ' '.join(text)
#     for word in result:
#         if word not in stopwords:
#             if word != "\t":
#                 sentence += word
#     with open(r"空格-半角-标点-简体-分词自定停止.txt", 'w',encoding="utf-8") as f2:
#         f2.write(sentence)

'''
6.汉字数字转阿拉伯数字
'''
# with open(r"空格-半角-标点-简体-分词自定停止.txt", 'r', encoding="utf-8") as f1,open(r"空格-半角-标点-简体-分词自定停止-数字.txt",'w',encoding="utf-8") as f2:
#     content = f1.read()
#     new_content = content.replace(" 一 "," 1 ").replace(" 二 "," 2 ").replace(" 三 "," 3 ").replace(" 四 "," 4 ").\
#         replace(" 五 "," 5 ").replace(" 六 "," 6 ").replace(" 七 "," 7 ").replace(" 八 "," 8 ").replace(" 九 "," 9 ").\
#         replace(" 十 "," 10 ")
#     f2.write(new_content)

'''
7.词形统一（元素化学式）
'''

# with open(r"空格-半角-标点-简体-分词自定停止-数字.txt", 'r', encoding="utf-8") as f1,open(r"空格-半角-标点-简体-分词自定停止-数字-统一.txt",'w',encoding="utf-8") as f2:
#     content = f1.read()
#     new_content = content.replace(" 氢 "," H ").replace(" 氦 "," He ").replace(" 锂 "," Li ").replace(" 铍 "," Be ").\
#         replace(" 硼 "," B ").replace(" 碳 "," C ").replace(" 氮 "," N ").replace(" 氧 "," O ").replace(" 氟 "," F ").\
#         replace(" 氖 "," Ne ").replace(" 钠 "," Na ").replace(" 镁 "," Mg ").replace(" 铝 "," Al ").replace(" 硅 "," Si ").\
#         replace(" 磷 "," P ").replace(" 硫 "," S ").replace(" 氯 "," Cl ").replace(" 氩 "," Ar ").replace(" 钾 "," K ").\
#         replace(" 钙 "," Ca ").replace(" 钪 "," Sc ").replace(" 钛 "," Ti ").replace(" 钒 "," V ").replace(" 铬 "," Cr ").\
#         replace(" 锰 "," Mn ").replace(" 铁 "," Fe ").replace(" 钴 "," Co ").replace(" 镍 "," Ni ").replace(" 铜 "," Cu ").\
#         replace(" 锌 "," Zn ").replace(" 镓 "," Ga ").replace(" 锗 "," Ge ").replace(" 砷 "," As ").replace(" 硒 "," Se ").\
#         replace(" 溴 "," Br ").replace(" 氪 "," Kr ").replace(" 铷 "," Rb ").replace(" 锶 "," Sr ").replace(" 钇 "," Y ").\
#         replace(" 锆 "," Zr ").replace(" 铌 "," Nb ").replace(" 钼 "," Mo ").replace(" 锝 "," Tc ").replace(" 钌 "," Ru ").\
#         replace(" 铑 "," Rh ").replace(" 钯 "," Pd ").replace(" 银 "," Ag ").replace(" 镉 "," Cd ").replace(" 铟 "," In ").\
#         replace(" 锡 "," Sn ").replace(" 锑 "," Sb ").replace(" 碲 "," Te ").replace(" 碘 "," I ").replace(" 氙 "," Xe ").\
#         replace(" 铯 "," Cs ").replace(" 钡 "," Ba ").replace(" 铪 "," Hf ").replace(" 钽 "," Ta ").replace(" 钨 "," W ").\
#         replace(" 铼 "," Re ").replace(" 锇 "," Os ").replace(" 铱 "," Ir ").replace(" 铂 "," Pt ").replace(" 金 "," Au ").\
#         replace(" 汞 "," Hg ").replace(" 铊 "," Tl ").replace(" 铅 "," Pb ").replace(" 铋 "," Bi ").replace(" 钋 "," Po ").\
#         replace(" 砹 "," At ").replace(" 氡 "," Rn ").replace(" 钫 "," Fr ").replace(" 镭 "," Ra ").\
#         replace(" 氢气 "," H2 ").replace(" 氢氧化钠 "," NaOH ").replace(" 氢氧化镁 "," Mg(OH)2 ").replace(" 氢氧化铝 "," Al(OH)3 ").\
#         replace(" 氢氧化钙 "," Ca(OH)2 ").replace(" 氢氧化钡 "," Ba(OH)2 ").replace(" 氢氧化钾 "," KOH ").replace(" 氢氧化锂 "," LiOH ").\
#         replace(" 氢氧化镍 "," Ca(OH)2 ").replace(" 氢氟酸 "," HF ").replace(" 碳酸氢铵 "," NH4HCO3 ").replace(" 硫化氢 "," H2S ").\
#         replace(" 碳酸氢钠 "," NaHCO3 ").replace(" 氢氧化铵 "," NH₃·H₂O ").replace(" 氢氧化铅 "," Pb(OH)2 ").replace(" 氨基锂 "," LiNH2 ").\
#         replace(" 氧化铍 "," BeO ").replace(" 硼酸 "," H3BO3 ").replace(" 钕铁硼 "," Nd2Fe14B ").replace(" 碳化硼 "," B4C ").\
#         replace(" 氮化硼 "," BN ").replace(" 硼砂 "," Na2B4O7·10H2O ").replace(" 硼酸锌 "," B2O6Zn3 ").replace(" 氨硼烷 "," NH3·BH3 ").\
#         replace(" 苯硼酸 "," Phenylboronic-Acid ").replace(" 氧化硼 "," B2O3 ").replace(" 碳化硅 "," SiC ").replace(" 碳酸钙 "," CaCO3 ").\
#         replace(" 二氧化碳 "," CO2 ").replace(" 聚碳酸酯 "," Polycarbonate ").replace(" 纳米碳酸钙 "," CaCO3 ").replace(" 碳化钨 "," WC ").\
#         replace(" 碳酸钠 "," Na2CO3 ").replace(" 碳酸镁 "," MgCO3 ").replace(" 碳酸铵 "," (NH4)2CO3 ").replace(" 石墨碳 "," graphite ").\
#         replace(" 碳酸锰 "," MnCO3 ").replace(" 二硫化碳 "," CS2 ").replace(" 四氯化碳 "," CCl4 ").replace(" 氮化钛 "," TiN ").\
#         replace(" 氮化硅 "," Si3N4 ").replace(" 氮气 "," N2 ").replace(" 环氧树脂 "," Phenolic-epoxy-resin ").replace(" 二氧化硅 "," SiO2 ").\
#         replace(" 二氧化钛 "," TiO2 ").replace(" 氧化铝 "," Al2O3 ").replace(" 氧气 "," O2 ").replace(" 二氧化锰 "," MnO2 ").replace(" 氧化锆 "," ZrO2 ").\
#         replace(" 氧化锌 "," ZnO ").replace(" 双氧水 "," hydrogen-peroxide ").replace(" 硅氧烷 "," Siloxane ").replace(" 氧化镁 "," MgO ").\
#         replace(" 二氧化锡 "," SnO2 ").replace(" 纳米氧化锌 "," ZnO ").replace(" 氧化硅 "," SiO2 ").replace(" 氧化硅 "," SiO2 ").replace(" 纳米氧化铝 "," Al2O3 ").\
#         replace(" 氧化铁 "," Fe2O3 ").replace(" 二氧化锆 "," ZrO2 ").replace(" 环氧氯丙烷 "," Epichlorohydrin ").replace(" 氧化铈 "," CeO2 ").replace(" 氧化钙 "," CaO ").\
#         replace(" 二氧化钒 "," VO2 ").replace(" 环氧乙烷 "," Epoxyethane ").replace(" 三氧化钼 "," MoO3 ").replace(" 氧化钛 "," TiO2 ").replace(" 氧化镧 "," La2O3 ").\
#         replace(" 四氧化三铁 "," Fe3O4 ").replace(" 三氧化钨 "," WO3 ").replace(" 二氧化铈 "," CeO2 ").replace(" 二氧化铅 "," PbO2 ").replace(" 三氧化二铁 "," Fe2O3 ").\
#         replace(" 三氧化硫 "," SO3 ").replace(" 氧化锰 "," MnO2 ").replace(" 纳米氧化镁 "," MgO ").replace(" 二氧化硅基 "," SiO2 ").replace(" 氧化锡 "," SnO2 ").\
#         replace(" 纳米氧化铁 "," Fe2O3 ").replace(" 氧化镍 "," NiO ").replace(" 氧氯化锆 "," dichlorooxozirconium ").replace(" 氧化铜 "," CuO ").replace(" 氧化钇 "," Y2O3 ").\
#         replace(" 纳米氧化硅 "," SiO2 ").replace(" 五氧化二钒 "," V2O5 ").replace(" 纳米氧化镍 "," NiO ").replace(" 环氧丙烷 "," Propylene-oxide ").replace(" 二氧化氯 "," ClO2 ").\
#         replace(" 氧化钼 "," MoO3 ").replace(" 纳米氧化铈 "," CeO2 ").replace(" 氧化铍 "," BeO ").replace(" 氧化硼 "," B2O3 ").replace(" 氧化钽 "," Ta2O5 ").\
#         replace(" 二氧化硫 "," SO2 ").replace(" 氧化钆 "," Gd2O3 ").replace(" 氯氧化铋 "," BiClO ").replace(" 硅酸钠 "," Sodium-silicate ").replace(" 苯磺酸钠 "," Sodium-benzenesulfonate ").\
#         replace(" 丙烯酸钠 "," Sodium-acrylate ").replace(" 次氯酸钠 "," NaClO ").replace(" 硬脂酸钠 "," Sodium-stearate ").replace(" 硫酸钠 "," Na2SO4 ").replace(" 亚硫酸钠 "," Na2SO3 ").\
#         replace(" 高氯酸钠 "," NaClO4 ").replace(" 硝酸钠 "," NaNO3 ").replace(" 甲酸钠 "," HCOONa ").replace(" 氯化钠 "," NaCl ").replace(" 氯化镁 "," MgCl2 ").replace(" 六水氯化镁 "," MgCl2·6H2O ").\
#         replace(" 硫酸镁 "," MgSO4 ").replace(" 硅酸镁 "," MgSiO3 ").replace(" 醋酸镁 "," C4H6O4Mg ").replace(" 硝酸镁 "," Mg(NO3)2 ").replace(" 硅酸铝 "," Al2SiO5 ").replace(" 铝酸钙 "," Ca3Al2O6 ").\
#         replace(" 异丙醇铝 "," Aluminium isopropoxide ").replace(" 铝酸锶 "," SrAl2O4 ").replace(" 氯化铝 "," AlCl3 ").replace(" 硝酸铝 "," Al(NO3)3 ").replace(" 硫铝酸钙 "," Ca4Al6SO16 ").\
#         replace(" 硅酸乙酯 "," (C2H5O)4Si ").replace(" 硅酸钙 "," CaSiO3 ").replace(" 硅氧烷 "," Siloxane ").replace(" 硅酸钠 "," Na2O·nSiO2 ").replace(" 二硅化钼 "," MoSi2 ").replace(" 三氯硅烷 "," SiHCl3 ").\
#         replace(" 硫化硅 "," SiS2 ").replace(" 硅酸锌 "," Zn2SiO4 ").replace(" 硅酸锆 "," ZrSiO4 ").replace(" 磷酸 "," H3PO4 ").replace(" 聚磷酸铵 "," ammonium-polyphosphate ").\
#         replace(" 磷酸铁锂 "," LiFePO4 ").replace(" 磷腈 "," N3P3Cl6 ").replace(" 磷酸钒锂 "," Li3V2(PO4)3 ").replace(" 聚磷酸钙 "," CPP ").replace(" 甘油磷酸 "," glycerophosphoric-acid ").\
#         replace(" 磷酸亚铁锂 "," LiFePO4 ").replace(" 磷烯 "," Phosphorene ").replace(" 磷酸亚铁 "," Fe3(PO4)2 ").replace(" 磷酸钙 "," Ca3(PO4)2 ").replace(" 磷酸锂 "," Li3PO4 ").\
#         replace(" 纳米磷酸钙 "," Ca3(PO4)2 ").replace(" 磷酸锌 "," Zn3(PO4)2 ").replace(" 硫酸钙 "," CaSO4 ").replace(" 硫酸铵 "," (NH4)2SO4 ").replace(" 二硫化钼 "," MoS2 ").\
#         replace(" 聚苯硫醚 "," Polyphenylene-sulfide ").replace(" 硫醚 "," thioether ").replace(" 硫酸钡 "," BaSO4 ").replace(" 硫酸亚铁 "," FeSO4 ").replace(" 硫化镉 "," CdS ").\
#         replace(" 硫酸镍 "," NiSO4 ").replace(" 硫化钨 "," WS2 ").replace(" 二硫化钨 "," WS2 ").replace(" 硫代乙酰胺 "," Thioacetamide ").replace(" 硫酸锌 "," ZnSO4 ").\
#         replace(" 硫酸铅 "," PbSO4 ").replace(" 硫酸钴 "," CoSO4 ").replace(" 聚氯乙烯 "," PVC ").replace(" 氯化铵 "," NH4Cl ").replace(" 三氯化铁 "," FeCl3 ").replace(" 二氯甲烷 "," Dichloromethane ").\
#         replace(" 氯化钙 "," CaCl2 ").replace(" 氯仿 "," Trichloromethane ").replace(" 四氯化钛 "," TiCl4 ").replace(" 氯酚 "," monochlorophenol ").replace(" 氯化钾 "," KCl ").\
#         replace(" 二氯乙烯 "," Dichloroethene ").replace(" 氯气 "," Cl2 ").replace(" 三氯甲烷 "," Trichloromethane ").replace(" 氯乙烯 "," C2H3Cl ").replace(" 氯铂酸 "," H2PtCl6 ").replace(" 氯化铁 "," FeCl3 ").\
#         replace(" 氯苯 "," Chlorobenzene ").replace(" 高氯酸铵 "," NH4ClO4 ").replace(" 苯扎氯铵 "," Benzalkonium-chloride ").replace(" 四氯化锡 "," SnCl4 ").replace(" 氧氯化锆 "," ZrOCI2·8H2O ").\
#         replace(" 氯乙酸 "," Chloroacetic-acid ").replace(" 高氯酸 "," HClO4 ").replace(" 六水氯化钙 "," CaCl2·6H2O ").replace(" 氯化锌 "," ZnCl2 ").replace(" 氯铵 "," NH4Cl ").replace(" 氯化镉 "," CdCl2 ").\
#         replace(" 氯酸 "," HClO3 ").replace(" 六氯化钨 "," WCl6 ").replace(" 三氯乙烯 "," C2HCl3 ").replace(" 氩气 "," Ar ").replace(" 高锰酸钾 "," KMnO4 ").replace(" 六钛酸钾 "," K4O4Ti ").\
#         replace(" 钙钛矿 "," Perovskite ").replace(" 锆酸钙 "," CaZrO3 ").replace(" 硬脂酸钙 "," Calcium-stearate ").replace(" 钛酸钙 "," CaTiO3 ").replace(" 硝酸钙 "," Ca(NO3)2 ").replace(" 钛酸 "," titanic-acid ").\
#         replace(" 钛酸丁酯 "," Tetrabutyl-titanate ").replace(" 锆钛酸铅 "," PZT ").replace(" 钛酸钡 "," BaTiO₃ ").replace(" 钛酸异丙酯 "," Titanium-tetraisopropanolate ").replace(" 硝酸锰 "," Mn(NO3)2 ").\
#         replace(" 硝酸铁 "," Fe(NO3)3 ").replace(" 草酸亚铁 "," FeC2O4 ").replace(" 铁酸铋 "," BiFeO3 ").replace(" 铁酸锌 "," ZnFe2O4 ").replace(" 二茂铁甲酸 "," Ferrocenecarboxylic-acid ").\
#         replace(" 硝酸钴 "," Co(NO3)2 ").replace(" 硝酸镍 "," Ni(NO3)2 ").replace(" 硬脂酸镍 "," Nickel-stearate ").replace(" 紫铜 "," red-copper ").replace(" 黄铜 "," brass ").replace(" 硝酸铜 "," Cu(NO3)2 ").\
#         replace(" 硝酸锌 "," Zn(NO3)2 ").replace(" 硬脂酸锌 ","  Octadecanoic-acid ").replace(" 锆酸钙 "," CaZrO3 ").replace(" 醋酸锌 "," zinc-acetate ").replace(" 丙烯酸锌 "," Zinc-acrylate ").replace(" 硒化锌 "," ZnSe ").\
#         replace(" 砷化镓 "," GaAs ").replace(" 溴化铵 "," NH4Br ").replace(" 苯扎溴铵 "," BenzyldodecyldimethylammoniumBromide ").replace(" 钼酸 "," Molybdic-Acid ").replace(" 八钼酸铵 "," Ammoniummolybdat ").\
#         replace(" 硝酸银 "," AgNO3 ").replace(" 硬脂酸银 "," Octadecanoic-acid ").replace(" 碘仿 "," triiodomethane ").replace(" 钨酸 "," Tungstic-acid ").replace(" 硝酸铅 "," Pb(NO3)2 ").replace(" 硝酸铋 "," Bi(NO3)3 ").\
#         replace(" 光伏 ", " photovoltaic ")
#     f2.write(new_content)

'''
英文部分 8.将每个词单独成行放入txt中（统计词数）(区分中英文)
'''
# def delblankline(infile, outfile):
#     infopen = open(infile, 'r',encoding="utf-8")
#     outfopen = open(outfile, 'w',encoding="utf-8")
#     db = infopen.read()
#     outfopen.write(db.replace(' ','\n'))
#     infopen.close()
#     outfopen.close()
# delblankline(r"空格-半角-标点-简体-分词自定停止-数字-统一.txt",
#              r"每个词单独成行.txt")

# with open(r"每个词单独成行.txt", 'r', encoding="utf-8") as f1,open(r"每个词单独成行1.txt",'w',encoding="utf-8") as f2:
#     for line in f1.readlines():
#         if line.split():
#             f2.write(line)


'''
英文部分 9.英文小写、词形统一
'''
def is_not_en_word(word):
    '''
    判断一个词是否是非英文词,只要包含一个中文，就认为是非英文词汇
    :param word:
    :return:
    '''
    count = 0
    for s in word.encode('utf-8').decode('utf-8'):
        if u'\u4e00' <= s <= u'\u9fff':
            count += 1
            break
    if count > 0:
        return True
    else:
        return False


def is_en_mail(mail_text):
    '''
    判断一个词是否是非英文词,只要包含一个中文，就认为是非英文词汇
    :param word:
    :return:
    '''
    tmp_text = ''.join(mail_text.split())
    count = 0
    # print('tmp_text:', tmp_text)
    for s in tmp_text.encode('utf-8').decode('utf-8'):
        if u'\u4e00' <= s <= u'\u9fff':
            count += 1
    if float(count/(tmp_text.__len__())) > 0.1:
        return False
    else:
        return True
# aa = []
# with open(r"E:\中英\Chinese-English-PV-Test0-main\中文\processing\每个词单独成行1.txt", 'r', encoding="utf-8") as f1,open(r"E:\中英\Chinese-English-PV-Test0-main\中文\processing\空格-半角-标点-简体-分词自定停止-数字-统一-英文小写与统一.txt", "w",encoding='utf-8') as f2:
#     for line in f1.readlines():
#         if is_en_mail(mail_text=line):
#             a = text_processor.process(line)
#             if a[1] == []:
#                 line = line
#             else:
#                 line = a[1][0][1]
#         else:
#             line = line
#         aa.append(line)
#     print('*****')
#     print(aa)
#     for i in aa:
#         print(i)
#         sentence1 = i + ' '
#         f2.write(sentence1)

### 去除换行符
# with open(r"E:\中英\Chinese-English-PV-Test0-main\中文\processing\空格-半角-标点-简体-分词自定停止-数字-统一-英文小写与统一.txt", 'r', encoding="utf-8") as f1,open(r"E:\中英\Chinese-English-PV-Test0-main\中文\processing\空格-半角-标点-简体-分词自定停止-数字-统一-英文小写与统一1.txt", "w",encoding='utf-8') as f2:
#     document = f1.read()
#     line = re.sub('\n', '', document)
#     f2.write(line)

'''
英文部分 10.词性标注、词形还原
'''
# def get_wordnet_pos(tag):
#     if tag.startswith('J'):
#         return wordnet.ADJ
#     elif tag.startswith('V'):
#         return wordnet.VERB
#     elif tag.startswith('N'):
#         return wordnet.NOUN
#     elif tag.startswith('R'):
#         return wordnet.ADV
#     else:
#         return None
# with open(r"E:\中英\Chinese-English-PV-Test0-main\中文\processing\空格-半角-标点-简体-分词自定停止-数字-统一-英文小写与统一1.txt", "r",encoding='utf-8') as f,open(r"E:\中英\Chinese-English-PV-Test0-main\中文\processing\空格-半角-标点-简体-分词自定停止-数字-统一-英文小写与统一1-标注还原.txt", "w",encoding='utf-8') as f1:
#     content = f.read()
#     sentence = content
#     tokens = word_tokenize(sentence)  # 分词
#     tagged_sent = pos_tag(tokens)     # 获取单词词性
#
#     wnl = WordNetLemmatizer()
#     lemmas_sent = []
#     for tag in tagged_sent:
#         wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
#         lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos)) # 词形还原
#     print(lemmas_sent)
#     sentence1 = str()
#     for word in lemmas_sent:
#         print(word)
#         sentence1 = word + ' '
#         f1.write(sentence1)

'''
11.建立模型
'''

# sentences = word2vec.LineSentence(r"E:\中英\Chinese-English-PV-Test0-main\中文\空格-半角-标点-简体-分词自定停止-数字-统一-小写-标注还原.txt") #正式训练前的格式化
# model = word2vec.Word2Vec(sentences, min_count=3, vector_size=100, window=15,sg=0,sample=1e-3,epochs=5) #训练word2vec模型,设置超参数
# model.save(r"E:\中英\Chinese-English-PV-Test0-main\中文\CBOW-model-100-15.model")
# model.wv.save_word2vec_format(r"E:\中英\Chinese-English-PV-Test0-main\中文\CBOW-model-100-15.txt") #保存为模型类文件

'''
12.测试模型准确度（相关词与非相关词各150组）
'''
# model = KeyedVectors.load_word2vec_format(r"E:\中英\Chinese-English-PV-Test0-main\中文\pkuseg+CBOW+windows=15\CBOW-model-100-15.txt" ,binary= False) #以Word2vec文件形式打开刚刚保存的模型类文件
#
# dataset = r'E:\中英\Chinese-English-PV-Test0-main\中文\模型评估.xlsx'
# data = pd.DataFrame(pd.read_excel(dataset))
# first_word = data.values[:,10]
# second_word = data.values[:,11]
# for i in range(len(first_word)):
#     sim = model.similarity(first_word[i], second_word[i])
#     # print(first_word[i],second_word[i],sim)
#     print(sim)



'''
13.提取出模型中的所有单词并存入model_all_word.txt文件
'''
# sentence=str()
# with open(r"E:\中英\Chinese-English-PV-Test0-main\中文\pkuseg+skip-gram+windows=10\skip-gram-model-100-10.txt", 'r', encoding="utf-8") as f2,open(r"E:\中英\Chinese-English-PV-Test0-main\中文\pkuseg+skip-gram+windows=10\model-all-word.txt",'w',encoding="utf-8") as f3:
#     for line in f2:
#         print(line.split(' ')[0])
#         sentence = line.split(' ')[0] + '\n'
#         f3.write(sentence)

'''
14.利用chemdataextractor提取模型中包含的所有化学词汇
'''
# i = 0
# j = str()
# with open(r"E:\中英\Chinese-English-PV-Test0-main\中文\pkuseg+skip-gram+windows=10\model-all-word.txt", encoding='utf-8') as f1,open(r"E:\中英\Chinese-English-PV-Test0-main\中文\pkuseg+skip-gram+windows=10\model-all-chemdata.txt",'w',encoding='utf-8') as f2:
#      for line in f1.readlines():
#         doc = Document(line)
#         tokens = doc.cems
#         i += 1
#         if len(tokens) > 0:
#             # print(i,tokens)
#             # print(tokens[0])
#             m = str(tokens[0]) + '\n'
#             j += m
#      f2.write(j)

'''
15.利用余弦相似度 计算出与photovoltaic最相关的5000个词汇（未使用chemdata） 并导入photovoltaic-5000.xlsx文件
'''
# model = KeyedVectors.load_word2vec_format('English-model.txt' ,binary= False) #以Word2vec文件形式打开刚刚保存的模型类文件
# Me = model.most_similar('photovoltaic',topn=5000) #与' '最相似的单词
#
# dataset = 'photovoltaic-5000.xlsx'
# data = pd.DataFrame(pd.read_excel(dataset))
# word = data.values[:,0]
# print(word)

'''
16.利用chemdataextractor 从第9步中的5000个词中按排名提取出所以化学词汇 并导入photovoltaic-5000-chemdata.xlsx文件中
'''
# model = KeyedVectors.load_word2vec_format('English-model-100-5.txt' ,binary= False) #以Word2vec文件形式打开刚刚保存的模型类文件
# Me = model.most_similar('photovoltaic',topn=5000) #与' '最相似的单词
#
# aa = []
# lines = ""
# # 将 model_all_chemdata.txt复制到 model_all_chemdata.xlsx里的第一列
# dataset = r'model_all_chemdata.xlsx'
# data = pd.DataFrame(pd.read_excel(dataset))
# first_word = data.values[:,0]
# print(first_word)
# file = open(r'photovoltaic-sim-5000.csv', mode='w', encoding='utf-8')
# for i in Me:
#     if i[0] in first_word:
#         print(i[0],i[1])
#         lines = i[0] + ", " # 化学式
#         # lines = str(i[1]) + ", " # 余弦相似度
#         lines += "\n"
#         file.write(lines)

































'''
17.将每个词单独成行放入txt中（统计词数）
'''
# def delblankline(infile, outfile):
#     infopen = open(infile, 'r',encoding="utf-8")
#     outfopen = open(outfile, 'w',encoding="utf-8")
#     db = infopen.read()
#     outfopen.write(db.replace(' ','\n'))
#     infopen.close()
#     outfopen.close()
# delblankline(r"E:\中英\Chinese-English-PV-Test0-main\中文\WF-CNKI-去空格-分词-停止词-自定义.txt",
#              r"E:\中英\Chinese-English-PV-Test0-main\中文\WF-CNKI-去空格-分词-停止词-自定义-.txt")

# with open(r"E:\中英\Chinese-English-PV-Test0-main\中文\WF-CNKI-去空格-分词-停止词-自定义-.txt", 'r', encoding="utf-8") as f1,open(r"E:\中英\Chinese-English-PV-Test0-main\中文\WF-CNKI-去空格-分词-停止词-自定义-换行.txt",'w',encoding="utf-8") as f2:
#     for line in f1.readlines():
#         if line.split():
#                 f2.write(line)