import os
import math
import re
import unicodedata
from collections import Counter
import jieba
import matplotlib.pyplot as plt

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_stopwords_from_file(file_path):
    # 从文件加载停用词并去除首尾空格
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f if line.strip())
    except Exception as e:
        print(f"无法加载停用词文件 {file_path}: {str(e)}")
        return set()

def read_corpus_files(root_directory):
    # 遍历语料库目录，读取每个文件的内容并生成文本。
    for foldername, _, filenames in os.walk(root_directory):
        for filename in filenames:
            file_path = os.path.join(foldername, filename)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    yield f.read()
            except Exception as e:
                print(f"读取文件 {file_path} 时出错: {str(e)}")

def clean_text_for_statistics(text):
    # 清洗文本以进行字符统计
    text = re.sub(r'[\n\r\t]', '', text)  # 删除换行、回车、制表符
    text = re.sub(r'[\s\u3000]', '', text)  # 删除所有空格
    text = re.sub(r'[A-Za-z0-9=]', '', text)  # 删除字母和数字
    text = ''.join(ch for ch in text if not unicodedata.category(ch).startswith('P'))  # 删除标点
    return text

def prepare_text_for_segmentation(text):
    # 对文本进行轻度清洗以保留标点，便于分词
    return re.sub(r'[\n\r\t]', '', text)

def filter_words_by_stopwords(words, stopwords):
    # 过滤分词结果，去除停用词和无效词
    return [word for word in words if word not in stopwords and word.strip()]

def plot_frequency_distribution(counter, title="频率分布图"):
    # 绘制频率分布图
    sorted_freq = sorted(counter.values(), reverse=True)
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_freq)
    plt.yscale('log')
    plt.title(title)
    plt.xlabel("排名")
    plt.ylabel("频率 (对数坐标)")
    plt.show()

def main(corpus_directory, stopwords_file):
    # 主函数，处理语料库并计算信息熵
    stopwords = load_stopwords_from_file(stopwords_file)
    char_counter = Counter()
    word_counter = Counter()
    total_filtered_chars = 0
    total_filtered_words = 0

    file_count = 0
    for text in read_corpus_files(corpus_directory):
        file_count += 1
        print(f"处理文件 {file_count}，字符数: {len(text)}")
        
        # 字符统计处理
        cleaned_text = clean_text_for_statistics(text)
        filtered_chars = ''.join(ch for ch in cleaned_text if ch not in stopwords)
        char_counter.update(filtered_chars)
        total_filtered_chars += len(filtered_chars)
        
        # 分词处理
        seg_text = prepare_text_for_segmentation(text)
        words = jieba.lcut(seg_text)
        filtered_words = filter_words_by_stopwords(words, stopwords)
        word_counter.update(filtered_words)
        total_filtered_words += len(filtered_words)
    
    # 计算字符信息熵
    char_entropy = -sum((count / total_filtered_chars) * math.log2(count / total_filtered_chars) for count in char_counter.values()) if total_filtered_chars > 0 else 0.0
    print(f"字符信息熵: {char_entropy:.4f} bits/char")
    
    # 计算词信息熵
    word_entropy = -sum((count / total_filtered_words) * math.log2(count / total_filtered_words) for count in word_counter.values()) if total_filtered_words > 0 else 0.0
    print(f"词信息熵: {word_entropy:.4f} bits/word")
    
    # 绘制分布图
    plot_frequency_distribution(char_counter, title="字符频率分布")
    plot_frequency_distribution(word_counter, title="词频率分布")

    # 找出字频最高的字符及其频率
    if char_counter:
        most_common_char, most_common_char_count = char_counter.most_common(1)[0]
        print(f"字频最高的字符是: '{most_common_char}'，出现了 {most_common_char_count} 次")
    else:
        print("未统计到任何字符")

if __name__ == "__main__":
    CORPUS_DIR = "C:/Users/wjl/Documents/dlnlp/DLnlp-1/wiki_zh"  # 语料库根目录
    STOPWORDS_FILE = "C:/Users/wjl/Documents/dlnlp/DLnlp-1/cn_stopwords.txt"  # 停用词文件
    main(CORPUS_DIR, STOPWORDS_FILE)