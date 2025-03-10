import nltk
import math
import re
from collections import Counter
from nltk.corpus import gutenberg
from nltk.tokenize import word_tokenize
import string
import matplotlib.pyplot as plt

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def calculate_entropy(frequencies):
    # 计算信息熵
    total_count = sum(frequencies.values())
    entropy = 0
    for count in frequencies.values():
        probability = count / total_count
        entropy -= probability * math.log2(probability)
    return entropy


def process_english_text(text):
    # 转换为小写
    text = text.lower()
    
    # 计算字母级别的频率（只考虑字母）
    letters = [c for c in text if c in string.ascii_lowercase]
    letter_frequencies = Counter(letters)
    letter_entropy = calculate_entropy(letter_frequencies)
    
    # 分词并计算词级别的频率
    words = word_tokenize(text)
    # 移除标点符号和数字
    words = [word for word in words if word.isalpha()]
    word_frequencies = Counter(words)
    word_entropy = calculate_entropy(word_frequencies)
    
    return letter_entropy, word_entropy, letter_frequencies, word_frequencies


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


def main():
    # 下载必要的NLTK资源
    nltk.download('gutenberg')
    nltk.download('punkt')
    
    # 读取Gutenberg语料库中的所有文本
    corpus_text = ""
    for fileid in gutenberg.fileids():
        corpus_text += gutenberg.raw(fileid)
        if len(corpus_text) > 1000000:  # 限制处理的文本量
            break
    
    # 处理文本
    letter_entropy, word_entropy, letter_frequencies, word_frequencies = process_english_text(corpus_text)
    
    # 输出信息熵
    print(f"英文字母信息熵: {letter_entropy:.2f} bits")
    print(f"英文单词信息熵: {word_entropy:.2f} bits")
    
    # 绘制频率分布图
    plot_frequency_distribution(letter_frequencies, title="字母频率分布")
    plot_frequency_distribution(word_frequencies, title="单词频率分布")

if __name__ == "__main__":
    main()