"""
Borrowed from https://github.com/tensorflow/nmt/blob/master/nmt/scripts/bleu.py

Python implementation of BLEU and smooth-BLEU.

This module provides a Python implementation of BLEU and smooth-BLEU.
Smooth BLEU is computed following the method outlined in the paper:
Chin-Yew Lin, Franz Josef Och. ORANGE: a method for evaluating automatic
evaluation metrics for machine translation. COLING 2004.
"""

import collections
import math


def _get_ngrams(segment, max_order):
  """Extracts all n-grams upto a given maximum order from an input segment.

  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.

  Returns:
    The Counter containing all n-grams upto max_order in segment
    with a count of how many times each n-gram occurred.
  """
  #創建一個字典型態的計數器，用來計算可迭代物件中每個物件的數量
  ngram_counts = collections.Counter()
  for order in range(1, max_order + 1):
    for i in range(0, len(segment) - order + 1):
      #tuple類似list，但它使用小括號且後面加逗號，另外建立後就不能修改內容
      #分別擷取文字片段的gram(數量由max_order決定)，並以tuple型態儲存
      ngram = tuple(segment[i:i+order])
      #gram的count加1，且由大到小排序
      ngram_counts[ngram] += 1
  #回傳每一個n-gram的結果
  return ngram_counts

# gn = _get_ngrams("These is", 2)
# print(gn)

#smooth策略是指當n>1時，將1個count加到重疊的n-gram和總ngram的count中
#reference_corpus是ground truth；translation_corpus是模型預測的文本
def compute_bleu(reference_corpus, translation_corpus, max_order=4, smooth=False):
  """Computes BLEU score of translated segments against one or more references.

  Args:
    reference_corpus: list of lists of references for each translation. Each
        reference should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation
        should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    smooth: Whether or not to apply Lin et al. 2004 smoothing.

  Returns:
    3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
    precisions and brevity penalty.
  """
  #初始化一個全為0的list，用來存重疊的文本中每個n-gram的數量，也就是公式BLEU的分子
  matches_by_order = [0] * max_order
  #用來存預測的文本中針對給定的n-gram可能的組數，也就是公式BLEU的分母
  possible_matches_by_order = [0] * max_order
  reference_length = 0
  translation_length = 0
  for (references, translation) in zip(reference_corpus, translation_corpus):
    #加總所有最短ground truth的長度
    #如果預測文本比ground truth短會被懲罰，因此選最短的ground truth計算懲罰，讓結果比較保守
    reference_length += min(len(r) for r in references)
    #加總所有預測文本的長度
    translation_length += len(translation)

    #用來儲存所有ground truth的n-gram和count合併的結果
    merged_ref_ngram_counts = collections.Counter()
    for reference in references:
      #|=表只儲存不重複的n-gram
      merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
    #用來儲存預測文本的n-gram和count的結果
    translation_ngram_counts = _get_ngrams(translation, max_order)
    #計算ground truth與預測結果重複的n-gram與次數
    overlap = translation_ngram_counts & merged_ref_ngram_counts
    #累加重疊的文本中每個n-gram的數量，ex:n-gram=2加到matches_by_order[1]中
    for ngram in overlap:
      matches_by_order[len(ngram)-1] += overlap[ngram]
    
    for order in range(1, max_order+1):
      #計算在預測的文本中針對給定的n-gram可能的組數，ex:文本為the movie is good.且n-gram為2，則possible_matches則是3(the movie, movie is, is good)
      possible_matches = len(translation) - order + 1
      if possible_matches > 0:
        possible_matches_by_order[order-1] += possible_matches

  precisions = [0] * max_order
  for i in range(0, max_order):
    #判斷是否使用smooth技術，若是則分子與分母都加1
    if smooth:
      precisions[i] = ((matches_by_order[i] + 1.) / (possible_matches_by_order[i] + 1.))
    else:
      if possible_matches_by_order[i] > 0:
        precisions[i] = (float(matches_by_order[i]) / possible_matches_by_order[i])
      else:
        precisions[i] = 0.0

  #判斷是否所有n-gram的precision都大於0，若是則計算幾何平均；若否則跳過並給定為0
  if min(precisions) > 0:
    #對每個n-gram的precision取log，然後算它們的平均值
    p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
    #再把log平均值轉回原始空間，得到幾何平均值
    geo_mean = math.exp(p_log_sum)
  else:
    geo_mean = 0

  #計算ground truth和模型預測文本長度的比例，用以計算簡短懲罰
  ratio = float(translation_length) / reference_length

  #如果比例大於1，表示模型預測文本的長度較長，則懲罰較大
  if ratio > 1.0:
    bp = 1.
  else:
    bp = math.exp(1 - 1. / ratio)

  bleu = geo_mean * bp

  #回傳bleu分數、各個n-gram順序的precisions、簡短懲罰、ground truth和模型預測文本長度的比例、模型預測文本的總長度、ground truth的總長度
  return (bleu, precisions, bp, ratio, translation_length, reference_length)
