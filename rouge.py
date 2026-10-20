"""
Borrowed from https://github.com/tensorflow/nmt/blob/master/nmt/scripts/rouge.py

ROUGE metric implementation.

Copy from tf_seq2seq/seq2seq/metrics/rouge.py.
This is a modified and slightly extended verison of
https://github.com/miso-belica/sumy/blob/dev/sumy/evaluation/rouge.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import itertools
import numpy as np

#pylint: disable=C0103


def _get_ngrams(n, text):
  """Calcualtes n-grams.

  Args:
    n: which n-grams to calculate
    text: An array of tokens

  Returns:
    A set of n-grams
  """
  ngram_set = set()
  text_length = len(text)
  #計算在預測的文本中針對給定的n-gram可能的組數
  max_index_ngram_start = text_length - n
  for i in range(max_index_ngram_start + 1):
    #分別擷取文字片段的gram(數量由n決定)，並以tuple型態儲存，再加到ngram_set
    ngram_set.add(tuple(text[i:i + n]))
  #回傳每一個n-gram的結果
  return ngram_set


#把句子逐字拆分成word，形狀為['a','b','c']
def _split_into_words(sentences):
  """Splits multiple sentences into words and flattens the result"""
  return list(itertools.chain(*[_.split(" ") for _ in sentences]))


def _get_word_ngrams(n, sentences):
  """Calculates word n-grams for multiple sentences.
  """
  #檢查句子與n-gram是否皆大於0，否則中斷程式
  assert len(sentences) > 0
  assert n > 0

  #把句子拆分為word計算n-gram
  words = _split_into_words(sentences)
  return _get_ngrams(n, words)


#(拆分成word的模型預測的文本, 拆分成word的ground truth的文本)
def _len_lcs(x, y):
  """
  Returns the length of the Longest Common Subsequence between sequences x
  and y.
  Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

  Args:
    x: sequence of words
    y: sequence of words

  Returns
    integer: Length of LCS between x and y
  """
  table = _lcs(x, y)
  n, m = len(x), len(y)
  return table[n, m]


#(拆分成word的模型預測的文本, 拆分成word的ground truth的文本)
def _lcs(x, y):
  """
  Computes the length of the longest common subsequence (lcs) between two
  strings. The implementation below uses a DP programming algorithm and runs
  in O(nm) time where n = len(x) and m = len(y).
  Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

  Args:
    x: collection of words
    y: collection of words

  Returns:
    Table of dictionary of coord and len lcs
  """
  #計算拆分成word的模型預測的文本和拆分成word的ground truth的文本的長度
  n, m = len(x), len(y)
  #初始劃一個字典用來儲存lcs長度
  table = dict()
  for i in range(n + 1):
    for j in range(m + 1):
      #若有一個字串為0，則lcs長度為0
      if i == 0 or j == 0:
        table[i, j] = 0
      #如果兩個字串match，則lcs長度加1
      elif x[i - 1] == y[j - 1]:
        table[i, j] = table[i - 1, j - 1] + 1
      #否則，取table[i-1,j]和table[i,j-1]中的最大值，這代表忽略當前字串後的最大lcs長度
      else:
        table[i, j] = max(table[i - 1, j], table[i, j - 1])
  #回傳lcs的長度
  return table


#(拆分成word的模型預測的文本, 拆分成word的ground truth的文本)
def _recon_lcs(x, y):
  """
  Returns the Longest Subsequence between x and y.
  Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

  Args:
    x: sequence of words
    y: sequence of words

  Returns:
    sequence: LCS of x and y
  """
  #計算拆分成word的模型預測的文本和拆分成word的ground truth的文本的長度
  i, j = len(x), len(y)
  #計算拆分成word的模型預測的文本和拆分成word的ground truth的文本的最長公共子序列(lcs)的長度
  table = _lcs(x, y)

  def _recon(i, j):
    """private recon calculation"""
    #如果有一個字串為0，則回傳空list
    if i == 0 or j == 0:
      return []
    #如果x[i-1]等於y[j-1]，則把這個match的word加入到結果中，並繼續遞歸計算前一個位置的lcs
    elif x[i - 1] == y[j - 1]:
      return _recon(i - 1, j - 1) + [(x[i - 1], i)]
    #如果table[i-1,j]大於table[i,j-1]，則遞歸計算去掉x中目前word的lcs
    elif table[i - 1, j] > table[i, j - 1]:
      return _recon(i - 1, j)
    #否則，遞歸計算去掉y中目前word的lcs
    else:
      return _recon(i, j - 1)

  #用map提取每個match word的word部分(忽略index)，再轉換為tuple
  recon_tuple = tuple(map(lambda x: x[0], _recon(i, j)))
  #回傳lcs的結果
  return recon_tuple


#(模型預測的文本, ground truth的文本, n-gram的數量)
def rouge_n(evaluated_sentences, reference_sentences, n=2):
  """
  Computes ROUGE-N of two text collections of sentences.
  Sourece: http://research.microsoft.com/en-us/um/people/cyl/download/
  papers/rouge-working-note-v1.3.1.pdf

  Args:
    evaluated_sentences: The sentences that have been picked by the summarizer
    reference_sentences: The sentences from the referene set
    n: Size of ngram.  Defaults to 2.

  Returns:
    A tuple (f1, precision, recall) for ROUGE-N

  Raises:
    ValueError: raises exception if a param has len <= 0
  """
  #如果模型預測的文本或ground truth的文本小於等於0，則報錯
  if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
    raise ValueError("Collections must contain at least 1 sentence.")

  #計算模型預測文本的n-gram
  evaluated_ngrams = _get_word_ngrams(n, evaluated_sentences)
  #計算ground truth文本的n-gram
  reference_ngrams = _get_word_ngrams(n, reference_sentences)
  #計算ground truth文本的長度
  reference_count = len(reference_ngrams)
  #計算模型預測文本的長度
  evaluated_count = len(evaluated_ngrams)

  # Gets the overlapping ngrams between evaluated and reference
  #計算兩個集合都包含的元素，也就是ground truth與預測結果重複的n-gram
  overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
  #計算ground truth與預測結果重疊的n-gram的長度
  overlapping_count = len(overlapping_ngrams)

  # Handle edge case. This isn't mathematically correct, but it's good enough
  #判斷模型預測文本的長度是否為0(以免分母為0)，若是，則precision也為0；否則，precision為重疊的n-gram的長度在模型預測文本的長度的占比率
  if evaluated_count == 0:
    precision = 0.0
  else:
    precision = overlapping_count / evaluated_count

  #判斷ground truth文本的長度是否為0(以免分母為0)，若是，則recall也為0；否則，recall為重疊的n-gram的長度在ground truth文本的長度的占比率
  if reference_count == 0:
    recall = 0.0
  else:
    recall = overlapping_count / reference_count

  #計算f1-socre，分母加上極小數(1e-8)避免為分母為0
  f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))

  # return overlapping_count / reference_count
  return f1_score, precision, recall


def _f_p_r_lcs(llcs, m, n):
  """
  Computes the LCS-based F-measure score
  Source: http://research.microsoft.com/en-us/um/people/cyl/download/papers/
  rouge-working-note-v1.3.1.pdf

  Args:
    llcs: Length of LCS
    m: number of words in reference summary
    n: number of words in candidate summary

  Returns:
    Float. LCS-based F-measure score
  """
  #計算lcs長度在ground truth文本長度的占比，也就是計算在ground truth文本中有多少word跟模型預測的文本match
  r_lcs = llcs / m
  #計算lcs長度在模型預測文本長度的占比，也就是計算在模型預測文本中有多少word跟ground truth的文本match
  p_lcs = llcs / n
  #beta是precision和recall之間的比例，用來加權F-measure計算，分母加上1e-12是為了防止分母為0
  beta = p_lcs / (r_lcs + 1e-12)
  #計算F-score分數
  num = (1 + (beta**2)) * r_lcs * p_lcs
  denom = r_lcs + ((beta**2) * p_lcs)
  f_lcs = num / (denom + 1e-12)
  return f_lcs, p_lcs, r_lcs


#(模型預測的文本, ground truth的文本)
def rouge_l_sentence_level(evaluated_sentences, reference_sentences):
  """
  Computes ROUGE-L (sentence level) of two text collections of sentences.
  http://research.microsoft.com/en-us/um/people/cyl/download/papers/
  rouge-working-note-v1.3.1.pdf

  Calculated according to:
  R_lcs = LCS(X,Y)/m
  P_lcs = LCS(X,Y)/n
  F_lcs = ((1 + beta^2)*R_lcs*P_lcs) / (R_lcs + (beta^2) * P_lcs)

  where:
  X = reference summary
  Y = Candidate summary
  m = length of reference summary
  n = length of candidate summary

  Args:
    evaluated_sentences: The sentences that have been picked by the summarizer
    reference_sentences: The sentences from the referene set

  Returns:
    A float: F_lcs

  Raises:
    ValueError: raises exception if a param has len <= 0
  """
  #如果模型預測的文本或ground truth的文本小於等於0，則報錯
  if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
    raise ValueError("Collections must contain at least 1 sentence.")
  #把句子拆分為word計算n-gram
  reference_words = _split_into_words(reference_sentences)
  evaluated_words = _split_into_words(evaluated_sentences)
  #計算ground truth文本的長度
  m = len(reference_words)
  #計算模型預測文本的長度
  n = len(evaluated_words)
  
  #計算模型預測的文本和ground truth的文本的最長公共子序列(lcs)
  lcs = _len_lcs(evaluated_words, reference_words)
  #回傳F-score、precision和recall
  return _f_p_r_lcs(lcs, m, n)


def _union_lcs(evaluated_sentences, reference_sentence):
  """
  Returns LCS_u(r_i, C) which is the LCS score of the union longest common
  subsequence between reference sentence ri and candidate summary C. For example
  if r_i= w1 w2 w3 w4 w5, and C contains two sentences: c1 = w1 w2 w6 w7 w8 and
  c2 = w1 w3 w8 w9 w5, then the longest common subsequence of r_i and c1 is
  "w1 w2" and the longest common subsequence of r_i and c2 is "w1 w3 w5". The
  union longest common subsequence of r_i, c1, and c2 is "w1 w2 w3 w5" and
  LCS_u(r_i, C) = 4/5.

  Args:
    evaluated_sentences: The sentences that have been picked by the summarizer
    reference_sentence: One of the sentences in the reference summaries

  Returns:
    float: LCS_u(r_i, C)

  ValueError:
    Raises exception if a param has len <= 0
  """
  #如果模型預測的文本小於等於0，則報錯
  if len(evaluated_sentences) <= 0:
    raise ValueError("Collections must contain at least 1 sentence.")

  #初始化一個空集合，用來儲存聯集的lcs
  lcs_union = set()
  #把ground truth的文本拆分為word
  reference_words = _split_into_words([reference_sentence])
  #用來累加所有lcs的總長度
  combined_lcs_length = 0
  for eval_s in evaluated_sentences:
    #把模型預測的文本分別拆分為word
    evaluated_words = _split_into_words([eval_s])
    #lcs的結果
    lcs = set(_recon_lcs(reference_words, evaluated_words))
    #計算lcs的長度並加總到combined_lcs_length
    combined_lcs_length += len(lcs)
    #把lcs_union和lcs進行聯集
    lcs_union = lcs_union.union(lcs)

  #計算lcs_union的元素數量
  union_lcs_count = len(lcs_union)
  #計算聯集lcs的值，用lcs_union的元素數量除以lcs的總長度
  union_lcs_value = union_lcs_count / combined_lcs_length
  return union_lcs_value


def rouge_l_summary_level(evaluated_sentences, reference_sentences):
  """
  Computes ROUGE-L (summary level) of two text collections of sentences.
  http://research.microsoft.com/en-us/um/people/cyl/download/papers/
  rouge-working-note-v1.3.1.pdf

  Calculated according to:
  R_lcs = SUM(1, u)[LCS<union>(r_i,C)]/m
  P_lcs = SUM(1, u)[LCS<union>(r_i,C)]/n
  F_lcs = ((1 + beta^2)*R_lcs*P_lcs) / (R_lcs + (beta^2) * P_lcs)

  where:
  SUM(i,u) = SUM from i through u
  u = number of sentences in reference summary
  C = Candidate summary made up of v sentences
  m = number of words in reference summary
  n = number of words in candidate summary

  Args:
    evaluated_sentences: The sentences that have been picked by the summarizer
    reference_sentence: One of the sentences in the reference summaries

  Returns:
    A float: F_lcs

  Raises:
    ValueError: raises exception if a param has len <= 0
  """
  #如果模型預測的文本或ground truth的文本小於等於0，則報錯
  if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
    raise ValueError("Collections must contain at least 1 sentence.")

  # total number of words in reference sentences
  #計算ground truth文本中word的長度
  m = len(_split_into_words(reference_sentences))

  # total number of words in evaluated sentences
  #計算模型預測文本中word的長度
  n = len(_split_into_words(evaluated_sentences))

  union_lcs_sum_across_all_references = 0
  for ref_s in reference_sentences:
    #計算模型預測的文本和一個ground truth句子之間的聯集lcs的比例值
    union_lcs_sum_across_all_references += _union_lcs(evaluated_sentences, ref_s)
  #回傳F-score、precision和recall
  return _f_p_r_lcs(union_lcs_sum_across_all_references, m, n)


#(串起來的模型預測文本序列, 串起來的ground truth文本序列)
def rouge(hypotheses, references):
  """Calculates average rouge scores for a list of hypotheses and
  references"""

  # Filter out hyps that are of 0 length
  # hyps_and_refs = zip(hypotheses, references)
  # hyps_and_refs = [_ for _ in hyps_and_refs if len(_[0]) > 0]
  # hypotheses, references = zip(*hyps_and_refs)

  # Calculate ROUGE-1 F1, precision, recall scores
  #計算rouge-1的F1-score、precision和recall
  rouge_1 = [
      rouge_n([hyp], [ref], 1) for hyp, ref in zip(hypotheses, references)
  ]
  #分別計算平均值
  rouge_1_f, rouge_1_p, rouge_1_r = map(np.mean, zip(*rouge_1))

  # Calculate ROUGE-2 F1, precision, recall scores
  #計算rouge-2的F1-score、precision和recall
  rouge_2 = [
      rouge_n([hyp], [ref], 2) for hyp, ref in zip(hypotheses, references)
  ]
  rouge_2_f, rouge_2_p, rouge_2_r = map(np.mean, zip(*rouge_2))

  # Calculate ROUGE-L F1, precision, recall scores
  #計算rouge-l的F1-score、precision和recall
  rouge_l = [
      rouge_l_sentence_level([hyp], [ref])
      for hyp, ref in zip(hypotheses, references)
  ]
  rouge_l_f, rouge_l_p, rouge_l_r = map(np.mean, zip(*rouge_l))

  return {
      "rouge_1/f_score": rouge_1_f,
      "rouge_1/r_score": rouge_1_r,
      "rouge_1/p_score": rouge_1_p,
      "rouge_2/f_score": rouge_2_f,
      "rouge_2/r_score": rouge_2_r,
      "rouge_2/p_score": rouge_2_p,
      "rouge_l/f_score": rouge_l_f,
      "rouge_l/r_score": rouge_l_r,
      "rouge_l/p_score": rouge_l_p,
  }
