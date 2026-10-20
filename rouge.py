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
  # number of n-gram start positions in the text
  max_index_ngram_start = text_length - n
  for i in range(max_index_ngram_start + 1):
    # add every n-gram of length n (as a tuple) to ngram_set
    ngram_set.add(tuple(text[i:i + n]))
  # return the set of n-grams
  return ngram_set


# split sentences into a flat list of words, e.g. ['a', 'b', 'c']
def _split_into_words(sentences):
  """Splits multiple sentences into words and flattens the result"""
  return list(itertools.chain(*[_.split(" ") for _ in sentences]))


def _get_word_ngrams(n, sentences):
  """Calculates word n-grams for multiple sentences.
  """
  # require at least one sentence and n > 0
  assert len(sentences) > 0
  assert n > 0

  # split the sentences into words and compute their n-grams
  words = _split_into_words(sentences)
  return _get_ngrams(n, words)


# x: tokenized generated text, y: tokenized reference
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


# x: tokenized generated text, y: tokenized reference
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
  # lengths of the two token sequences
  n, m = len(x), len(y)
  # DP table mapping (i, j) to the LCS length of x[:i] and y[:j]
  table = dict()
  for i in range(n + 1):
    for j in range(m + 1):
      # an empty prefix has LCS length 0
      if i == 0 or j == 0:
        table[i, j] = 0
      # matching tokens extend the LCS by 1
      elif x[i - 1] == y[j - 1]:
        table[i, j] = table[i - 1, j - 1] + 1
      # otherwise take the better of dropping the last token of x or of y
      else:
        table[i, j] = max(table[i - 1, j], table[i, j - 1])
  # return the full DP table
  return table


# x: tokenized generated text, y: tokenized reference
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
  # lengths of the two token sequences
  i, j = len(x), len(y)
  # DP table of LCS lengths
  table = _lcs(x, y)

  def _recon(i, j):
    """private recon calculation"""
    # an empty prefix yields an empty LCS
    if i == 0 or j == 0:
      return []
    # matching tokens belong to the LCS; recurse on the shorter prefixes
    elif x[i - 1] == y[j - 1]:
      return _recon(i - 1, j - 1) + [(x[i - 1], i)]
    # otherwise follow the larger subproblem: drop the last token of x
    elif table[i - 1, j] > table[i, j - 1]:
      return _recon(i - 1, j)
    # ... or drop the last token of y
    else:
      return _recon(i, j - 1)

  # keep only the words (drop the positions) and convert to a tuple
  recon_tuple = tuple(map(lambda x: x[0], _recon(i, j)))
  # return the LCS
  return recon_tuple


# arguments: generated sentences, reference sentences, n-gram order
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
  # both collections must be non-empty
  if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
    raise ValueError("Collections must contain at least 1 sentence.")

  # n-grams of the generated text
  evaluated_ngrams = _get_word_ngrams(n, evaluated_sentences)
  # n-grams of the reference
  reference_ngrams = _get_word_ngrams(n, reference_sentences)
  # number of distinct reference n-grams
  reference_count = len(reference_ngrams)
  # number of distinct generated n-grams
  evaluated_count = len(evaluated_ngrams)

  # Gets the overlapping ngrams between evaluated and reference
  overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
  # number of overlapping n-grams
  overlapping_count = len(overlapping_ngrams)

  # Handle edge case. This isn't mathematically correct, but it's good enough
  # precision = overlap / generated n-grams (0 if there are none)
  if evaluated_count == 0:
    precision = 0.0
  else:
    precision = overlapping_count / evaluated_count

  # recall = overlap / reference n-grams (0 if there are none)
  if reference_count == 0:
    recall = 0.0
  else:
    recall = overlapping_count / reference_count

  # F1; the 1e-8 term avoids division by zero
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
  # recall: share of reference words covered by the LCS
  r_lcs = llcs / m
  # precision: share of generated words covered by the LCS
  p_lcs = llcs / n
  # beta = P/R weights the F-measure; 1e-12 avoids division by zero
  beta = p_lcs / (r_lcs + 1e-12)
  # F-measure
  num = (1 + (beta**2)) * r_lcs * p_lcs
  denom = r_lcs + ((beta**2) * p_lcs)
  f_lcs = num / (denom + 1e-12)
  return f_lcs, p_lcs, r_lcs


# arguments: generated sentences, reference sentences
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
  # both collections must be non-empty
  if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
    raise ValueError("Collections must contain at least 1 sentence.")
  # split the sentences into words
  reference_words = _split_into_words(reference_sentences)
  evaluated_words = _split_into_words(evaluated_sentences)
  # reference length in words
  m = len(reference_words)
  # generated length in words
  n = len(evaluated_words)
  
  # LCS length between generated text and reference
  lcs = _len_lcs(evaluated_words, reference_words)
  # return F-score, precision and recall
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
  # the generated collection must be non-empty
  if len(evaluated_sentences) <= 0:
    raise ValueError("Collections must contain at least 1 sentence.")

  # union of the LCS words over all generated sentences
  lcs_union = set()
  # split the reference sentence into words
  reference_words = _split_into_words([reference_sentence])
  # sum of the individual LCS lengths
  combined_lcs_length = 0
  for eval_s in evaluated_sentences:
    # split the generated sentence into words
    evaluated_words = _split_into_words([eval_s])
    # LCS between the reference and this generated sentence
    lcs = set(_recon_lcs(reference_words, evaluated_words))
    # add its length to combined_lcs_length
    combined_lcs_length += len(lcs)
    # add its words to the union
    lcs_union = lcs_union.union(lcs)

  # size of the union LCS
  union_lcs_count = len(lcs_union)
  # union LCS size divided by the combined LCS length
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
  # both collections must be non-empty
  if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
    raise ValueError("Collections must contain at least 1 sentence.")

  # total number of words in reference sentences
  m = len(_split_into_words(reference_sentences))

  # total number of words in evaluated sentences
  n = len(_split_into_words(evaluated_sentences))

  union_lcs_sum_across_all_references = 0
  for ref_s in reference_sentences:
    # add the union-LCS score of the generated text against this reference sentence
    union_lcs_sum_across_all_references += _union_lcs(evaluated_sentences, ref_s)
  # return F-score, precision and recall
  return _f_p_r_lcs(union_lcs_sum_across_all_references, m, n)


# arguments: list of generated texts, list of reference texts
def rouge(hypotheses, references):
  """Calculates average rouge scores for a list of hypotheses and
  references"""

  # Filter out hyps that are of 0 length
  # hyps_and_refs = zip(hypotheses, references)
  # hyps_and_refs = [_ for _ in hyps_and_refs if len(_[0]) > 0]
  # hypotheses, references = zip(*hyps_and_refs)

  # Calculate ROUGE-1 F1, precision, recall scores
  rouge_1 = [
      rouge_n([hyp], [ref], 1) for hyp, ref in zip(hypotheses, references)
  ]
  # average over all pairs
  rouge_1_f, rouge_1_p, rouge_1_r = map(np.mean, zip(*rouge_1))

  # Calculate ROUGE-2 F1, precision, recall scores
  rouge_2 = [
      rouge_n([hyp], [ref], 2) for hyp, ref in zip(hypotheses, references)
  ]
  rouge_2_f, rouge_2_p, rouge_2_r = map(np.mean, zip(*rouge_2))

  # Calculate ROUGE-L F1, precision, recall scores
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
