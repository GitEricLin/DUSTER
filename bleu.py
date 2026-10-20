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
  # Counter that maps each n-gram to its number of occurrences
  ngram_counts = collections.Counter()
  for order in range(1, max_order + 1):
    for i in range(0, len(segment) - order + 1):
      # n-grams are stored as tuples (immutable, hence hashable)
      # extract every n-gram of length `order` (1..max_order) from the segment
      ngram = tuple(segment[i:i+order])
      # increment the count of this n-gram
      ngram_counts[ngram] += 1
  # return the n-gram counts
  return ngram_counts

# gn = _get_ngrams("These is", 2)
# print(gn)

# smooth=True adds 1 to both the matched and the possible n-gram counts of every order
# reference_corpus: ground-truth references; translation_corpus: generated texts
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
  # matched n-gram counts per order (numerator of the modified precision)
  matches_by_order = [0] * max_order
  # possible n-gram counts per order in the generated text (denominator)
  possible_matches_by_order = [0] * max_order
  reference_length = 0
  translation_length = 0
  for (references, translation) in zip(reference_corpus, translation_corpus):
    # accumulate the length of the shortest reference of each segment
    # (the brevity penalty uses the shortest reference, which is the conservative choice)
    reference_length += min(len(r) for r in references)
    # accumulate the length of the generated texts
    translation_length += len(translation)

    # n-gram counts merged over all references
    merged_ref_ngram_counts = collections.Counter()
    for reference in references:
      # |= keeps, for each n-gram, the maximum count over the references
      merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
    # n-gram counts of the generated text
    translation_ngram_counts = _get_ngrams(translation, max_order)
    # clipped overlap: each n-gram counted at most as often as in the references
    overlap = translation_ngram_counts & merged_ref_ngram_counts
    # add overlapping counts by order, e.g. a bigram goes to matches_by_order[1]
    for ngram in overlap:
      matches_by_order[len(ngram)-1] += overlap[ngram]
    
    for order in range(1, max_order+1):
      # number of n-grams of this order in the generated text, e.g. 'the movie is good' has 3 bigrams
      possible_matches = len(translation) - order + 1
      if possible_matches > 0:
        possible_matches_by_order[order-1] += possible_matches

  precisions = [0] * max_order
  for i in range(0, max_order):
    # with smoothing, add 1 to numerator and denominator
    if smooth:
      precisions[i] = ((matches_by_order[i] + 1.) / (possible_matches_by_order[i] + 1.))
    else:
      if possible_matches_by_order[i] > 0:
        precisions[i] = (float(matches_by_order[i]) / possible_matches_by_order[i])
      else:
        precisions[i] = 0.0

  # geometric mean of the precisions if all are positive, otherwise 0
  if min(precisions) > 0:
    # average of the log precisions
    p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
    # exponentiate to obtain the geometric mean
    geo_mean = math.exp(p_log_sum)
  else:
    geo_mean = 0

  # length ratio of generated text to reference, used for the brevity penalty
  ratio = float(translation_length) / reference_length

  # no penalty if the generated text is longer than the reference; otherwise exp(1 - 1/ratio)
  if ratio > 1.0:
    bp = 1.
  else:
    bp = math.exp(1 - 1. / ratio)

  bleu = geo_mean * bp

  # returns BLEU, per-order precisions, brevity penalty, length ratio, generated length, reference length
  return (bleu, precisions, bp, ratio, translation_length, reference_length)
