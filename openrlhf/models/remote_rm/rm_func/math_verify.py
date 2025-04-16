import json
import os
import random
import re
import numpy as np
from argparse import ArgumentParser
from multiprocessing import Process, Queue

import Levenshtein
from flask import Flask, jsonify, request
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from openrlhf.models.math_utils import is_equal


format_pattern = r"^<think>(?:(?!</think>).)*</think>\s*<answer>(?:(?!</answer>).)*</answer>\Z"
response_prefix = r"<\|im_start\|>assistant\n"

def verify_format(content):
    """
    Verify if the string meets the format requirements:
    - Must start with <think> and end with </answer>
    - Must contain exactly one pair of <think>...</think> and <answer>...</answer> tags
    - No extra characters allowed between </think> and <answer> tags
    """
    think_count = content.count("<think>")
    answer_count = content.count("<answer>")
    return bool(re.match(format_pattern, content, re.DOTALL)) and think_count == 1 and answer_count == 1

def math_parse(content):
    return parse(
            content,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
 
def verify_math(queries, prompts, labels):
    gold_parsed = labels
    if len(gold_parsed) != 0:
        # We require the answer to be provided in correct latex (no malformed operators)
        pattern = re.compile(r"(\\boxed{.*})", re.DOTALL)
        matches = re.findall(pattern, queries)
        if len(matches) > 0:
            answer_parsed = matches[-1]
        else:
            answer_parsed = ''
        try:
            math_answer = math_parse(f'${answer_parsed}$')[-1]
        except:
            math_answer = ''
        # Reward 1 if the content is the same as the ground truth, 0 otherwise
        try:
            reward = float(is_equal(math_answer, gold_parsed, 'None'))
        except Exception as e:
            print (math_answer, gold_parsed)
            reward = 0.0
            print("Failed to verify: ", e)
    else:
        # If the gold solution is not parseable, we reward 1 to skip this example
        reward = 1.0
        print("Failed to parse gold solution: ", queries)
    return reward

