import asyncio
import os
import re
import sys
import textwrap
from collections import Counter
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import json

# torch import is moved to where it's needed to avoid import errors
# import torch

# Add project root to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from swift.llm import PtEngine, RequestConfig, RolloutInferRequest, Template, to_device
from swift.llm.infer.protocol import ChatCompletionResponse, ChatCompletionResponseChoice
from swift.plugin import ORM, orms, rm_plugins
# register context manager(used in gym training)
from swift.plugin.context_manager import ContextManager, context_managers
from swift.plugin.env import Env, envs
from swift.plugin.multi_turn import MultiTurnScheduler, multi_turns
from swift.plugin.rm_plugin import DefaultRMPlugin
from swift.utils import get_logger

logger = get_logger()
"""
TO CUSTOMIZE REWARD FUNCTION:
    Step 1: Define a Reward Class
        Implement your custom reward calculation logic within the __call__ method.
        The method accepts the model's output completions and dataset columns (passed as kwargs) as input parameters.

    Step 2: Add your reward function to the orms registry:
        orms['my_reward_function'] = MyRewardFunction

    Step 3: Configure the Arguments
        Run the script with:
        --external_plugins /path/to/plugin.py \
        --reward_funcs my_reward_function
"""


# For additional reward functions, refer to swift/plugin/orm.py.
class CountdownORM(ORM):

    def __call__(self, completions, target, nums, **kwargs) -> List[float]:
        """
        Evaluates completions based on Mathematical correctness of the answer

        Args:
            completions (list[str]): Generated outputs
            target (list[str]): Expected answers
            nums (list[str]): Available numbers

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        for completion, gt, numbers in zip(completions, target, nums):
            try:
                # Check if the format is correct
                match = re.search(r'<answer>(.*?)<\/answer>', completion)
                if match is None:
                    rewards.append(0.0)
                    continue
                # Extract the "answer" part from the completion
                equation = match.group(1).strip()
                if '=' in equation:
                    equation = equation.split('=')[0]
                # Extract all numbers from the equation
                used_numbers = [int(n) for n in re.findall(r'\d+', equation)]

                # Check if all numbers are used exactly once
                if sorted(used_numbers) != sorted(numbers):
                    rewards.append(0.0)
                    continue
                # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
                allowed_pattern = r'^[\d+\-*/().\s]+$'
                if not re.match(allowed_pattern, equation):
                    rewards.append(0.0)
                    continue

                # Evaluate the equation with restricted globals and locals
                result = eval(equation, {"__builti'ns__": None}, {})
                # Check if the equation is correct and matches the ground truth
                if abs(float(result) - float(gt)) < 1e-5:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            except Exception:
                # If evaluation fails, reward is 0
                rewards.append(0.0)
        return rewards


orms['external_countdown'] = CountdownORM


class MultiModalAccuracyORM(ORM):

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer format to extract option letter
        
        Args:
            answer: Original answer string
            
        Returns:
            Normalized answer (option letter A-D or processed text)
        """
        # Extract answer part from tags
        processed_answer = answer.strip()
        answer_start = processed_answer.find("<answer>")
        answer_end = processed_answer.find("</answer>")
        
        if answer_start != -1 and answer_end != -1 and answer_end > answer_start:
            processed_answer = processed_answer[answer_start+8:answer_end].strip()
        
        # Extract option letter if present
        final_answer = ""
        # Try to extract from formats like "(B) ..." or "B. ..."
        match = re.search(r'\(([A-D])\)|([A-D])\.', processed_answer)
        if match:
            final_answer = match.group(1) or match.group(2)
        else:
            # If no parenthesis format, look for option letters directly
            for char in processed_answer:
                if char in ['A', 'B', 'C', 'D']:
                    final_answer = char
                    break
        
        # If no option letter found, return the processed answer
        if not final_answer:
            final_answer = processed_answer
        
        return final_answer.strip()
    
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        from math_verify import parse, verify
        for content, sol in zip(completions, solution):
            reward = 0.0
            # Try symbolic verification first
            try:
                answer = parse(content)
                if float(verify(answer, parse(sol))) > 0:
                    reward = 1.0
            except Exception:
                pass  # Continue to next verification method if this fails

            # If symbolic verification failed, try string matching
            if reward == 0.0:
                try:
                    # Extract answer from solution
                    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                    ground_truth = self._normalize_answer(ground_truth)

                    # Extract answer from content
                    content_match = re.search(r'<answer>(.*?)</answer>', content)
                    student_answer = content_match.group(1).strip() if content_match else content.strip()
                    student_answer = self._normalize_answer(student_answer)

                    # Compare the extracted answers (case-insensitive)
                    if student_answer.upper() == ground_truth.upper():
                        reward = 1.0
                except Exception:
                    pass  # Keep reward as 0.0 if both methods fail
            rewards.append(reward)
        return rewards


orms['external_r1v_acc'] = MultiModalAccuracyORM

class ToolCallFormat(ORM):
    """
    Format reward that supports both tool calls and final answers
    
    Valid formats:
    1. <think>...</think><tool_call>...</tool_call>
    2. <think>...</think><answer>...</answer>
    """
    
    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if completion has valid format with tool calls or answers"""
        rewards = []
        
        for content in completions:
            # Pattern 1: <think> followed by <tool_call>
            tool_call_pattern = r'^<think>.*?</think>\s*<tool_call>.*?</tool_call>(?![\s\S])'
            
            # Pattern 2: <think> followed by <answer>
            answer_pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
            
            tool_call_match = re.match(tool_call_pattern, content, re.DOTALL | re.MULTILINE)
            answer_match = re.match(answer_pattern, content, re.DOTALL | re.MULTILINE)
            
            if tool_call_match or answer_match:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        
        return rewards


orms['external_tool_call_format'] = ToolCallFormat


class MultiTurnToolCallFormat(ORM):
    """
    Simplified multi-turn format reward that only checks if the completion has <answer>...</answer> format.
    
    Since completions only contain the last turn's output, we simply check if it has the correct <answer> format.
    
    Usage:
        swift rlhf --reward_funcs external_multiturn_format --max_turns 3
    """
    
    def __call__(self, completions, **kwargs) -> List[float]:
        """
        Reward function that checks if completions have <answer> format
        
        Args:
            completions: List of generated completions (only last turn's output)
            **kwargs: Additional parameters (not used in simplified version)
        
        Returns:
            List of format rewards (1.0 for correct, 0.0 for incorrect)
        """

        pattern = r"<answer>.*?</answer>"
        completion_contents = [completion for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]



orms['external_multiturn_format'] = MultiTurnToolCallFormat


class MultiTurnToolCallFormatProgressive(ORM):
    """
    Progressive multi-turn format reward with fine-grained feedback.
    
    Reward breakdown:
    - 0.0: No <think> tag
    - 0.3: Has <think>...</think>
    - 0.5: Has <think> + correct action tag (<tool_call> or <answer>)
    - 0.8: Correct structure but has wrong additional tags
    - 1.0: Perfect format for the turn
    
    Non-final turns expect: <think>...</think><tool_call>...</tool_call>
    Final turn expects: <think>...</think><answer>...</answer>
    """
    
    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function with progressive feedback"""
        rewards = []
        
        # Get trajectory information
        trajectory_ids: List[str] = kwargs.get('request_id', [])
        global_trajectories: Dict[str, List[Dict]] = kwargs.get('trajectory_inputs', {})
        max_turns = kwargs.get('max_turns', 3)
        
        # Fallback if no trajectory info
        if not trajectory_ids or not global_trajectories:
            logger.warning("No trajectory information available for progressive format check")
            return [0.0] * len(completions)
        
        for local_tra_id, completion in zip(trajectory_ids, completions):
            if local_tra_id not in global_trajectories:
                rewards.append(0.0)
                continue
            
            trajectory = global_trajectories[local_tra_id]
            current_turn = len(trajectory)
            is_final_turn = (current_turn >= max_turns)
            
            reward = self._compute_progressive_reward(completion, is_final_turn, current_turn, max_turns)
            rewards.append(reward)
        
        return rewards
    
    def _compute_progressive_reward(self, content: str, is_final_turn: bool, 
                                    current_turn: int, max_turns: int) -> float:
        """
        Compute progressive reward based on format quality
        
        Args:
            content: Model completion
            is_final_turn: Whether this is the final turn
            current_turn: Current turn number
            max_turns: Maximum turns
            
        Returns:
            Float reward between 0.0 and 1.0
        """
        content = content.strip()
        reward = 0.0
        
        # Check for <think> tag (30% of reward)
        if not content.startswith('<think>'):
            return 0.0
        
        if '</think>' not in content:
            return 0.0
        
        reward = 0.3  # Has valid <think> block
        
        # Check for required action tag
        has_tool_call = '<tool_call>' in content and '</tool_call>' in content
        has_answer = '<answer>' in content and '</answer>' in content
        
        if is_final_turn:
            # Final turn should have <answer>
            if not has_answer:
                return reward  # Only 0.3 for missing answer
            
            reward = 0.5  # Has <think> + <answer>
            
            # Check if incorrectly has <tool_call>
            if has_tool_call:
                return 0.5  # Penalize having tool_call in final turn
            
            reward = 0.8  # Has correct tags, no wrong tags
            
            # Check proper structure (nothing after </answer>)
            pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>\s*$'
            if re.match(pattern, content, re.DOTALL):
                reward = 1.0  # Perfect format
            
        else:
            # Non-final turn should have <tool_call>
            if not has_tool_call:
                return reward  # Only 0.3 for missing tool_call
            
            reward = 0.5  # Has <think> + <tool_call>
            
            # Check if incorrectly has <answer>
            if has_answer:
                return 0.5  # Penalize having answer in non-final turn
            
            reward = 0.8  # Has correct tags, no wrong tags
            
            # Check proper structure (nothing after </tool_call>)
            pattern = r'^<think>.*?</think>\s*<tool_call>.*?</tool_call>\s*$'
            if re.match(pattern, content, re.DOTALL):
                # Additionally validate JSON structure in tool_call
                try:
                    tool_call_match = re.search(r'<tool_call>\s*({.*?})\s*</tool_call>', 
                                               content, re.DOTALL)
                    if tool_call_match:
                        json_str = tool_call_match.group(1)
                        tool_data = json.loads(json_str)
                        
                        # Check required fields
                        if 'name' in tool_data and 'arguments' in tool_data:
                            reward = 1.0  # Perfect format with valid JSON
                        else:
                            reward = 0.9  # Valid structure but missing fields
                except (json.JSONDecodeError, AttributeError):
                    reward = 0.8  # Valid structure but invalid JSON
        
        return reward


orms['external_multiturn_format_progressive'] = MultiTurnToolCallFormatProgressive


class MultiTurnThinkingTips(ORM):
    """
    A reward function example designed for use with the `ThinkingTipsScheduler`.

    This class demonstrates how to handle reward computation when a single
    training sample (or request) is split into multiple "turns" or steps.
    Specifically, it computes the reward based on the **last turn** of each
    multi-turn trajectory using a math accuracy function.

    NOTE
    ----
    If you feed fragments of the *same* trajectory as independent samples, this
    function **must return an identical reward for every fragment**
    """

    def __init__(self):
        from swift.plugin.orm import MathAccuracy
        self.acc_func = MathAccuracy()

    def __call__(self, completions, **kwargs) -> List[float]:
        trajectory_ids: List[str] = kwargs.get('request_id')

        global_trajectorys: Dict[str, List[Dict]] = kwargs.get('trajectory_inputs')

        rewards = []
        for local_tra_id in trajectory_ids:
            total_trajectory_inputs = global_trajectorys[local_tra_id]
            # For reward calculation, we use the entire trajectory of this sample.
            # Here, we specifically evaluate only the last turn.
            last_turn_messages = total_trajectory_inputs[-1]['messages']
            last_turn_completion = last_turn_messages[-1]['content']
            last_turn_solution = total_trajectory_inputs[-1]['solution']
            # Compute reward based on math accuracy for the final completion.
            reward = self.acc_func([last_turn_completion], [last_turn_solution])[0]
            rewards.append(reward)
        return rewards


orms['thinking_tips'] = MultiTurnThinkingTips


# ref implementation: https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py
class CodeReward(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('e2b') is not None, (
            "The e2b package is required but not installed. Please install it using 'pip install e2b-code-interpreter'."
        )
        from dotenv import load_dotenv
        load_dotenv()

    @staticmethod
    def extract_code(completion: str, language: str) -> str:
        pattern = re.compile(rf'```{language}\n(.*?)```', re.DOTALL)
        matches = pattern.findall(completion)
        extracted_answer = matches[-1] if len(matches) >= 1 else ''
        return extracted_answer

    def run_async_from_sync(self, scripts: List[str], languages: List[str]) -> List[float]:
        """Function wrapping the `run_async` function."""
        # Create a new event loop and set it
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Run the async function and get the result
            rewards = loop.run_until_complete(self.run_async(scripts, languages))
        finally:
            loop.close()

        return rewards

    async def run_async(self, scripts: List[str], languages: List[str]) -> List[float]:
        from e2b_code_interpreter import AsyncSandbox

        # Create the sandbox by hand, currently there's no context manager for this version
        try:
            sbx = await AsyncSandbox.create(timeout=30, request_timeout=3)
        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            return [0.0] * len(scripts)
        # Create a list of tasks for running scripts concurrently
        tasks = [self.run_script(sbx, script, language) for script, language in zip(scripts, languages)]

        # Wait for all tasks to complete and gather their results as they finish
        results = await asyncio.gather(*tasks)
        rewards = list(results)  # collect results

        # Kill the sandbox after all the tasks are complete
        await sbx.kill()

        return rewards

    async def run_script(self, sbx, script: str, language: str) -> float:
        try:
            execution = await sbx.run_code(script, language=language, timeout=30)
        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            return 0.0
        try:
            return float(execution.text)
        except (TypeError, ValueError):
            return 0.0

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that evaluates code snippets using the E2B code interpreter.

        Assumes the dataset contains a `verification_info` column with test cases.
        """
        evaluation_script_template = """
        import subprocess
        import json

        def evaluate_code(code, test_cases):
            passed = 0
            total = len(test_cases)
            exec_timeout = 5

            for case in test_cases:
                process = subprocess.run(
                    ["python3", "-c", code],
                    input=case["input"],
                    text=True,
                    capture_output=True,
                    timeout=exec_timeout
                )

                if process.returncode != 0:  # Error in execution
                    continue

                output = process.stdout.strip()
                if output.strip() == case["output"].strip():
                    passed += 1

            success_rate = (passed / total)
            return success_rate

        code_snippet = {code}
        test_cases = json.loads({test_cases})

        evaluate_code(code_snippet, test_cases)
        """
        verification_info = kwargs['verification_info']
        languages = [info['language'] for info in verification_info]
        code_snippets = [
            self.extract_code(completion, language) for completion, language in zip(completions, languages)
        ]
        scripts = [
            evaluation_script_template.format(
                code=json.dumps(code), test_cases=json.dumps(json.dumps(info['test_cases'])))
            for code, info in zip(code_snippets, verification_info)
        ]
        try:
            rewards = self.run_async_from_sync(scripts, languages)

        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            rewards = [0.0] * len(completions)

        return rewards


orms['external_code_reward'] = CodeReward


class CodeFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        verification_info = kwargs['verification_info']
        rewards = []
        for content, info in zip(completions, verification_info):
            pattern = r'^<think>.*?</think>\s*<answer>.*?```{}.*?```.*?</answer>(?![\s\S])'.format(info['language'])
            match = re.match(pattern, content, re.DOTALL | re.MULTILINE)
            reward = 1.0 if match else 0.0
            rewards.append(reward)
        return rewards


orms['external_code_format'] = CodeFormat


class CodeRewardByJudge0(ORM):
    LANGUAGE_ID_MAP = {
        'assembly': 45,
        'bash': 46,
        'basic': 47,
        'c': 50,
        'c++': 54,
        'clojure': 86,
        'c#': 51,
        'cobol': 77,
        'common lisp': 55,
        'd': 56,
        'elixir': 57,
        'erlang': 58,
        'executable': 44,
        'f#': 87,
        'fortran': 59,
        'go': 60,
        'groovy': 88,
        'haskell': 61,
        'java': 62,
        'javascript': 63,
        'kotlin': 78,
        'lua': 64,
        'multi-file program': 89,
        'objective-c': 79,
        'ocaml': 65,
        'octave': 66,
        'pascal': 67,
        'perl': 85,
        'php': 68,
        'plain text': 43,
        'prolog': 69,
        'python': 71,
        'python2': 70,
        'python3': 71,
        'r': 80,
        'ruby': 72,
        'rust': 73,
        'scala': 81,
        'sql': 82,
        'swift': 83,
        'typescript': 74,
        'visual basic.net': 84
    }
    PYTHON_ID = 71

    def __init__(self):
        self.endpoint = os.getenv('JUDGE0_ENDPOINT')
        assert self.endpoint is not None, (
            'Judge0 endpoint is not set. Please set the JUDGE0_ENDPOINT environment variable.')
        x_auth_token = os.getenv('JUDGE0_X_AUTH_TOKEN')
        self.headers = {'Content-Type': 'application/json'}
        if x_auth_token is not None:
            self.headers['X-Auth-Token'] = x_auth_token

    @staticmethod
    def extract_code(completion: str, language: str) -> str:
        pattern = re.compile(rf'```{language}\n(.*?)```', re.DOTALL)
        matches = pattern.findall(completion)
        extracted_answer = matches[-1] if len(matches) >= 1 else ''
        return extracted_answer

    @classmethod
    def get_language_id(cls, language):
        if language is None:
            return cls.PYTHON_ID
        return cls.LANGUAGE_ID_MAP.get(language.lower().strip(), cls.PYTHON_ID)

    async def _evaluate_code(self, code, test_cases, language_id):
        import aiohttp
        try:
            passed = 0
            total = len(test_cases)

            for case in test_cases:
                if code is not None and code != '':
                    async with aiohttp.ClientSession() as session:
                        payload = {
                            'source_code': code,
                            'language_id': language_id,
                            'stdin': case['input'],
                            'expected_output': case['output']
                        }
                        logger.debug(f'Payload: {payload}')
                        async with session.post(
                                self.endpoint + '/submissions/?wait=true', json=payload,
                                headers=self.headers) as response:
                            response_json = await response.json()
                            logger.debug(f'Response: {response_json}')
                            if response_json['status']['description'] == 'Accepted':
                                passed += 1

            success_rate = (passed / total)
            return success_rate
        except Exception as e:
            logger.warning(f'Error from Judge0 executor: {e}')
            return 0.0

    def run_async_from_sync(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            rewards = loop.run_until_complete(self.run_async())
        finally:
            loop.close()
        return rewards

    async def run_async(self):
        tasks = [
            self._evaluate_code(code, info['test_cases'], CodeRewardByJudge0.get_language_id(info['language']))
            for code, info in zip(self.code_snippets, self.verification_info)
        ]
        results = await asyncio.gather(*tasks)
        rewards = list(results)
        return rewards

    def __call__(self, completions, **kwargs) -> List[float]:
        self.verification_info = kwargs['verification_info']

        languages = [info['language'] for info in self.verification_info]
        self.code_snippets = [
            self.extract_code(completion, language) for completion, language in zip(completions, languages)
        ]

        try:
            rewards = self.run_async_from_sync()
        except Exception as e:
            logger.warning(f'Error from Judge0 executor: {e}')
            rewards = [0.0] * len(completions)
        return rewards


orms['external_code_reward_by_judge0'] = CodeRewardByJudge0


# ref implementation: https://github.com/qiancheng0/ToolRL/blob/main/verl/utils/reward_score/rlla.py
# arxiv paper: https://arxiv.org/abs/2504.13958
# MAX1STEP30MAX3: enable Two stage reward Setting include Format and Correctness
# SCHEDULEREWARD: enable Dynamic (Finegrained) reward Setting include Format and Correctness
# Correctness Reward Granularity:
# COARSEREWARD -> Coarse, INTERMEDIATEREWARD -> Intermediate, REFINEDREWARD -> Finegrained
class ToolUseFormatReward(ORM):

    def __init__(self):
        self.format_max_possible = 1.0
        self.format_min_possible = 0.0

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        trainer_state = kwargs.get('trainer_state')
        global_step = trainer_state.global_step
        max_possible_reward = self.format_max_possible
        min_possible_reward = self.format_min_possible
        # Two stage (Coarse) Setting, divide training into two phases. Format Reward in [0,0.5] if step < 30 else [0,1]
        if str(os.getenv('MAX1STEP30MAX3', 0)) == '1':
            if global_step >= 30:
                max_possible_reward = self.format_max_possible / 2
                min_possible_reward = self.format_min_possible / 2
            else:
                max_possible_reward = self.format_max_possible
                min_possible_reward = self.format_min_possible

        # apply continuous interpolation between the two reward scales throughout training.
        if str(os.getenv('SCHEDULEREWARD', 0)) == '1':
            max_possible_reward = 2 - (2 - max_possible_reward) * global_step / 150
            min_possible_reward = -2 + (2 + min_possible_reward) * global_step / 150
            if max_possible_reward < 1.0:
                max_possible_reward = 1.0
            if min_possible_reward > -1.0:
                min_possible_reward = -1.0

        rewards = []
        responses = completions

        for response, ans in zip(responses, solution):
            reward = min_possible_reward
            if '<response>' in ans and '<tool_call>' not in ans:
                pattern = r'^<think>.*?</think>\s*<response>.*?</response>$'
                if re.search(pattern, response,
                             re.DOTALL) and response.count('<response>') == 1 and response.count('</response>') == 1:
                    reward = max_possible_reward
            elif '<response>' not in ans and '<tool_call>' in ans:
                pattern = r'^<think>.*?</think>\s*<tool_call>.*?</tool_call>$'
                if re.search(pattern, response,
                             re.DOTALL) and response.count('<tool_call>') == 1 and response.count('</tool_call>') == 1:
                    reward = max_possible_reward
            elif '<response>' in ans and '<tool_call>' in ans:
                pattern = r'^<think>.*?</think>\s*<tool_call>.*?</tool_call>\s*<response>.*?</response>$'
                if (re.search(pattern, response, re.DOTALL) and response.count('<tool_call>') == 1
                        and response.count('</tool_call>') == 1 and response.count('<response>') == 1
                        and response.count('</response>') == 1):
                    reward = max_possible_reward
            else:
                pattern = r'^<think>.*?</think>$'
                if re.search(pattern, response, re.DOTALL):
                    reward = max_possible_reward

            rewards.append(reward)

        return rewards


orms['external_tooluse_format_reward'] = ToolUseFormatReward


class ToolUseLengthReward(ORM):

    def __init__(self):
        self.length_max_possible = 1.0
        self.length_min_possible = 0.0

    # customized reward functions: length
    def __call__(self, completions, solution, **kwargs):
        max_possible_reward = self.length_max_possible
        min_possible_reward = self.length_min_possible
        trainer_state = kwargs.get('trainer_state')
        global_step = trainer_state.global_step
        # SCHEDULELENGTH: enable Dynamic Length Reward
        if os.getenv('SCHEDULELENGTH', 0) == '1':
            max_reward_len = (640 - 384) * global_step / 105 + 384
        else:
            max_reward_len = 512
        """Reward function that gives higher scores to longer completions."""
        responses = completions
        rewards = []

        for response, ans in zip(responses, solution):
            if '<think>' not in response or '</think>' not in response:
                rewards.append(min_possible_reward)
                continue
            think_responses = response.split('<think>')[-1].split('</think>')[0].strip()
            reward = round(len(think_responses.split()) / max_reward_len, 2)
            if reward > 1.0:
                reward = 1.0

            final_reward = reward * (max_possible_reward - min_possible_reward) + min_possible_reward
            rewards.append(final_reward)

        return rewards


orms['external_tooluse_length_reward'] = ToolUseLengthReward


class ToolUseCorrectnessReward(ORM):

    def __init__(self):
        if str(os.getenv('CORRECTMAX1', 0)) == '1':
            self.tool_max_possible = 1.0
            self.tool_min_possible = -1.0
        else:
            self.tool_max_possible = 3.0
            self.tool_min_possible = -3.0

    def match_score(self, list1, list2):
        if list1 == list2:
            return 1.0

        if os.getenv('REFINEDREWARD', 0) == '1':
            if list1 != list2:
                return 0.0

        if not list1 or not list2:
            return 0.0

        count1 = Counter(list1)  # Frequency count for list1
        count2 = Counter(list2)  # Frequency count for list2

        intersection = sum(min(count1[k], count2[k]) for k in count1.keys() & count2.keys())
        max_possible = len(list1) + len(list2) - intersection

        return intersection / max_possible if max_possible > 0 else 0.0

    def compute_tool_call_reward(self, gt_tools, pd_tools, max_possible_reward, min_possible_reward):
        if gt_tools == pd_tools:
            return max_possible_reward

        if os.getenv('COARSEREWARD', 0) == '1':
            if gt_tools != pd_tools:
                return min_possible_reward

        gt_names = [tool['name'] for tool in gt_tools]
        pd_names = [tool['name'] for tool in pd_tools]
        score = self.match_score(list(gt_names), list(pd_names))

        local_max_possible = 1.0
        used_pd_indices = set()  # Keep track of matched pd_tools

        for gt_tool in gt_tools:
            gt_name = gt_tool['name']
            gt_params = gt_tool['parameters']

            if str(os.getenv('INTERMEDIATEREWARD', 0)) == '1':
                local_max_possible += 1.0
            else:
                local_max_possible += 1.0 + len(gt_params)

            best_match = None
            best_match_score = 0.0
            best_match_index = -1

            # Find the best matching unused pd_tool
            for i, pd_tool in enumerate(pd_tools):
                if i in used_pd_indices or pd_tool['name'] != gt_name:
                    continue

                if str(os.getenv('INTERMEDIATEREWARD', 0)) == '1':
                    if gt_tool == pd_tool:
                        best_match = pd_tool
                        best_match_index = i
                        best_match_score = 1.0
                        break
                    else:
                        continue

                pd_params = pd_tool['parameters']
                param_score = self.match_score(list(gt_params.keys()), list(pd_params.keys()))

                # Calculate correctness score for parameter values
                correctness_score = sum(1.0 for k, v in gt_params.items() if k in pd_params and pd_params[k] == v)

                total_score = param_score + correctness_score

                if total_score > best_match_score:
                    best_match_score = total_score
                    best_match = pd_tool
                    best_match_index = i

            if best_match:
                used_pd_indices.add(best_match_index)
                score += best_match_score

        return (max_possible_reward - min_possible_reward) * score / local_max_possible + min_possible_reward

    # custoimzed reward functions: tool call correctness
    def __call__(self, completions, solution, **kwargs):
        trainer_state = kwargs.get('trainer_state')
        global_step = trainer_state.global_step
        max_possible_reward = self.tool_max_possible
        min_possible_reward = self.tool_min_possible
        # two stage (Coarse) Setting, divide training into two phases.
        if str(os.getenv('MAX1STEP30MAX3', 0)) == '1':
            if global_step < 30:
                max_possible_reward = max_possible_reward / 3
                min_possible_reward = min_possible_reward / 3
            else:
                max_possible_reward = max_possible_reward
                min_possible_reward = min_possible_reward
        # apply continuous interpolation between the two reward scales throughout training.
        if str(os.getenv('SCHEDULEREWARD', 0)) == '1':
            max_possible_reward = (max_possible_reward - 2) * global_step / 150 + 2
            min_possible_reward = (min_possible_reward + 2) * global_step / 150 - 2
            if max_possible_reward > 3.0:
                max_possible_reward = 3.0
            if min_possible_reward < -3.0:
                min_possible_reward = -3.0

        responses = completions
        rewards = []

        for response, ans in zip(responses, solution):
            reward = 0.0

            if '<tool_call>' not in ans:
                # if "<tool_call>" not in response and "</tool_call>" not in response:
                #     reward = max_possible_reward
                # else:
                #     reward = min_possible_reward
                rewards.append(reward)
                continue

            gt_tool_call = ans.split('<tool_call>')[1].split('</tool_call>')[0].strip()
            gt_tools = gt_tool_call.split('\n')
            gt_tools = [json.loads(tool) for tool in gt_tools]  # each diction contains "name" and "parameter"

            try:
                # if the format is not correct, directly give the lowest possible score
                assert '<tool_call>' in response
                assert '</tool_call>' in response
                pd_tools = response.split('<tool_call>')[1].split('</tool_call>')[0].strip().split('\n')
                pd_tools = [json.loads(tool) for tool in pd_tools]
                reward = self.compute_tool_call_reward(gt_tools, pd_tools, max_possible_reward,
                                                       min_possible_reward)  # top reward is 2
            except (ValueError, IndexError, AssertionError):
                reward = min_possible_reward

            rewards.append(reward)

        return rewards


orms['external_tooluse_correct_reward'] = ToolUseCorrectnessReward
"""
TO CUSTOMIZE REWARD MODEL:
    Step 1: Define a Reward Class
        Implement your custom reward calculation logic within the __call__ method.
        The method accepts the messages generated by the model during interactions
        and dataset columns as inputs parameters.

    Step 2: Add your reward model plugin to the rm_plugins registry:
        rm_plugins['my_rm_plugin'] = MyRMPlugin

    Step 3: Configure the Arguments
        Run the script with:
        --external_plugins /path/to/plugin.py \
        --reward_model_plugin my_rm_plugin

For GenRM you can refer to swift/llm/plugin/rm_plugin/GenRMPlugin
"""


class CustomizedRMPlugin:
    """
    Customized Reward Model Plugin, same to DefaultRMPlugin

    It assumes that `self.model` is a classification model with a value head(output dimmension 1).
    The first logits value from the model's output is used as the reward score.
    """

    def __init__(self, model, template):
        self.model = model
        self.template: Template = template

    def __call__(self, inputs, **kwargs):
        import torch
        batched_inputs = [self.template.encode(deepcopy(infer_request)) for infer_request in inputs]
        reward_inputs = to_device(self.template.data_collator(batched_inputs), self.model.device)

        with torch.inference_mode():
            return self.model(**reward_inputs).logits[:, 0]


class QwenLongPlugin(DefaultRMPlugin):
    # https://arxiv.org/abs/2505.17667
    # NOTE: you should customize the verified reward function, you can refer to
    # https://github.com/Tongyi-Zhiwen/QwenLong-L1/tree/main/verl/verl/utils/reward_score
    # hf_dataset: https://huggingface.co/datasets/Tongyi-Zhiwen/DocQA-RL-1.6K/viewer/default/train
    # ms_dataset: https://modelscope.cn/datasets/iic/DocQA-RL-1.6K
    def __init__(self, model, template, accuracy_orm=None):
        super().__init__(model, template)
        # initilize PTEngine to infer
        self.engine = PtEngine.from_model_template(self.model, self.template, max_batch_size=0)  # 0: no limit
        self.request_config = RequestConfig(temperature=0)  # customise your request config here
        self.system = textwrap.dedent("""
            You are an expert in verifying if two answers are the same.

            Your input consists of a problem and two answers: Answer 1 and Answer 2.
            You need to check if they are equivalent.

            Your task is to determine if the two answers are equivalent, without attempting to solve the original problem.
            Compare the answers to verify they represent identical values or meanings,
            even when expressed in different forms or notations.

            Your output must follow this format:
            1) Provide an explanation for why the answers are equivalent or not.
            2) Then provide your final answer in the form of: [[YES]] or [[NO]]

            Problem: {problem_placeholder}
            Answer 1: {answer1_placeholder}
            Answer 2: {answer2_placeholder}
        """)  # noqa
        self.accuracy_orm = accuracy_orm

    def __call__(self, inputs, **kwargs):
        completions = [example['messages'][-1]['content'] for example in inputs]
        ground_truths = [example['reward_model']['ground_truth'] for example in inputs]
        rm_inputs = self.prepare_rm_inputs(inputs, completions, ground_truths)

        results = self.engine.infer(rm_inputs, self.request_config, use_tqdm=False)
        llm_rewards = self.compute_rewards(results)

        if self.accuracy_orm:
            verified_rewards = self.accuracy_orm(completions, ground_truths)
        else:
            verified_rewards = [0.0] * len(llm_rewards)

        rewards = [max(r1, r2) for r1, r2 in zip(llm_rewards, verified_rewards)]
        import torch
        return torch.tensor(rewards, dtype=torch.float32)

    def prepare_rm_inputs(self, inputs: List[Dict], completions, ground_truths) -> List[Dict]:
        rm_inputs = []
        for infer_request, completion, ground_truth in zip(inputs, completions, ground_truths):
            # Deep copy to prevent modification of original input
            rm_infer_request = deepcopy(infer_request)
            problem = infer_request['messages'][0]['content']
            start_index = problem.index('</text>')
            end_index = problem.index('Format your response as follows:')
            question = problem[start_index:end_index].replace('</text>', '').strip()
            prompt = self.system.format(
                problem_placeholder=question, answer1_placeholder=completion, answer2_placeholder=ground_truth)

            # Construct new messages tailored for the reward model
            rm_messages = [{'role': 'user', 'content': prompt}]

            # Update the messages in the reward infer request
            rm_infer_request['messages'] = rm_messages
            rm_inputs.append(rm_infer_request)
        return rm_inputs

    @staticmethod
    def extract_reward(model_output: str) -> float:
        match = re.search(r'\[([A-Z]+)\]', model_output)
        if match:
            answer = match.group(1)
            if answer == 'YES':
                return 1.0
            elif answer == 'NO':
                return 0.0
            else:
                logger.warning("Unexpected answer, expected 'YES' or 'NO'.")
                return 0.0
        else:
            logger.warning("Unable to extract reward score from the model's output, setting reward to 0")
            return 0.0  # Or raise ValueError("Format incorrect")

    def compute_rewards(self, results: List[ChatCompletionResponse]) -> List[float]:
        """
        Compute average reward scores from the reward model's outputs.

        Args:
            results (List[ChatCompletionResponse]): A list of results from the reward model.

        Returns:
            List[float]: A list of average reward scores.
        """
        rewards = []
        for idx, output in enumerate(results):
            try:
                cur_rewards = []
                for choice in output.choices:
                    response = choice.message.content
                    reward = self.extract_reward(response)
                    cur_rewards.append(reward)
                cur_rewards = [r for r in cur_rewards if r is not None]
                if cur_rewards:
                    average_reward = sum(cur_rewards) / len(cur_rewards)
                else:
                    average_reward = 0.0
                    logger.warning('No valid rewards extracted. Assigning reward score of 0.0.')

                rewards.append(average_reward)
            except Exception as e:
                logger.error(f'Error computing reward: {e}')
                rewards.append(0.0)  # Assign default reward score on failure
        return rewards


rm_plugins['my_rmplugin'] = CustomizedRMPlugin
rm_plugins['qwenlong'] = QwenLongPlugin
"""
TO CUSTOMIZE MULTITURN SCHEDULER:
    Step 1: Define a Scheduler Class
        Implement your custom scheduler with the following methods:
            - step (Required): Constructs the next round of the infer request.
            - check_finished (Optional): Determines whether the current round has finished,
                which defaults to ending when the inference result is truncated (over length) or
                when the maximum number of rounds is reached.
            or override run method in MultiTurnScheduler class.

        Both methods accept:
            - the last turn's InferRequest/response_choice
            - the current turn count

    Step 2: Add your scheduler to the multi_turns registry:
        multi_turns['my_scheduler'] = MyScheduler

    Step 3: Configure the Arguments
        Run the script with:
        swift rollout \
            --external_plugins /path/to/plugin.py \
            --multi_turn_scheduler my_scheduler
"""


class ToolCallScheduler(MultiTurnScheduler):
    # A simple scheduler that supports tool calls by overriding the `step` method
    # Tool parsing uses the ReAct format
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # A simple tool registry. Extend or replace with your own tools as needed.
        self.tools = {
            'calculator': self._calculator_tool,
        }

    def _calculator_tool(self, expression: str) -> str:
        # A very small sandboxed calculator
        # The calculator tool implemented here can perform only basic arithmetic operations and
        # may not be able to solve all math problems in the dataset.
        import ast
        import operator

        def _evaluate_ast_node(node) -> Union[int, float]:
            operators = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.USub: operator.neg,
                ast.UAdd: operator.pos,
            }

            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    return node.value
                else:
                    raise TypeError(f'Unsupported constant type: {type(node.value)}')

            elif isinstance(node, ast.Num):
                return node.n

            elif isinstance(node, ast.BinOp):
                left = _evaluate_ast_node(node.left)
                right = _evaluate_ast_node(node.right)
                op = operators.get(type(node.op))

                if op is None:
                    raise TypeError(f'Unsupported operation: {type(node.op).__name__}')

                if isinstance(node.op, ast.Div) and right == 0:
                    raise ZeroDivisionError('Division by zero')

                return op(left, right)

            elif isinstance(node, ast.UnaryOp):
                operand = _evaluate_ast_node(node.operand)
                op = operators.get(type(node.op))

                if op is None:
                    raise TypeError(f'Unsupported unary operation: {type(node.op).__name__}')

                return op(operand)

            else:
                raise TypeError(f'Unsupported AST node type: {type(node).__name__}')

        try:
            expression = expression.strip().replace(' ', '')

            if not re.match(r'^[0-9+\-*/().\s]+$', expression):
                return 'Error: expression contains disallowed characters.'

            if expression.count('(') != expression.count(')'):
                return 'Error: unmatched parentheses.'

            try:
                result = ast.literal_eval(expression)
                return f'Result: {result}'
            except (ValueError, SyntaxError):
                node = ast.parse(expression, mode='eval')
                result = _evaluate_ast_node(node.body)
                return f'Result: {result}'

        except Exception as e:
            return f'Calculation error: {e}'

    def _extract_tool_calls(self, text: str):
        """
        Parse tool-call patterns using ReAct format from model output.
        Format: Action: tool_name\nAction Input: parameters
        """
        import re

        pattern = r'Action:\s*(.*?)\s*\nAction Input:\s*(.*?)(?:\n|$)'
        matches = re.findall(pattern, text, re.DOTALL)
        if not matches:
            return None
        return [{'tool': name.strip(), 'params': params.strip()} for name, params in matches]

    def _execute_tools(self, tool_calls):
        """Run each requested tool and collect its observation string."""
        results = []
        for call in tool_calls:
            name, params = call['tool'], call['params']
            if name in self.tools:
                try:
                    result = self.tools[name](params)
                    results.append(result)
                except Exception as e:
                    results.append(f'tool error {e}')
            else:
                results.append(f'unknown tool {name}')
        return results

    def check_finished(self, infer_request: 'RolloutInferRequest', response_choice: 'ChatCompletionResponseChoice',
                       current_turn: int) -> bool:
        completion = response_choice.message.content
        tool_calls = self._extract_tool_calls(completion)
        if tool_calls is None:
            return True

        return super().check_finished(infer_request, response_choice, current_turn)

    def step(self, infer_request: 'RolloutInferRequest', response_choice: 'ChatCompletionResponseChoice',
             current_turn: int) -> Dict:
        completion = response_choice.message.content
        token_ids = response_choice.token_ids
        loss_mask = [1] * len(token_ids)
        tool_calls = self._extract_tool_calls(completion)
        # assert len(tool_calls) == 1, 'this scheduler is designed for one tool call per turn'
        tool_results = self._execute_tools(tool_calls)
        # append tool result to the completion
        infer_request.messages[-1]['content'] += (tool_results[0])

        tokenizer = self.infer_engine.default_template.tokenizer
        result_tokens = tokenizer.encode(tool_results[0], add_special_tokens=False)
        token_ids.extend(result_tokens)
        loss_mask.extend([0] * len(result_tokens))

        return {
            'infer_request': infer_request,
            'response_token_ids': token_ids,
            'response_loss_mask': loss_mask,
            'rollout_infos': {
                'tool_results': tool_results[0],
                'num_turns': current_turn,
            }
        }


multi_turns['tool_call_scheduler'] = ToolCallScheduler


class SPAgentToolCallingScheduler(MultiTurnScheduler):
    """
    Scheduler for agent that emits <tool_call>…</tool_call> to call tools,
    and terminates when it emits <answer>…</answer>.
    
    Example usage in training:
        # Method 1: Use string name (auto-registers mock tools)
        args = RLHFArguments(
            multi_turn_scheduler="spagent_tool_call_scheduler",
            max_turns=3
        )
        
        # Method 2: Create instance with custom tools (pass as object)
        scheduler = SPAgentToolCallingScheduler(max_turns=3)
        scheduler.register_tool(MyCustomTool())
        args = RLHFArguments(
            multi_turn_scheduler=scheduler  # Pass instance directly
        )
    
    Note:
        - When instantiated by string name, only max_turns is passed by the framework
        - max_workers defaults to 4 (can't be changed via string instantiation)
        - Tools are auto-registered with mock=True if dependencies are available
    """
    def __init__(self, max_turns: int = 5, *args, **kwargs):
        super().__init__(max_turns=max_turns, *args, **kwargs)
        self.logger = logger
        # max_workers is fixed at 4 since we can't pass it via framework
        self.max_workers = 4
        
        # Initialize tool registry for tool execution
        from spagent.core.tool import ToolRegistry
        self.tool_registry = ToolRegistry()
        
        # Auto-register common tools if available (with robust error handling)
        self._auto_register_tools()
    
    def _auto_register_tools(self):
        """
        Automatically register common tools if they are available.
        Uses robust error handling to prevent training failures due to missing dependencies.
        """
        registered_count = 0
        failed_tools = []
        
        # Try to register each tool individually with error handling
        tool_classes = [
            ('DepthEstimationTool', 'depth_estimation_tool'),
            ('SegmentationTool', 'segmentation_tool'),
            ('ObjectDetectionTool', 'object_detection_tool'),
            ('Pi3Tool', 'pi3_tool'),
        ]
        
        for tool_class_name, tool_name in tool_classes:
            try:
                # Dynamically import each tool
                from spagent import tools as tools_module
                if hasattr(tools_module, tool_class_name):
                    ToolClass = getattr(tools_module, tool_class_name)
                    # Use real tools instead of mock for training
                    if tool_class_name == 'Pi3Tool':
                        # Configure Pi3Tool with real server
                        tool = ToolClass(use_mock=False, server_url="http://127.0.0.1:20030")
                    else:
                        # Other tools still use mock for now
                        tool = ToolClass(use_mock=True)
                    self.tool_registry.register(tool)
                    registered_count += 1
                    self.logger.debug(f"✓ Registered {tool_name} (real: {not tool.use_mock})")
                else:
                    failed_tools.append(tool_name)
                    self.logger.debug(f"✗ Tool class {tool_class_name} not found")
            except Exception as e:
                failed_tools.append(tool_name)
                self.logger.debug(f"✗ Failed to register {tool_name}: {e}")
        
        if registered_count > 0:
            self.logger.info(f"Auto-registered {registered_count} tools for SPAgent scheduler")
        else:
            self.logger.warning(
                f"No tools auto-registered. Tools can be registered manually via register_tool(). "
                f"Failed: {failed_tools}"
            )
    
    def register_tools(self, tools):
        """
        Register multiple tools for execution during training
        
        Args:
            tools: List of Tool instances or ToolRegistry
        """
        if hasattr(tools, 'get'):  # It's a ToolRegistry
            self.tool_registry = tools
        else:  # It's a list of tools
            from spagent.core.tool import Tool
            for tool in tools:
                if isinstance(tool, Tool):
                    self.tool_registry.register(tool)
                    self.logger.info(f"Registered tool: {tool.name}")
    
    def register_tool(self, tool):
        """
        Register a single tool for execution during training
        
        Args:
            tool: Tool instance to register
        """
        from spagent.core.tool import Tool
        if isinstance(tool, Tool):
            self.tool_registry.register(tool)
            self.logger.info(f"Registered tool: {tool.name}")
        else:
            self.logger.error(f"Invalid tool type: {type(tool)}")
    
    def list_registered_tools(self):
        """
        List all registered tools
        
        Returns:
            List of tool names
        """
        return self.tool_registry.list_tools()
    
    def get_tool_info(self):
        """
        Get information about all registered tools
        
        Returns:
            Dictionary with tool names and their schemas
        """
        tools_info = {}
        for tool_name in self.tool_registry.list_tools():
            tool = self.tool_registry.get(tool_name)
            if tool:
                tools_info[tool_name] = {
                    'name': tool.name,
                    'description': tool.description,
                    'parameters': tool.parameters
                }
        return tools_info

    def _parse_tool_calls(self, content: str):
        """Parse all tool-call JSON blobs inside <tool_call>…</tool_call>."""
        tool_calls = []
        # 匹配 <tool_call> { … } </tool_call>
        pattern = r'<tool_call>\s*({.*?})\s*</tool_call>'
        matches = re.findall(pattern, content, re.DOTALL)
        for m in matches:
            try:
                call = json.loads(m)
                if 'name' in call and 'arguments' in call:
                    tool_calls.append(call)
                else:
                    self.logger.warning(f"Invalid tool_call format: {m}")
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse tool_call JSON: {m} error: {e}")
        return tool_calls

    def _validate_and_fix_tool_calls(self, tool_calls: List[Dict[str, Any]], infer_request: RolloutInferRequest) -> List[Dict[str, Any]]:
        """
        Validate tool calls and fix missing required parameters
        
        Args:
            tool_calls: List of parsed tool calls
            infer_request: Current inference request containing images
            
        Returns:
            List of fixed tool calls
        """
        fixed_calls = []
        
        for call in tool_calls:
            tool_name = call['name']
            arguments = call['arguments'].copy()
            
            # Get tool from registry to check required parameters
            tool = self.tool_registry.get(tool_name)
            if tool is None:
                self.logger.warning(f"Tool {tool_name} not found in registry, skipping validation")
                fixed_calls.append(call)
                continue
            
            # Validate and fix image_path for pi3_tool (fallback if model provides wrong paths)
            if tool_name == 'pi3_tool':
                # Extract real image paths from infer_request
                real_image_paths = []

                # 首先尝试从 data_dict 获取原始图片
                if hasattr(infer_request, 'data_dict') and infer_request.data_dict:
                    if 'original_images' in infer_request.data_dict:
                        real_image_paths = infer_request.data_dict['original_images']
                        self.logger.info(f"Using original images from data_dict: {real_image_paths}")

                # 如果 data_dict 中没有，再尝试其他方式
                if not real_image_paths:
                    if hasattr(infer_request, 'images') and infer_request.images:
                        real_image_paths = infer_request.images
                    elif hasattr(infer_request, 'data_dict') and infer_request.data_dict:
                        if 'images' in infer_request.data_dict:
                            real_image_paths = infer_request.data_dict['images']
                        elif 'image' in infer_request.data_dict:
                            real_image_paths = [infer_request.data_dict['image']]
                
                # Check if model provided image_path
                model_paths = arguments.get('image_path', [])
                
                if real_image_paths:
                    # Check if model provided paths that match the real paths
                    needs_replacement = False
                    
                    if not model_paths:
                        # No paths provided
                        needs_replacement = True
                        self.logger.info(f"No image_path provided by model for {tool_name}")
                    else:
                        # Check if model paths match real paths by comparing filenames
                        real_filenames = [os.path.basename(path) for path in real_image_paths]
                        model_filenames = [os.path.basename(str(path)) for path in model_paths]
                        
                        # Check if filenames match
                        if sorted(real_filenames) != sorted(model_filenames):
                            needs_replacement = True
                            self.logger.info(f"Model paths don't match real paths:")
                            self.logger.info(f"  Model filenames: {model_filenames}")
                            self.logger.info(f"  Real filenames: {real_filenames}")
                        else:
                            # Filenames match, but check if full paths are different (ignore order)
                            if sorted(model_paths) != sorted(real_image_paths):
                                needs_replacement = True
                                self.logger.info(f"Model paths have correct filenames but wrong directories:")
                                self.logger.info(f"  Model paths: {model_paths}")
                                self.logger.info(f"  Real paths: {real_image_paths}")
                    
                    if needs_replacement:
                        arguments['image_path'] = real_image_paths
                        self.logger.info(f"Replaced image paths for {tool_name}: {model_paths} -> {real_image_paths}")
                    else:
                        self.logger.info(f"Model provided correct image paths for {tool_name}: {model_paths}")
                else:
                    self.logger.warning(f"Could not find real image paths for {tool_name}, tool call may fail")
            
            # Add default angles if missing for pi3_tool
            if tool_name == 'pi3_tool':
                if 'azimuth_angle' not in arguments:
                    arguments['azimuth_angle'] = 0
                if 'elevation_angle' not in arguments:
                    arguments['elevation_angle'] = 0
            
            # Create fixed call
            fixed_call = {
                'name': tool_name,
                'arguments': arguments
            }
            fixed_calls.append(fixed_call)
        
        return fixed_calls

    def _execute_tools(self, tool_calls: List[Dict[str, Any]], video_path: Optional[str] = None, pi3_target_fps: float = 1.0) -> Dict[str, Any]:
        """
        Execute tool calls in parallel when possible
        
        Args:
            tool_calls: List of tool call dictionaries
            video_path: Optional path to original video (for pi3 tool re-sampling)
            pi3_target_fps: Target FPS for pi3 tool frame extraction
            
        Returns:
            Dictionary of tool_name -> result
        """
        tool_results = {}
        
        # Group tool calls by tool name to handle multiple calls to same tool
        tool_groups = {}
        for i, call in enumerate(tool_calls):
            tool_name = call['name']
            if tool_name not in tool_groups:
                tool_groups[tool_name] = []
            tool_groups[tool_name].append((i, call))
        
        # Execute tools in parallel, but Pi3Tool and Pi3MultiimgTool sequentially to avoid server issues
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_tool = {}
            
            # Handle Pi3Tool and Pi3MultiimgTool calls sequentially first
            pi3_calls = []
            other_calls = {}
            
            for tool_name, calls in tool_groups.items():
                if tool_name in ['pi3_tool', 'pi3_multiimg_tool']:
                    pi3_calls.extend(calls)
                else:
                    other_calls[tool_name] = calls
            
            # Execute Pi3Tool and Pi3MultiimgTool calls sequentially
            if pi3_calls:
                self.logger.info(f"Executing {len(pi3_calls)} Pi3 tool calls sequentially...")
                
                # Extract more frames for pi3 if video_path is provided
                pi3_frame_paths = []
                if video_path and Path(video_path).exists():
                    self.logger.info(f"Extracting frames for pi3 tool from video: {video_path} at {pi3_target_fps} fps")
                    pi3_frame_paths = self._extract_frames_for_pi3(video_path, pi3_target_fps)
                
                for call_idx, call in pi3_calls:
                    tool_name = call['name']
                    tool = self.tool_registry.get(tool_name)
                    if tool:
                        # If we extracted frames for pi3, update the arguments
                        arguments = call['arguments'].copy()
                        if pi3_frame_paths:
                            # Update image_path in arguments
                            if 'image_path' in arguments:
                                self.logger.info(f"Updating pi3 tool arguments with {len(pi3_frame_paths)} newly extracted frames")
                                arguments['image_path'] = pi3_frame_paths
                        
                        result = self._safe_tool_call(tool, arguments)
                        result_key = tool_name if len(pi3_calls) == 1 else f"{tool_name}_{call_idx}"
                        tool_results[result_key] = result
                        # Add small delay between Pi3 tool calls
                        import time
                        time.sleep(1)
                    else:
                        self.logger.error(f"{tool_name} not found")
                        result_key = tool_name if len(pi3_calls) == 1 else f"{tool_name}_{call_idx}"
                        tool_results[result_key] = {
                            "success": False,
                            "error": f"{tool_name} not found"
                        }
            
            # Execute other tools in parallel as before
            for tool_name, calls in other_calls.items():
                tool = self.tool_registry.get(tool_name)
                if tool is None:
                    self.logger.error(f"Tool not found: {tool_name}")
                    for _, call in calls:
                        tool_results[f"{tool_name}_{_}"] = {
                            "success": False,
                            "error": f"Tool not found: {tool_name}"
                        }
                    continue
                
                # Submit tool execution for each call
                for call_idx, call in calls:
                    future = executor.submit(self._safe_tool_call, tool, call['arguments'])
                    future_to_tool[future] = (tool_name, call_idx)
            
            # Collect results from parallel execution
            for future in as_completed(future_to_tool):
                tool_name, call_idx = future_to_tool[future]
                try:
                    result = future.result()
                    # Use unique key for multiple calls to same tool
                    result_key = tool_name if len([t for t in other_calls.get(tool_name, [])]) == 1 else f"{tool_name}_{call_idx}"
                    tool_results[result_key] = result
                except Exception as e:
                    self.logger.error(f"Tool execution failed for {tool_name}: {e}")
                    result_key = tool_name if len([t for t in other_calls.get(tool_name, [])]) == 1 else f"{tool_name}_{call_idx}"
                    tool_results[result_key] = {
                        "success": False,
                        "error": str(e)
                    }
        
        return tool_results

    def _safe_tool_call(self, tool, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Safely execute a tool call with error handling
        
        Args:
            tool: Tool instance
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        try:
            self.logger.info(f"Executing tool: {tool.name} with args: {arguments}")
            result = tool.call(**arguments)
            self.logger.info(f"Tool {tool.name} completed successfully")
            return result
        except Exception as e:
            self.logger.error(f"Tool {tool.name} execution failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _extract_frames_for_pi3(self, video_path: str, target_fps: float = 1.0) -> List[str]:
        """
        Extract frames from video for pi3 tool
        
        Args:
            video_path: Path to video file
            target_fps: Target frame rate
            
        Returns:
            List of paths to extracted frame images
        """
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.logger.error("cv2 is required for video frame extraction. Please install opencv-python.")
            return []
        
        cap = cv2.VideoCapture(video_path)
        
        # Get video info
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_duration = total_frames / original_fps
        
        # Calculate frames to extract based on target fps
        num_frames = int(total_duration * target_fps)
        frame_interval = total_frames / num_frames
        
        frame_paths = []
        temp_dir = Path("temp_frames_pi3")
        temp_dir.mkdir(exist_ok=True)
        
        # Extract video filename (without extension)
        video_filename = Path(video_path).stem
        
        # Extract frames evenly
        for i in range(num_frames):
            frame_idx = int(i * frame_interval)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_path = temp_dir / f"{video_filename}_pi3_frame_{i}.jpg"
                cv2.imwrite(str(frame_path), frame)
                frame_paths.append(str(frame_path))
        
        cap.release()
        self.logger.info(f"Extracted {len(frame_paths)} frames from video for pi3 tool (duration: {total_duration:.2f}s, original fps: {original_fps:.2f}, target fps: {target_fps})")
        return frame_paths

    def _cleanup_pi3_frames(self):
        """
        Clean up temporary frames extracted for pi3 tool
        """
        import shutil
        
        temp_dir = Path("temp_frames_pi3")
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                self.logger.info("Cleaned up temporary pi3 frames")
            except Exception as e:
                self.logger.warning(f"Failed to clean up temporary pi3 frames: {e}")

    def _create_continuation_prompt(
        self, 
        question: str, 
        last_response: str, 
        tool_results: Dict[str, Any],
        original_images: List[str],
        additional_images: List[str],
        current_turn: int,
        max_turns: int
    ) -> str:
        """
        Create continuation prompt for multi-turn workflow
        
        Args:
            question: Original user question
            last_response: Last model response
            tool_results: Tool results from this turn
            original_images: Original image paths
            additional_images: Additional images generated in this turn
            current_turn: Current turn number
            max_turns: Maximum turns allowed
            
        Returns:
            Continuation prompt string
        """
        tool_summary = []
        angle_info = []
        
        for tool_name, result in tool_results.items():
            if result.get('success'):
                tool_summary.append(f"- {tool_name}: Successfully executed")
                
                # Extract angle information if available
                if 'azimuth_angle' in result and 'elevation_angle' in result:
                    azim = result.get('azimuth_angle')
                    elev = result.get('elevation_angle')
                    angle_info.append(f"  └─ Viewing angle: azimuth={azim}°, elevation={elev}°")
                
                if result.get('description'):
                    tool_summary.append(f"  Description: {result.get('description')}")
            else:
                tool_summary.append(f"- {tool_name}: Failed - {result.get('error', 'Unknown error')}")
        
        tool_summary_text = "\n".join(tool_summary) if tool_summary else "None"
        angle_info_text = "\n".join(angle_info) if angle_info else ""
        
        original_images_info = "\n".join([f"- {path}" for path in original_images]) if original_images else "None"
        additional_images_info = "\n".join([f"- {Path(path).name}" for path in additional_images]) if additional_images else "None"
        
        remaining = max_turns - current_turn
        is_second_to_last_turn = current_turn == max_turns - 1
        
        # Build dynamic prompt sections based on turn

        if is_second_to_last_turn:
            next_steps_section = f"""=== Next Steps ===

⚠️ **WARNING: This is turn {current_turn}/{max_turns} - Next turn will be the FINAL turn**

You have only 1 more turn available. You MUST provide your final answer now, as next turn is the final turn.

Instructions:
- Output your reasoning in <think></think> tags
- Then provide your final comprehensive answer in <answer></answer> tags
- Reference the specific viewpoints and tools that helped you understand the structure

IMPORTANT: You MUST start your response with <think>...</think> tags to explain your reasoning!
"""
        else:
            next_steps_section = f"""=== Next Steps ===

You have {remaining} more turn(s) available. You can:

# **Continue investigating** - Call tools with DIFFERENT parameters:
   - **IMPORTANT**: Your original input images are already at (azimuth=0°, elevation=0°). DO NOT call Pi3 tools with (0°, 0°) again!
   - For Pi3 tools: Try NEW viewing angles to understand the 3D structure better
   - Recommended NEW angles (NOT 0°,0°!):
     * Left: (-45°, 0°) or (-90°, 0°)
     * Right: (45°, 0°) or (90°, 0°)
     * Top: (0°, 45°) or (0°, 60°)
     * Bottom: (0°, -45°)
     * Back: (180°, 0°) or (±135°, 0°)
     * Diagonal: (45°, 30°) or (-45°, 30°)
   - Each NEW angle reveals different aspects of the 3D structure
   - You can just call Pi3 tool **once** in this turn.

   **Advanced Pi3 Parameters**:
   - **rotation_reference_camera** (integer, 1-based): When you have multiple input images, try DIFFERENT camera positions as rotation centers
     * Default is 1 (first camera), Set to 2, 3, etc. to rotate around different camera positions
     * Example: rotation_reference_camera=2 rotates around the second camera's viewpoint
     * Useful for analyzing different parts of the scene from various perspectives
   
   - **camera_view** (boolean): Control the visualization perspective
     * False (default): Global bird's-eye view showing the entire scene
     * True: First-person camera view - see the scene from the selected camera's perspective (as if standing at that camera)
     * Combine with rotation_reference_camera to experience different camera viewpoints
     * Example: camera_view=True with rotation_reference_camera=2 shows first-person view from camera 2
     * Useful for understanding what each camera can see and spatial relationships

Instructions:
- Think: Do you need to see the object from another NEW angle (NOT 0°,0°!) to answer the question better?
- If YES: Use <tool_call></tool_call> to request a DIFFERENT viewing angle (avoid 0°,0° as you already have it!)
- If NO: output your thinking process in <think></think> and your final answer in <answer></answer>. Only put letters of option in <answer></answer> tags, do not put any other text.
- **Do not request the same angle as before.**

Note that in 3D reconstruction, the camera numbering corresponds directly to the image numbering — cam1 represents the first frame.
You can examine the image to understand what is around cam1.
The 3D reconstruction provides relative positional information, so you should reason interactively and complementarily between the 2D image and the 3D reconstruction to form a complete understanding.

# **Provide final answer** - If you have sufficient information from current viewpoints:
   - Output your comprehensive analysis in <think></think> tags
   - Reference the specific viewpoints that helped you understand the structure
   - if you want to provide final answer, the reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>
"""
        
        if is_second_to_last_turn:
            prompt = f"""

⚠️ **WARNING: This is the FINAL turn**
Original Question: {question}



Original Images:
{original_images_info}

The 3D Reconstruction image.
{additional_images_info}

Note that in 3D reconstruction, the camera numbering corresponds directly to the image numbering — cam1 represents the first frame.
You can examine the image to understand what is around cam1.
The 3D reconstruction provides relative positional information, so you should reason interactively and complementarily between the 2D image and the 3D reconstruction to form a complete understanding.

All available tools have been used up, and you can no longer call any additional tools. You have no remaining steps for tool invocation. You can now see the different perspectives generated by the previous tool calls, as well as the original image. Please use the provided content to answer the original question.

You MUST NOT call any tools.  
You MUST NOT output <tool_call>.  
You MUST directly reason and answer in this round.  
All reasoning MUST be written explicitly within <think> </think> tags.  
The final answer MUST be written within <answer> </answer> tags.  

Format strictly as follows:
<think>[Your reasoning process here — show step-by-step thinking, explanations, or derivations]</think><answer>[Your final answer here — only put your choice here]</answer>

Example:
<think>First, analyze the question carefully. Then, derive the solution using logical reasoning.</think><answer>A</answer>
"""
        else:
            prompt = f"""=== Multi-Turn Analysis: Turn {current_turn}/{max_turns - 1} ===

    Original Question: {question}

    Your Previous Response: 
    {last_response}

    Tool Execution Summary:
    {tool_summary_text}
    {angle_info_text}

    Original Images:
    {original_images_info}

    Generated Images Available for Analysis:
    {additional_images_info}

    {next_steps_section}

    Please continue:"""
        
        
        
        return prompt

    def check_finished(
        self, 
        infer_request: RolloutInferRequest,
        response_choice: ChatCompletionResponseChoice,
        current_turn: int
    ) -> bool:
        # Log tool status on first turn for debugging
        if current_turn == 1 and not hasattr(self, '_logged_tool_status'):
            self._logged_tool_status = True
            registered_tools = self.tool_registry.list_tools()
            if registered_tools:
                self.logger.info(f"SPAgent scheduler active with {len(registered_tools)} tools: {registered_tools}")
            else:
                self.logger.warning("SPAgent scheduler active but NO TOOLS registered!")
        
        # If model output includes <answer>…</answer>, finish
        content = response_choice.message.content
        if '<answer>' in content and '</answer>' in content:
            return True

        # Fallback to default logic (truncation or max turns)
        return super().check_finished(infer_request, response_choice, current_turn)

    def step(
        self,
        infer_request: RolloutInferRequest,
        response_choice: ChatCompletionResponseChoice,
        current_turn: int
    ) -> dict:
        """
        Build next‐turn infer_request based on current response.
        If there is a tool call, execute it and feed result back into next message content.
        """
        content = response_choice.message.content

        # 初始化：收集工具生成的新图片
        collected_new_images = []

        # 辅助函数：转换为绝对路径
        def to_absolute_path(path_str):
            """Convert to absolute path and return as string"""
            if path_str:
                return str(Path(path_str).resolve())
            return path_str

        # ✅ 保存原始输入图片和原始问题（只在第一轮保存）
        if current_turn == 1:
            # 第一轮时，保存原始图片和问题到 data_dict
            if not hasattr(infer_request, 'data_dict') or infer_request.data_dict is None:
                infer_request.data_dict = {}
            if 'original_images' not in infer_request.data_dict:
                infer_request.data_dict['original_images'] = infer_request.images or []
            if 'original_question' not in infer_request.data_dict:
                # 提取原始问题
                for msg in infer_request.messages:
                    if msg['role'] == 'user':
                        infer_request.data_dict['original_question'] = msg['content']
                        break

        # 1) parse tool calls
        # calls = self._parse_tool_calls(content)
        calls = [self._parse_tool_calls(content)[0]]
        if calls:
            self.logger.info(f"Detected {len(calls)} tool calls: {[call['name'] for call in calls]}")
            
            # Check if any tools are registered
            if not self.tool_registry.list_tools():
                self.logger.error(
                    "Model attempted to call tools but no tools are registered! "
                    "Please register tools using scheduler.register_tool() or check auto-registration."
                )
                # Return empty results to avoid crashes
                tool_results = {}
                for i, call in enumerate(calls):
                    tool_results[call['name']] = {
                        'success': False,
                        'error': 'No tools registered in scheduler'
                    }
            else:
                # Validate and fix tool calls (e.g., add missing image_path)
                fixed_calls = self._validate_and_fix_tool_calls(calls, infer_request)
                
                # Execute all tool calls using parallel execution
                tool_results = self._execute_tools(fixed_calls)
            
            # Collect new images from tool results
            for result_key, result in tool_results.items():
                if result.get('success'):
                    # Add output paths if available
                    if 'output_path' in result:
                        collected_new_images.append(to_absolute_path(result['output_path']))
                    if 'vis_path' in result:
                        collected_new_images.append(to_absolute_path(result['vis_path']))
            
            # Get original question from data_dict or extract from messages
            original_question = ""
            if hasattr(infer_request, 'data_dict') and infer_request.data_dict:
                original_question = infer_request.data_dict.get('original_question', '')
            
            if not original_question:
                # Fallback: extract from messages
                for msg in infer_request.messages:
                    if msg['role'] == 'user':
                        original_question = msg['content']
                        break
            
            # Get original images
            original_images = []
            if hasattr(infer_request, 'data_dict') and infer_request.data_dict:
                original_images = infer_request.data_dict.get('original_images', [])
            
            # Get max_turns from args
            max_turns = getattr(self, 'max_turns', 3)
            
            # Create smart continuation prompt
            continuation_prompt = self._create_continuation_prompt(
                question=original_question,
                last_response=content,
                tool_results=tool_results,
                original_images=original_images,
                additional_images=collected_new_images,
                current_turn=current_turn,
                max_turns=max_turns
            )
            
            # ✅ 构建极简消息列表：只有 continuation prompt
            next_messages = [{
                "role": "user",
                "content": continuation_prompt
            }]

        else:
            # No tool calls: 构建简洁的最终答案提示
            next_messages = [{
                "role": "user",
                "content": """Based on your analysis, please provide your final answer.
                            
Instructions:
- You MUST start with <think></think> tags to explain your reasoning
- Then provide your final answer in <answer></answer> tags
- Reference any relevant information from the images and previous analysis

IMPORTANT: Always output <think>...</think> followed by <answer>...</answer>!"""
            }]

        # ✅ 只保留原图和本轮新生成的图
        original_images = []
        if hasattr(infer_request, 'data_dict') and infer_request.data_dict:
            original_images = infer_request.data_dict.get('original_images', [])

        new_images = original_images + collected_new_images

        if collected_new_images:
            self.logger.info(f"Total images for next turn: {len(original_images)} original + {len(collected_new_images)} new = {len(new_images)} total")
            self.logger.debug(f"New image paths: {new_images}")

        # ✅ 构建新请求时，传递 data_dict（包含原始图片）
        new_data_dict = infer_request.data_dict if hasattr(infer_request, 'data_dict') else {}

        # Build new infer_request for next turn
        new_request = RolloutInferRequest(
            messages=next_messages,
            images=new_images,
            audios=infer_request.audios,
            videos=infer_request.videos,
            tools=infer_request.tools,
            objects=infer_request.objects,
            data_dict=new_data_dict
        )
        
        # Clean up temporary pi3 frames if any were created
        self._cleanup_pi3_frames()
        
        return {"infer_request": new_request}

multi_turns['spagent_tool_call_scheduler'] = SPAgentToolCallingScheduler


# register GYM env
class CustomEnv(Env):
    pass


envs['custom_env'] = CustomEnv


class CustomCtxManager(ContextManager):
    pass


context_managers['custom_ctx'] = CustomCtxManager
