import re
from ast import literal_eval
from typing import Tuple


def extract_xml(text: str, tag: str) -> str:
    """
    Extracts the content of the specified XML tag from the given text. Used for parsing structured responses

    Args:
        text (str): The text containing the XML.
        tag (str): The XML tag to extract content from.

    Returns:
        str: The content of the specified XML tag, or an empty string if the tag is not found.
    """
    match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1) if match else ""


def parse_parameter(parameter: str) -> dict:
    """
    Parser the parameter string into a dictionary.
    key is the name of the parameter, value is the value of the parameter.
    parameter is like:
    '<param_name>point</param_name><param_value>[426, 270]</param_value>' -> {'point': [426, 270]}
    '<param_name>direction</param_name><param_value>up</param_value>' -> {'direction': 'up'}
    '<param_name>text</param_name><param_value>Click to manage account information.</param_value>' -> {'text': 'Click to manage account information.'}
    'region: [20, 300, 400, 500]' -> {'region': [20, 300, 400, 500]}
    """
    try:
        key = None
        value = None
        if parameter.startswith("region:"):
            # 解析gt数据中的region内容
            parameter = parameter.strip().split(":", 1)
            key = parameter[0].strip()
            value = literal_eval(parameter[1].strip())
        else:
            param_name = extract_xml(parameter, "param_name").strip()
            key = param_name
            param_value = extract_xml(parameter, "param_value").strip()
            if param_name in ["point"]:
                value = literal_eval(param_value)
            else:
                value = param_value
        # print(f"key: {key}, value: {value}")
        if key is None or key == "":
            return {}
        return {key: value}
    except Exception as e:
        print(f"Error parsing parameter: {e}")
        print(f"Parameter: {parameter}")
        return {}


def extract_action(text: str) -> str:
    """
    Extract the ground truth action from the anwser.
    <answer>
        <action_description>
            Click on the ['My account\nManage account info'] to (Click to manage account information.)
        </action_description>
        <action>
            <action_name>
                TAP
            </action_name>
            <parameters>
                <parameter>
                    <param_name>point</param_name>
                    <param_value>[426, 270]</param_value>
                </parameter>
            </parameters>
        </action>
        <active_region>
            region: [20, 300, 400, 500]
        </active_region>
    </answer>

    a well defined action contains: name, parameters, active_region
    """
    answer = extract_xml(text, "answer").strip()
    action = extract_xml(answer, "action").strip()
    action_name = extract_xml(action, "action_name").strip()
    parameter = extract_xml(action, "parameter").strip()
    # print(f"parameter: {parameter}")
    parameter = parse_parameter(parameter)
    active_region = parse_parameter(extract_xml(answer, "active_region").strip())
    return action_name, parameter, active_region


def point_in_region(point: Tuple[int, int], region: Tuple[int, int, int, int]) -> bool:
    """
    Check if the point is in the region.
    """
    return region[0] <= point[0] <= region[2] and region[1] <= point[1] <= region[3]


def eval_action(gt_action, gt_parameter, gt_active_region, action, parameter) -> float:
    """
    Evaluate the action is correct and return the reward.
    Action Space includes:
        TAP, SWIPE, TYPE, with parameters: point, direction, text
        TASK_COMPLETE, PRESS_ENTER, TASK_IMPOSSIBLE, PRESS_BACK, PRESS_HOME, WAIT, with no parameters
    """

    def lcs(s1, s2):
        """
        Calculate the longest common subsequence (LCS) of two strings.
        """
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    reward = 0.0

    if gt_action != action:
        return reward  # 动作不一致，奖励为0
    if isinstance(gt_active_region, dict):
        gt_active_region = gt_active_region.get("region", None)

    if gt_action == "TAP":
        point = parameter.get("point", None)
        if isinstance(point, list) and isinstance(gt_active_region, list) and point_in_region(point, gt_active_region):
            reward += 1  # 点在区域内，奖励1

    elif gt_action == "SWIPE":
        direction = parameter.get("direction", None)
        gt_direction = gt_parameter.get("direction", None)
        if direction is not None and direction in ["up", "down", "left", "right"] and direction == gt_direction:
            reward += 1  # 方向一致，奖励1

    elif gt_action == "TYPE":
        text = parameter.get("text", None)
        gt_text = gt_parameter.get("text", None)
        if text is not None and lcs(text, gt_text) / max(len(text), len(gt_text)) > 0.3:
            reward += 1  # 内容相似，奖励1

    elif gt_action == "TASK_COMPLETE" and parameter == {}:
        reward += 1  # 动作一致，奖励1

    elif gt_action == "PRESS_ENTER" and parameter == {}:
        reward += 1  # 动作一致，奖励1

    elif gt_action == "TASK_IMPOSSIBLE" and parameter == {}:
        reward += 1  # 动作一致，奖励1

    elif gt_action == "PRESS_BACK" and parameter == {}:
        reward += 1  # 动作一致，奖励1

    elif gt_action == "PRESS_HOME" and parameter == {}:
        reward += 1  # 动作一致，奖励1

    elif gt_action == "WAIT" and parameter == {}:
        reward += 1  # 动作一致，奖励1
    else:
        reward += 0.0  # 动作不一致，奖励为0

    return reward


def action_accuracy_reward(completion, solution):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    gt_action, gt_parameter, gt_active_region = extract_action(solution)
    action, parameter, _ = extract_action(completion)
    # print(f"completion: {completion}")
    # print(f"gt_action: {gt_action}, gt_parameter: {gt_parameter}, gt_active_region: {gt_active_region}")
    # print(f"solution: {solution}")
    # print(f"action: {action}, parameter: {parameter}")
    reward = 0.0
    reward += eval_action(gt_action, gt_parameter, gt_active_region, action, parameter)
    return reward


def format_reward(completion, solution=None):
    """Reward function that checks if the completion has a specific format."""
    format_pattern = r"^<think>.*?</think><answer>.*?</answer>$"  # 设定输出内容 <think>...</think><answer>...</answer>
    # TODO 额外加一个格式检查，action中包含 <action_description> 和 <action> 标签, action_description 只是为了增加输出内容可读性。
    action_desc_pattern = r"<action_description>.*?</action_description>"
    action_pattern = r"<action>.*?</action>"

    reward = 0.0
    match = re.match(format_pattern, completion, re.DOTALL)
    if match:
        reward += 1.0

    answer = extract_xml(completion, "answer")
    action_desc_count = len(re.findall(action_desc_pattern, answer, re.DOTALL))
    action_count = len(re.findall(action_pattern, answer, re.DOTALL))

    if action_count == 1:
        reward += 1.0

    if action_desc_count == 1:
        reward += 1.0

    return reward


if __name__ == "__main__":

    wrong_format0 = """<thinking>I think I need to tap on the button to manage account information.</thinking>
    <act>I need to tap on the button to manage account information.</act>"""
    wrong_format1 = """<think>
        I think I need to tap on the button to manage account information.
    </think>
    <answer>
        I need to tap on the button to manage account information.
    </answer>"""
    wrong_format2 = """<think>
        I think I need to tap on the button to manage account information.
    </think><answer>
        I need to tap on the button to manage account information.
    </answer>"""
    wrong_format3 = """<think>
        I think I need to tap on the button to manage account information.
    </think><answer>
        <action_description>
            Click on the ['My account\nManage account info'] to (Click to manage account information.)
        </action_description>
    </answer>"""
    wrong_format4 = """<think>
        I think I need to tap on the button to manage account information.
    </think><answer>
        <action>
            <action_name>TAP</action_name>
        </action>
    </answer>"""

    test_completion_tap = """<think>
        I think I need to tap on the button to manage account information.
    </think><answer>
        <action_description>
            Click on the ['My account\nManage account info'] to (Click to manage account information.)
        </action_description>
        <action>
            <action_name>TAP</action_name>
            <parameters>
                <parameter>
                    <param_name>point</param_name>
                    <param_value>[426, 270]</param_value>
                </parameter>
            </parameters>
        </action>
    </answer>"""
    test_completion_tap_wrong1 = """<think>
        I think I need to tap on the button to manage account information.
    </think><answer>
        <action_description>
            Click on the ['My account\nManage account info'] to (Click to manage account information.)
        </action_description>
        <action>
            <action_name>TAP</action_name>
            <parameters>
                <parameter>
                    <param_name>point</param_name>
                    <param_value>[4, 20]</param_value>
                </parameter>
            </parameters>
        </action>
    </answer>"""
    test_completion_swipe = """<think>
        I think I need to swipe up to manage account information.
    </think><answer>
        <action_description>
            Click on the ['My account\nManage account info'] to (Click to manage account information.)
        </action_description>
        <action>
            <action_name>SWIPE</action_name>
            <parameters>
                <parameter>
                    <param_name>direction</param_name>
                    <param_value>up</param_value>
                </parameter>
            </parameters>
        </action>
    </answer>"""
    test_completion_swipe_wrong1 = """<think>
        I think I need to swipe up to manage account information.
    </think><answer>
        <action_description>
            Click on the ['My account\nManage account info'] to (Click to manage account information.)
        </action_description>
        <action>
            <action_name>SWIPE</action_name>
            <parameters>
                <parameter>
                    <param_name>direction</param_name>
                    <param_value>down</param_value>
                </parameter>
            </parameters>
        </action>
    </answer>"""
    test_completion_swipe_wrong2 = """<think>
        I think I need to swipe up to manage account information.
    </think><answer>
        <action_description>
            Click on the ['My account\nManage account info'] to (Click to manage account information.)
        </action_description>
        <action>
            <action_name>SWIPE</action_name>
            <parameters>
            </parameters>
        </action>
    </answer>"""
    test_completion_type = """<think>
        I think I need to type the text to manage account information.
    </think><answer>
        <action_description>
            Click on the ['My account\nManage account info'] to (Click to manage account information.)
        </action_description>
        <action>
            <action_name>TYPE</action_name>
            <parameters>
                <parameter>
                    <param_name>text</param_name>
                    <param_value>Click to manage account information.</param_value>
                </parameter>
            </parameters>
        </action>
    </answer>"""
    test_completion_type_wrong1 = """<think>
        I think I need to type the text to manage account information.
    </think><answer>
        <action_description>
            Click on the ['My account\nManage account info'] to (Click to manage account information.)
        </action_description>
        <action>
            <action_name>TYPE</action_name>
            <parameters>
                <parameter>
                    <param_name>text</param_name>
                    <param_value>Click</param_value>
                </parameter>
            </parameters>
        </action>
    </answer>"""

    test_completion_complete = """<think>
        I think I need to complete the task.
    </think><answer>
        <action_description>
            Click on the ['My account\nManage account info'] to (Click to manage account information.)
        </action_description>
        <action>
            <action_name>TASK_COMPLETE</action_name>
        </action>
    </answer>"""

    test_completion_complete_wrong1 = """<think>
        I think I need to complete the task.
    </think><answer>
        <action_description>
            Click on the ['My account\nManage account info'] to (Click to manage account information.)
        </action_description>
        <action>
            <action_name>TASK_COMPLETE</action_name>
            <parameters>
                <parameter>
                    <param_name>text</param_name>
                    <param_value>Click to manage account information.</param_value>
                </parameter>
            </parameters>
        </action>
    </answer>"""

    gt_tap = """<answer>
        <action_description>
            Click on the ['My account\nManage account info'] to (Click to manage account information.)
        </action_description>
        <action>
            <action_name>TAP</action_name>
            <parameters>
                <parameter>
                    <param_name>point</param_name>
                    <param_value>[426, 270]</param_value>
                </parameter>
            </parameters>
        </action>
        <active_region>
            region: [300, 250, 500, 300]
        </active_region>
    </answer>"""
    gt_swipe = """<answer>
        <action_description>
            Click on the ['My account\nManage account info'] to (Click to manage account information.)
        </action_description>
        <action>
            <action_name>SWIPE</action_name>
            <parameters>
                <parameter>
                    <param_name>direction</param_name>
                    <param_value>up</param_value>
                </parameter>
            </parameters>
        </action>
    </answer>"""
    gt_type = """<answer>
        <action_description>
            Click on the ['My account\nManage account info'] to (Click to manage account information.)
        </action_description>
        <action>
            <action_name>TYPE</action_name>
            <parameters>
                <parameter>
                    <param_name>text</param_name>
                    <param_value>Click to manage account information.</param_value>
                </parameter>
            </parameters>
        </action>
    </answer>"""
    gt_task_complete = """<answer>
        <action_description>
            Click on the ['My account\nManage account info'] to (Click to manage account information.)
        </action_description>
        <action>
            <action_name>TASK_COMPLETE</action_name>
        </action>
    </answer>"""
    gt_press_enter = """<answer>
        <action_description>
            Click on the ['My account\nManage account info'] to (Click to manage account information.)
        </action_description>
        <action>
            <action_name>PRESS_ENTER</action_name>
        </action>
    </answer>"""

    # format reward test
    r = format_reward(test_completion_tap, gt_tap)
    assert r == 3.0, f"test_completion_tap: {r}"
    r = format_reward(wrong_format0, gt_tap)
    assert r == 0.0, f"wrong_format0: {r}"
    r = format_reward(wrong_format1, gt_tap)
    assert r == 0.0, f"wrong_format1: {r}"
    r = format_reward(wrong_format2, gt_tap)
    assert r == 1.0, f"wrong_format2: {r}"
    r = format_reward(wrong_format3, gt_tap)
    assert r == 2.0, f"wrong_format3: {r}"
    r = format_reward(wrong_format4, gt_tap)
    assert r == 2.0, f"wrong_format4: {r}"

    # parse_parameter test
    r = parse_parameter("<param_name>point</param_name><param_value>[426, 270]</param_value>")
    assert r == {"point": [426, 270]}, f"parse_parameter: {r}"
    r = parse_parameter("<param_name>direction  \n</param_name><param_value>up</param_value>")
    assert r == {"direction": "up"}, f"parse_parameter: {r}"

    # action eval test
    r = eval_action("TAP", {"point": [426, 270]}, {"region": [300, 250, 500, 300]}, "TAP", {"point": [426, 270]})
    assert r == 1.0, f"eval_action: {r}"

    # action accuracy reward test
    r = action_accuracy_reward(test_completion_tap, gt_tap)
    assert r == 1.0, f"test_completion_tap: {r}"
    r = action_accuracy_reward(test_completion_tap_wrong1, gt_tap)
    assert r == 0.0, f"test_completion_tap_wrong1: {r}"

    r = action_accuracy_reward(test_completion_swipe, gt_swipe)
    assert r == 1.0, f"test_completion_swipe: {r}"

    r = action_accuracy_reward(test_completion_swipe_wrong1, gt_swipe)
    assert r == 0.0, f"test_completion_swipe_wrong1 up != down: {r}"

    r = action_accuracy_reward(test_completion_swipe, gt_type)
    assert r == 0.0, f"swipe against type: {r}"

    r = action_accuracy_reward(test_completion_swipe_wrong2, gt_type)
    assert r == 0.0, f"test_completion_swipe_wrong2 with no parameters: {r}"

    r = action_accuracy_reward(test_completion_type, gt_type)
    assert r == 1.0, f"test_completion_type: {r}"

    r = action_accuracy_reward(test_completion_type_wrong1, gt_type)
    assert r == 0.0, f"test_completion_type_wrong1 lcs < 0.5: {r}"

    r = action_accuracy_reward(test_completion_complete, gt_task_complete)
    assert r == 1.0, f"test_completion_complete: {r}"

    r = action_accuracy_reward(test_completion_complete_wrong1, gt_task_complete)
    assert r == 0.0, f"test_completion_complete_wrong1 with parameters: {r}"
