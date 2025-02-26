from string import Template

ACTION_SPACE = """<action>
    <action_name>
        SWIPE
    </action_name>
    <description>
        Swipe the screen in one direction
    </description>
    <parameters>
        <parameter>
            <param_name>
                direction
            </param_name>
            <param_description>
                the direction to swipe, only left, right, up, down are allowed
            </param_description>
        </parameter>
    </parameters>
</action>
<action>
    <action_name>
        TAP
    </action_name>
    <description>
        Tap the target element on the screen
    </description>
    <parameters>
        <parameter>
            <param_name>
                point
            </param_name>
            <param_description>
                the coordinates of the target element on the screen, format is like [x, y], 
                x and y are the coordinates of the target element on the screen, range from 0 to 999
            </param_description>
        </parameter>
    </parameters>
</action>
<action>
    <action_name>
        TYPE
    </action_name>
    <description>
        Type the text in the text input field
    </description>
    <parameters>
        <parameter>
            <param_name>
                text
            </param_name>
            <param_description>
                the text string to input
            </param_description>
        </parameter>
    </parameters>
</action>
<action>
    <action_name>
        TASK_COMPLETE
    </action_name>
    <description>
        The task is completed based on the current state of the screen
    </description>
</action>
<action>
    <action_name>
        PRESS_ENTER
    </action_name>
    <description>
        Press the enter key on the keyboard
    </description>
</action>
<action>
    <action_name>
        TASK_IMPOSSIBLE
    </action_name>
    <description>
        The task is impossible to be completed based on the current state of the screen   
    </description>
</action>
<action>
    <action_name>
        PRESS_BACK
    </action_name>
    <description>
        Press the back key on the keyboard
    </description>
</action>
<action>
    <action_name>
        PRESS_HOME
    </action_name>
    <description>
        Press the home key on the keyboard
    </description>
</action>
<action>
    <action_name>
        WAIT
    </action_name>
    <description>
        Wait for a while
    </description>
</action>"""

ACTION_OUTPUT_FORMAT = """<action_description>
    ACTION_DESCRIPTION_FOR_SHORT_IN_NATURAL_LANGUAGE
</action_description>
<action>
    <action_name>
        ACTION_NAME
    </action_name>
    <parameters> // SKIP IF THE ACTION HAS NO PARAMETERS
        <parameter>
            <param_name>
                PARAM_NAME
            </param_name>
            <param_value>
                PARAM_VALUE
            </param_value>
        </parameter>
    </parameters>
</action>"""

QUERY_TEMPLATE = Template(
"""Given the task:
<task>
    $instruction
</task>

Here is the action space you may take:
<action space>
    $action_spaces
</action space>

The output action format should look like this:
<output action format>
    $output_action_format
</output action format>

Please think step by step to find out the proper action to take. And only output one action at a time.
Output the thinking process in <think></think> and final answer in <answer> </answer> tags, i.e., <think>your thinking process</think><answer>your action</answer>."""
)


OUTPUT_EXAMPLE = """
<think>
    "Okay, let's see. The user wants to open TodayTix, search for \"Les Miserables,\" and check the reviews. Let me check the current screen layout provided.\n\nLooking at the current elements, there's a search button with the description ['Search', 'Search'] and a bbox [319, 909, 499, 972]. The first step should be to tap on that search button to open the search bar. That's the most logical starting point because searching requires accessing the search function.\n\nOnce I tap the search button, the app should open a text input field where I can type \"Les Miserables.\" Since the current elements don't show a search box yet, the initial action must be to open it. After typing, I'd need to press enter to submit the search. Then, the app should display results for Les Miserables. From there, the user can select the show and check the reviews.\n\nBut wait, after searching, how do the reviews get checked? Maybe each show listing has a reviews section. So after searching, the user would need to find the correct show in the results and tap on it to view details, including reviews. However, the current layout doesn't have Les Miserables listed. The visible elements are for Fawlty Towers and other shows, so perhaps the user needs to perform the search first.\n\nSo the steps would be: Tap Search, type \"Les Miserables,\" press enter, wait for search results, then find the show and check reviews. But since the current screen doesn't show Les Miserables, the first actions are to initiate the search. Let's proceed step by step.
</think><answer>
    <description>
        In this step, I will tap the Search button to open the search input field.
    </description>
    <action>
        <action_name>
            TAP
        </action_name>
        <parameters>
            <parameter>
                <param_name>
                    point
                </param_name>
                <param_value>
                    [409, 940.5]
                </param_value>
            </parameter>
        </parameters>
    </action>
</answer>
"""

QUERY_TEMPLATE_WITH_FEW_SHOT = Template(
"""Given the task:
<task>
    $instruction
</task>

Here is the action space you may take:
<action space>
    $action_spaces
</action space>

Here is an output example:
<output example>
    $output_example
</output example>

Please think step by step to find out the proper action to take. And only output one action at a time.
Output the thinking process in <think></think> and final answer in <answer> </answer> tags, i.e., <think>your thinking process</think><answer>your action</answer>."""
)
