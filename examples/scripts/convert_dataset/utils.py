from string import Template

ACTION_SPACE = """
<action>
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
</action>
"""

ACTION_OUTPUT_FORMAT = """
<action_description>
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
</action>
"""

QUERY_TEMPLATE = Template(
    """
Given the task:
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
Output the thinking process in <think></think> and final answer in <answer> </answer> tags, i.e., <think>your thinking process</think><answer>your action</answer>.
"""
)
