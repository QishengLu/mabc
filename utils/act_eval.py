import re


def _fix_action_string(action):
    """Fix common LLM output issues where string args are not quoted.

    e.g. ask_for_data_detective(question=What are the metrics for ts-ui-dashboard)
    →    ask_for_data_detective(question="What are the metrics for ts-ui-dashboard")
    """
    # Match: func_name(key=value) where value is an unquoted string
    m = re.match(r'^(\w+)\((\w+)=(.+)\)$', action, re.DOTALL)
    if m:
        func_name, param_name, param_value = m.group(1), m.group(2), m.group(3)
        # If param_value is already quoted, leave it alone
        param_value = param_value.strip()
        if not (param_value.startswith('"') or param_value.startswith("'")):
            # Escape any internal quotes
            param_value = param_value.replace('\\', '\\\\').replace('"', '\\"')
            action = f'{func_name}({param_name}="{param_value}")'
    return action


def act_eval(action, tool_env):
    try:
        action_result = eval(action, tool_env)
    except (SyntaxError, NameError, TypeError):
        # Try fixing unquoted string arguments
        fixed = _fix_action_string(action)
        try:
            action_result = eval(fixed, tool_env)
        except Exception as e:
            action_result = str(e)
    except Exception as e:
        action_result = str(e)
    return action_result