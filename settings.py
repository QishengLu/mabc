import os

# shubiaobiao — Claude Sonnet 4.5
OPENAI_API_KEY = os.environ.get(
    "OPENAI_API_KEY",
    "sk-TisdMcaR6FpD4yVrF63bA49cFe86429881370aB3E6629496",
)
OPENAI_BASE_URL = os.environ.get(
    "OPENAI_BASE_URL",
    "https://api.shubiaobiao.cn/v1",
)

OPENAI_MAX_RETRIES = 10
OPENAI_RETRY_SLEEP = 30
# OPENAI_MODEL = "gpt-4-turbo"
# OPENAI_MODEL = "kimi-k2-0905-preview"
# OPENAI_MODEL = "anthropic/claude-sonnet-4.5"
OPENAI_MODEL = "claude-sonnet-4-6"

# AGENT_STATUS_START = "Start"
# AGENT_STATUS_RE = "Reason"
# AGENT_STATUS_ACT = "Act"
# AGENT_STATUS_FINISH = "Finish"

# STOP_WORDS_REACT = "\nObservation"
# STOP_WORDS_NONE = ""

# ACTION_FAILURE = "action执行失败"
# DEBUG = False


# TOT_CHILDREN_NUM = 1

# TOT_MAX_DEPTH = 15

# # DEFAULT_MODEL = "gpt-3.5-turbo"  # gpt-3.5 -turbo-16k-0613
# # DEFAULT_MODEL = "gpt-4"  # gpt-3.5-turbo-16k-0613
# DEFAULT_MODEL = "gpt-3.5-turbo-0125"

