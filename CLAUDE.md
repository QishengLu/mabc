# CLAUDE.md — mABC

mABC（multi-Agent Blockchain-inspired Collaboration）：基于多 Agent + 区块链投票的微服务 RCA 框架。

论文：EMNLP 2024 Findings — Zhang et al.

---

## 项目结构

```
mABC/
├── main/main.py                    # 主入口：遍历 label.json 执行 RCA
├── agents/
│   ├── base/profile.py             # 7 个 Agent 角色定义
│   ├── base/run.py                 # ReActTotRun (ReAct 循环) + ThreeHotCotRun (投票)
│   └── tools/                      # 各 Agent 的工具函数
│       ├── process_scheduler_tools.py   # ProcessScheduler 调度子 Agent
│       ├── data_detective_tools.py      # 查询服务指标
│       ├── dependency_explorer_tools.py # 查询服务调用拓扑
│       ├── probability_oracle_tools.py  # 评估故障概率
│       ├── fault_mapper_tools.py        # 维护 Fault Web
│       ├── solution_engineer_tools.py   # 综合分析给出 Root Cause
│       └── fault_web.py                 # FaultWeb 数据结构
├── data/                           # 数据目录（Python 包 + JSON 数据）
│   ├── __init__.py
│   ├── metric_collect.py           # MetricExplorer：查询 endpoint_stats.json
│   ├── trace_collect.py            # TraceExplorer：查询 endpoint_maps.json
│   └── cases/                      # 500 个测试 case（openrca2-lite 全量）
│       └── <case_name>/
│           ├── metric/endpoint_stats.json
│           ├── topology/endpoint_maps.json
│           └── label/label.json
├── utils/
│   ├── llm.py                      # OpenAI API 调用（支持 base_url）
│   ├── act_eval.py                 # eval() 执行 LLM 生成的 tool 调用
│   └── generate_tools.py           # 从 .py 文件提取函数签名生成 tool prompt
├── settings.py                     # API 配置（当前用 openai/claude-sonnet-4-5-20250929）
├── convert_all.py                  # 数据转换：parquet → mABC JSON（500 cases）
├── convert_data.py                 # 数据转换（旧版，10 cases 合并模式）
└── test_smoke.py                   # 冒烟测试脚本
```

---

## 核心架构

### 7 个 Agent

| Agent | 职责 | 工具 |
|-------|------|------|
| ProcessScheduler | 主协调器，调度子 Agent | ask_for_data_detective/dependency_explorer/... |
| DataDetective | 查询服务指标（calls, error_rate, avg_duration） | query_endpoint_stats, query_endpoint_metrics_in_range |
| DependencyExplorer | 查询服务调用拓扑（上下游依赖） | get_endpoint_downstream, get_endpoint_upstream, ... |
| ProbabilityOracle | 评估节点故障概率 | assess_fault_probability |
| FaultMapper | 维护 Fault Web 拓扑 | update_fault_web |
| SolutionEngineer | 综合分析，给出最终 Root Cause | query_previous_cases |
| AlertReceiver | 告警优先级排序（当前未使用） | — |

### 执行流程

```
main.py 遍历 label.json 中的 (timestamp, alert_service)
  │
  ├── Stage 1: ProcessScheduler (ReActTotRun)
  │   └── 多轮 ReAct 循环：Thought → Action(调用子Agent) → Observation
  │       ├── ask_for_data_detective("查 ts-ui-dashboard 在 13:08 的指标")
  │       ├── ask_for_dependency_explorer("查 ts-ui-dashboard 的下游服务")
  │       └── ... 直到给出 Final Answer
  │
  └── Stage 2: SolutionEngineer (ReActTotRun)
      └── 基于 Stage 1 分析结果，给出最终判定
          → "Root Cause Endpoint: XXX, Root Cause Reason: XXX"
```

### 投票机制

ThreeHotCotRun 实现区块链式投票（alpha/beta 阈值），但当前代码中 `alpha=-1, beta=-1` → **投票被 bypass，直接返回 True**。

---

## 数据格式

### 三个 JSON 文件（每个 case 独立）

**1. `metric/endpoint_stats.json`** — 服务级每分钟指标

```json
{
  "ts-ui-dashboard": {
    "2025-07-24 13:08:00": {
      "calls": 1591,
      "success_rate": 100.0,
      "error_rate": 0.0,
      "average_duration": 52.15,
      "timeout_rate": 0.0
    }
  }
}
```

**2. `topology/endpoint_maps.json`** — 服务调用关系图

```json
{
  "None": {
    "2025-07-24 13:08:00": ["loadgenerator", "ts-ui-dashboard"]
  },
  "ts-ui-dashboard": {
    "2025-07-24 13:08:00": ["ts-auth-service", "ts-order-service", "ts-travel-service"]
  }
}
```

**3. `label/label.json`** — 测试样本（告警入口 + 调用链）

```json
{
  "2025-07-24 13:08:00": {
    "ts-ui-dashboard": [
      ["ts-ui-dashboard", "ts-contacts-service"]
    ]
  }
}
```

---

## 数据转换（OpenRCA2-lite → mABC）

### 源数据

- **DB**: `/home/nn/SOTA-agents/RCAgentEval/openrca2-lite.db`（500 条 case 元数据）
- **Parquet**: `/home/nn/SOTA-agents/RolloutRunner/data/<case_name>/`
  - 481 个 case 的 parquet 直接在根目录
  - 19 个 case 的 parquet 在 `converted/` 子目录下，且列名不同

### 列名映射

| 原始 schema (481 cases) | converted schema (19 cases) |
|-------------------------|---------------------------|
| Timestamp | time |
| SpanId | span_id |
| ParentSpanId | parent_span_id |
| ServiceName | service_name |
| Duration | duration |
| StatusCode | attr.status_code |

### 转换逻辑

```
abnormal_traces.parquet
  │
  ├── 按 ServiceName + minute 分组 → endpoint_stats.json
  │   (calls, success_rate, error_rate, average_duration, timeout_rate)
  │
  ├── 匹配 SpanId → ParentSpanId → endpoint_maps.json
  │   (parent_service → [child_services] per minute)
  │
  └── injection.json + env.json → label.json
      (timestamp = ABNORMAL_START, alert = root cause 的上游服务)
```

### 转换命令

```bash
cd /home/nn/SOTA-agents/mABC

# 全量转换 500 cases
python convert_all.py

# 转换单个 case
python convert_all.py --case ts0-mysql-loss-67k278

# 转换前 N 个
python convert_all.py --limit 10
```

### 转换结果

- **500/500** case 全部成功（481 原始 + 19 converted schema）
- 输出：`mABC/data/cases/<case_name>/`
- 总大小：22 MB

---

## LLM 配置

当前使用 **openai/claude-sonnet-4-5-20250929**（Moonshot API）。

```python
# settings.py
OPENAI_API_KEY = "sk-hFYU4xzrYBc1vCv8r0Jv3QmVhouD1urQ1ccintG0XyJJq3Kd"
OPENAI_BASE_URL = "https://api.shubiaobiao.cn/v1"
OPENAI_MODEL = "openai/claude-sonnet-4-5-20250929"
```

也支持环境变量覆盖：
```bash
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

---

## 运行方式

### 冒烟测试（1 case）

```bash
cd /home/nn/SOTA-agents/mABC
python test_smoke.py
```

### 批量评测（run_batch.py）

`run_batch.py` 遍历所有 case，逐个运行 mABC 两阶段分析，提取 Root Cause 并与 ground truth 比较，计算 AC@1。

```bash
cd /home/nn/SOTA-agents/mABC

# 运行全部 500 cases
python run_batch.py

# 运行前 N 个
python run_batch.py --limit 10

# 运行单个 case
python run_batch.py --case ts0-mysql-loss-67k278

# 断点恢复（跳过已完成的 case）
python run_batch.py --resume --run-id run_20260311_143000

# 后台运行
nohup python -u run_batch.py --run-id full_eval > eval.log 2>&1 &
```

结果保存在 `results/<run_id>/`：
- 每个 case 一个 JSON 文件：`<case_name>.json`
- 汇总：`summary.json`（含 AC@1、total、correct、failed 等）

### RolloutRunner 集成

`agent_runner.py` 实现了 RolloutRunner 的标准 stdin/stdout 接口：

```bash
cd /home/nn/SOTA-agents/RolloutRunner

# 冒烟测试
python scripts/run_rollout.py --agent mabc --source_exp_id <exp_id> --limit 1

# 全量运行
nohup python -u scripts/run_rollout.py --agent mabc --source_exp_id <exp_id> \
  > rollout_mabc.log 2>&1 &

# 带每样本实时日志（通过 run_rollout_with_retry.py + --log_dir）
python scripts/run_rollout_with_retry.py --agent mabc-qwen \
  --source_exp_id mabc-qwen3.5-plus \
  --max_concurrency 1 --initial_concurrency 1 \
  --log_dir logs/tmux-parallel/mabc-qwen3.5-plus
```

配置文件：`RolloutRunner/configs/agents/mabc.yaml` (claude) / `mabc-qwen.yaml` (qwen3.5-plus)

agent_runner.py 的 case 数据查找策略：
1. 尝试匹配 `data_dir` 路径名到 `mABC/data/cases/`
2. 从 `question` 文本中提取 case 名
3. 若 `data_dir` 含 parquet 文件，实时转换为 mABC JSON

### `--log-file` 参数（2026-04-14 新增）

`agent_runner.py` 支持 `--log-file <path>` 参数，把所有 stderr 输出 + Python logging 同时写到文件：

```bash
python agent_runner.py --log-file logs/mabc_idx_5.log < payload.json
```

实现机制（参考 Deep_Research/agent_runner.py 和 aiq/agent_runner.py）：

1. **`_TeeStream` 类**：替换 `sys.stderr`，对每次 write 同时写到原始 stderr 和文件。mABC 通过 `print(..., file=sys.stderr)` 输出进度（`[mABC] ...`），这部分被 tee 捕获。
2. **`logging.FileHandler`**：附加到 root logger，捕获 httpx/OpenAI SDK 等库通过 Python logging 输出的内容（如 `HTTP Request: POST ... "HTTP/1.1 200 OK"`）。
3. **argparse with `parse_known_args()`**：支持 `--log-file` CLI 参数，未知参数不会报错。

用途：在 `run_rollout_with_retry.py --log_dir` 批量跑时，每个样本自动生成独立日志文件（`idx_N_sample_M.log`），可以 `tail -f` 实时查看任意样本的 ReAct 循环进度（例如 ProcessScheduler 调度哪个子 Agent、SolutionEngineer 给出什么结论）。

---

## 相对原项目的修改

| 文件 | 修改内容 |
|------|---------|
| `settings.py` | 改用 kimi API，支持 env var |
| `utils/llm.py` | 加 `base_url`，stop 参数空值处理，精简日志 |
| `utils/act_eval.py` | 修复 LLM 输出无引号字符串参数的 eval 问题 |
| `agents/base/run.py` | 加 MAX_REACT_STEPS=8 防无限循环 |
| `agents/tools/probability_oracle_tools.py` | 去掉多余的 self 参数 |
| `agents/tools/fault_mapper_tools.py` | 修复 import 路径，去掉 self |
| `agents/tools/solution_engineer_tools.py` | 去掉 self |
| `agents/tools/dependency_explorer_tools.py` | 拷贝自 denpendency（修正拼写） |
| `data/__init__.py` | 新建 Python 包 |
| `data/metric_collect.py` | 新建，从 handle/ 迁移，支持 `set_case_data_dir()` 动态切换 case |
| `data/trace_collect.py` | 新建，从 handle/ 迁移，支持 `set_case_data_dir()` + upstream/call_chain |
| `agents/tools/data_detective_tools.py` | 加 `reload_explorer()` 支持 per-case 重载 |
| `convert_all.py` | 新建，500 case 批量转换脚本 |
| `run_batch.py` | 新建，批量评测脚本（AC@1 计算 + 断点恢复） |
| `agent_runner.py` | 新建，RolloutRunner 标准 stdin/stdout 接口；2026-04-14 加 `--log-file` 支持（TeeStream + FileHandler 双路捕获）|

---

## 性能特征

- 冒烟测试（1 case）：~33 次 LLM 调用，~7 分钟
- ProcessScheduler 嵌套调用子 Agent，每个子 Agent 有独立 ReAct 循环
- MAX_REACT_STEPS=8 限制最大步数
- kimi-k2 API 调用约 1-2 秒/次

---

## 已知限制

1. **服务级 vs Endpoint 级**：原 mABC 设计为 HTTP endpoint 粒度（如 `ts-travel-service-/api/v1/...`），我们降级为 service 级（如 `ts-travel-service`）。对 RCA 影响不大。
2. **指标有限**：只有 calls/error_rate/avg_duration/timeout_rate，没有 CPU/Memory 等资源指标。部分故障类型（如 bandwidth/corrupt）不一定在这些指标上体现。
3. **投票机制未启用**：ThreeHotCotRun 的 alpha/beta 为 -1，投票直接跳过。
4. **eval() 安全风险**：act_eval 使用 eval() 执行 LLM 生成的代码，仅限评测环境使用。
