# 风控策略 Agent

一个由 POST 请求触发的价格风控策略生成服务。浏览器负责聊天与下载，FastAPI 负责认证和请求调度，Agent 负责提示词、Ollama 工具调用和策略生成。服务在局域网运行；网页的生产构建由同一个 Python 进程托管。

## 目录与边界

| 目录 | 职责 |
| --- | --- |
| `frontend/` | React 页面；`src/api.ts` 集中处理 HTTP、SSE、预览和下载请求。API Key 与账号 ID 只在页面内存中。 |
| `backend/` | FastAPI 路由、Bearer 认证、环境配置、请求校验、SSE 映射和适配器装配。 |
| `agent/` | 原文迁移的系统提示词、按轮次添加上海日期、历史消息组织和有上限的工具调用循环。无 FastAPI 依赖。 |
| `interfaces/价格接口/` | 从旧项目迁入的 Python 接口源码。Agent 可列出和读取，运行时不写入。 |
| `artifacts/` | 生成的 Python 源码及迁移前遗留的 JSON 元数据。新产物归属和会话索引写入 SQLite。 |
| `data/` | 默认 SQLite 数据库目录，保存账号、会话、消息和产物索引；可由 `APP_DATA_DIR` 调整。 |

`AgentRunner.run(...)` 是后端调用 Agent 的边界，返回异步业务事件。模型、接口读取和产物存储都经接口注入。共享 Bearer API Key 负责认证，账号 ID 负责数据归属，不是身份凭证；持有共享 Key 的调用方可以查看所有账号的监控数据。聊天时第一次使用的账号 ID（去除首尾空白后）和 UUID 会创建账号及会话，会话绑定账号且不能转移。服务端从 SQLite 恢复对话历史，并把用户消息、Agent 回复和文件关联持久化。网页的监控总览可浏览账号、会话、消息和生成文件。

账号、会话和文件监控的实施路线见[真实账号与会话监控功能计划](docs/monitoring-feature-plan.md)。

## 本地启动

需要 Conda、Node.js 20+、Docker，以及一个支持工具调用的 Ollama 模型。先安装并启动 Ollama、拉取所选模型；模型名称以 `GET /api/models` 实际返回为准。功能检测使用预先下载的 Python 3.12 容器镜像：

```bash
docker pull python:3.12-slim
```

```bash
conda create -n risk-agent python=3.12 -y
conda activate risk-agent
cd /Users/eurus/Code/风控客制平台
python -m pip install -r requirements.txt
cd frontend
npm ci
npm run build
cd ..
cp .env.example .env
```

编辑 `.env`：设置真实的长随机 `AGENT_API_KEY`、已安装的 `OLLAMA_MODEL`，必要时修改 `OLLAMA_BASE_URL`。项目启动时会自动读取项目根目录下的 `.env`；已在 shell 中设置的环境变量优先。默认服务监听 `0.0.0.0:8765`，可通过 `APP_HOST` 和 `APP_PORT` 调整。数据库默认写入 `data/monitoring.sqlite3`；设置 `APP_DATA_DIR` 可换用其他目录。启动服务：

```bash
python -m backend
```

在浏览器打开 `http://<服务所在主机>:8765/`，进入“监控总览”并输入 API Key 查看全部账号；进入“策略对话”后输入账号 ID 开始聊天。前端开发时，在两个终端分别启动后端和 `cd frontend && npm run dev`；Vite 会将 `/api` 代理到本机的 8765 端口。生产部署时，请限制服务仅在可信局域网可达；跨网访问可在前面加 HTTPS 反向代理。

## HTTP 接口

所有 `/api` 接口都要求 `Authorization: Bearer <AGENT_API_KEY>`。聊天请求必须提供 `account_id`、UUID 格式的 `conversation_id` 和本轮 `message`。账号 ID 去首尾空白后区分，大小写敏感；全局唯一的会话 UUID 第一次使用时绑定账号，其他账号复用时返回 404。下载请求必须提供账号 ID，服务端会对照 SQLite 中的产物归属；归属不匹配或记录不存在时统一返回 404。聊天请求不再接收历史，历史以服务端数据库记录为准。

```bash
curl -N http://127.0.0.1:8765/api/agent/chat/stream \
  -H "Authorization: Bearer $AGENT_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"account_id":"demo-account","conversation_id":"550e8400-e29b-41d4-a716-446655440000","message":"SHFE rb2710 超过 3900 时预警","model":"qwen3:8b","options":{"temperature":0.3,"top_p":0.8,"num_ctx":16384},"think":false}'
```

`options.temperature` 范围为 `0–2`，`options.top_p` 范围为 `0–1`，`options.num_ctx` 范围为 `512–131072` 的整数。省略 `options` 或 `think` 时分别使用 `0.3`、`0.8`、`16384` 和 `false`。

`POST /api/agent/chat/stream` 返回 `text/event-stream`：

每条事件的 `data` JSON 都包含本次请求所属的 `conversation_id`，可据此把并发会话的事件对应到各自对话。此字段与事件自身的数据并列返回。

| 事件 | 数据 |
| --- | --- |
| `status` | `phase`（开始或工具执行状态）、可选 `tool`。 |
| `delta` | `text`，回复的流式片段。 |
| `artifact` | `artifact_id`、`filename`，表示文件已保存。 |
| `done` | 完整 `content`、`model`、本轮 `artifacts`、`usage`。 |
| `error` | `code`、`message`；上游不可用或工具循环失败等。 |

`GET /api/models` 返回已安装的 `models` 列表和服务端配置的 `default_model`。网页优先选择已安装的默认模型。

必要参数不足时，Agent 按提示词追问；用户补充后服务端从 SQLite 恢复此前历史。生成文件源码保存到 `artifacts/`，归属及会话索引写入 SQLite；Agent 不覆盖接口源码或已有文件。生成源码必须先通过 Python 3.7 语法检查，再由独立测试用例 Skill 根据对话需求构造行情用例，并在无网络、只读挂载且有限制的 Docker 沙盒中执行。沙盒模拟消息引擎及未实现的 `RHclient`；检测结果不代表真实行情引擎验证。

Skill 文件位于 `agent/skills/<skill_id>/SKILL.md`。`SkillLoader` 是可复用的异步接口，`load(skill_id)` 返回包含 `skill_id` 与 `instructions` 的 `Skill`；默认 `FilesystemSkillLoader` 会校验 ID、路径、UTF-8 编码和大小。添加 Skill 时只需新增符合 ID 规则的目录及 `SKILL.md`。加载失败会让功能检测失败，不会回退到其他提示词；测试或其他存储来源可以通过实现 `SkillLoader` 注入。

监控读取接口沿用相同的 Bearer Key：

| 方法与路径 | 作用 |
| --- | --- |
| `GET /api/monitor/accounts?offset=0&limit=50` | 分页返回账号汇总。 |
| `GET /api/monitor/accounts/{account_id}/conversations?offset=0&limit=50` | 返回会话摘要及未关联会话的旧文件列表。 |
| `GET /api/monitor/accounts/{account_id}/conversations/{conversation_id}` | 按序分页返回消息，并分页返回会话文件详情。 |

列表按更新时间和稳定 ID 排序；会话详情可用 `message_offset`、`message_limit`、`artifact_offset` 和 `artifact_limit` 分页。服务启动时会把旧的 `artifacts/<artifact_id>.json` 元数据导入 SQLite；没有会话 ID 的文件留在账号的“未关联会话文件”区域，原始文件不会被迁移删除。

```bash
curl http://127.0.0.1:8765/api/models \
  -H "Authorization: Bearer $AGENT_API_KEY"

curl -X POST http://127.0.0.1:8765/api/artifacts/download \
  -H "Authorization: Bearer $AGENT_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"artifact_id":"<artifact 事件里的 32 位 ID>","account_id":"demo-account"}' \
  -o strategy.py
```

下载接口按 `artifact_id` 查找文件，并使用请求中的 `account_id` 核对 SQLite 产物归属。成功响应以 `application/octet-stream` 分块发送原始 UTF-8 文件字节，并通过 `Content-Disposition` 提供 `.py` 文件名；浏览器客户端可读取响应字节并生成 `.py` 文件。生成文件以唯一的 `RHTemplate` 策略主类名命名，例如 `RiskAlertStrategy` 类保存为 `RiskAlertStrategy.py`。文件名只允许直接的 `.py` 文件名；接口读取和产物下载都拒绝路径穿越及符号链接。人工清理时删除 `artifacts/` 下对应的 `.py` 文件；迁移前旧产物若有 `.json` sidecar，也可一并清理。

Python 客户端示例见 [`examples/download_artifact_client.py`](examples/download_artifact_client.py)，它会分块接收二进制响应并增量解码为 UTF-8 源码文本。运行前，在文件末尾的 `if __name__ == "__main__"` 代码块中填写 API Key、账号 ID 和 artifact ID，并按需修改服务地址与输出文件名。

```bash
.venv/bin/python3.12 examples/download_artifact_client.py
```

## 验收

```bash
conda activate risk-agent
python -m pip install 'pytest>=8,<10' 'httpx>=0.27,<1'
python -m pytest -q
cd frontend
npm run test
npm run build
```

测试使用模拟模型和临时文件目录，覆盖工具流程、缺参追问、多轮修改、调用上限、并发隔离、认证、SSE、下载、非法路径、Ollama 故障和前端交互。端到端生成质量还取决于实际选用模型的工具调用能力及真实 Ollama 环境。
