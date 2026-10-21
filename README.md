# 风控策略 Agent

一个由 POST 请求触发的价格风控策略生成服务。浏览器负责聊天与下载，FastAPI 负责认证和请求调度，Agent 通过 Ollama 或 DeepSeek 模型服务进行工具调用和策略生成。服务在局域网运行；网页的生产构建由同一个 Python 进程托管。

## 目录与边界

| 目录 | 职责 |
| --- | --- |
| `frontend/` | React 页面；`src/api.ts` 集中处理 HTTP、SSE、预览和下载请求。服务 Bearer Key 与账号 ID 只在页面内存中。 |
| `backend/` | FastAPI 路由、Bearer 认证、在线模型服务配置、请求校验、SSE 映射和适配器装配。 |
| `agent/` | 原文迁移的系统提示词、按轮次添加上海日期、历史消息组织和有上限的工具调用循环。无 FastAPI 依赖。 |
| `interfaces/价格接口/` | 从旧项目迁入的 Python 接口源码。Agent 可列出和读取，运行时不写入。 |
| `artifacts/` | 生成的 Python 源码及迁移前遗留的 JSON 元数据。新产物归属和会话索引写入 SQLite。 |
| `data/` | 默认数据目录，保存账号、会话、消息、产物索引、DeepSeek Key 配置和会话交互日志；可由 `APP_DATA_DIR` 调整。 |

`AgentRunner.run(...)` 是后端调用 Agent 的边界，返回异步业务事件。模型、接口读取和产物存储都经接口注入。共享 Bearer API Key 负责认证，账号 ID 负责数据归属，不是身份凭证；持有共享 Key 的调用方可以查看所有账号的监控数据。聊天时第一次使用的账号 ID（去除首尾空白后）和 UUID 会创建账号及会话，会话绑定账号且不能转移。服务端从 SQLite 恢复对话历史，并把用户消息、Agent 回复和文件关联持久化。网页的监控总览可浏览账号、会话、消息和生成文件。

每个会话还会在 `APP_DATA_DIR/conversation_logs/<conversation_id>.jsonl`（默认 `data/conversation_logs/`）追加完整交互日志。日志包含每轮开始时的历史和用户消息、发送给主模型及功能检测模型的 system/user/assistant/tool 上下文、模型回复片段、工具调用参数与结果、沙盒用例和结果，以及完成、失败或中断状态。每条记录独立为一行 JSON；同一 `conversation_id` 续聊时追加到原文件。日志目录权限为 `0700`，文件权限为 `0600`。

账号、会话和文件监控的实施路线见[真实账号与会话监控功能计划](docs/monitoring-feature-plan.md)。

## 本地启动

需要 Python 3.12、Node.js 20+ 和 Docker。使用 Ollama 时，安装并启动 Ollama、拉取支持工具调用的模型；使用 DeepSeek 时，可在网页中配置 API Key，不需要本地启动 Ollama。功能检测使用预先下载的 Python 3.12 容器镜像：

```bash
docker pull python:3.12-slim
```

```bash
cd /Users/eurus/Code/风控客制平台
python3.12 -m venv .venv
.venv/bin/python3.12 -m pip install -r requirements.txt
cd frontend
npm ci
npm run build
cd ..
cp .env.example .env
```

编辑 `.env`：设置真实的长随机 `AGENT_API_KEY`；选择 Ollama 时设置已安装的 `OLLAMA_MODEL`，必要时修改 `OLLAMA_BASE_URL`。项目启动时会自动读取项目根目录下的 `.env`；已在 shell 中设置的环境变量优先。默认服务监听 `0.0.0.0:8765`，可通过 `APP_HOST` 和 `APP_PORT` 调整。数据库默认写入 `data/`；设置 `APP_DATA_DIR` 可换用其他目录。启动服务：

```bash
.venv/bin/python3.12 -m backend
```

在浏览器打开 `http://<服务所在主机>:8765/`，进入“监控总览”并输入服务 API Key 查看全部账号；进入“策略对话”后输入账号 ID，并选择 Ollama 或 DeepSeek。首次使用 DeepSeek 时，可在连接表单填写 DeepSeek API Key；连接后也可在“模型参数”面板保存、替换或清除 Key。服务端会验证 Key 并将其保存在 `APP_DATA_DIR/provider_settings.sqlite3`（默认 `data/provider_settings.sqlite3`），数据库文件权限限制为仅服务进程用户可读写，Key 不会返回到网页。前端开发时，在两个终端分别启动后端和 `cd frontend && npm run dev`；Vite 会将 `/api` 代理到本机的 8765 端口。生产部署时，请限制服务仅在可信局域网可达；跨网访问可在前面加 HTTPS 反向代理。

## HTTP 接口

所有 `/api` 接口都要求 `Authorization: Bearer <AGENT_API_KEY>`。聊天请求必须提供 `account_id`、UUID 格式的 `conversation_id` 和本轮 `message`；`provider` 可填 `ollama` 或 `deepseek`，省略时使用 `ollama`。账号 ID 去首尾空白后区分，大小写敏感；全局唯一的会话 UUID 第一次使用时绑定账号，其他账号复用时返回 404。下载请求必须提供账号 ID，服务端会对照 SQLite 中的产物归属；归属不匹配或记录不存在时统一返回 404。聊天请求不再接收历史，历史以服务端数据库记录为准。

```bash
curl -N http://127.0.0.1:8765/api/agent/chat/stream \
  -H "Authorization: Bearer $AGENT_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"account_id":"demo-account","conversation_id":"550e8400-e29b-41d4-a716-446655440000","message":"SHFE rb2710 超过 3900 时预警","model":"qwen3:8b","options":{"temperature":0.3,"top_p":0.8,"num_ctx":16384},"think":false}'
```

`options.temperature` 范围为 `0–2`，`options.top_p` 范围为 `0–1`，`options.num_ctx` 范围为 `512–262144` 的整数。省略 `options` 或 `think` 时分别使用 `0.3`、`0.8`、`16384` 和 `false`。选择 Ollama 的 Qwen3.8 模型时，服务端会忽略请求传入的 `num_ctx` 并固定为 `262144`；网页也会显示该固定值。

`POST /api/agent/chat/stream` 返回 `text/event-stream`：

每条事件的 `data` JSON 都包含本次请求所属的 `conversation_id`，可据此把并发会话的事件对应到各自对话。此字段与事件自身的数据并列返回。

| 事件 | 数据 |
| --- | --- |
| `status` | `phase`（开始或工具执行状态）、可选 `tool`。 |
| `delta` | `text`，回复的流式片段。 |
| `artifact` | `artifact_id`、`filename`，表示文件已保存。 |
| `done` | 完整 `content`、`model`、本轮 `artifacts`、`usage`、`ErrCode`、`ErrMsg`。有产物时 `ErrCode` 为 `ok`；无产物时仍正常结束，但返回 `file_not_generated`。 |
| `error` | `ErrCode`、`ErrMsg`；模型、Agent 或工具运行失败，以及 Agent 流提前结束。 |

只有最终的 `done` 或 `error` 事件带 `ErrCode`、`ErrMsg`。`done` 的两个字段与 `content` 同级；HTTP 错误响应也在 JSON 顶层返回这两个字段，并保留 `detail`，其内容与 `ErrMsg` 一致。错误消息为固定客户端文案，不包含上游异常原文。

| ErrCode | 场景 | HTTP 状态 / SSE 事件 | ErrMsg |
| --- | --- | --- | --- |
| `ok` | SSE 正常完成且至少生成一个文件 | `done` | 空字符串 |
| `file_not_generated` | SSE 正常完成，但 `artifacts` 为空 | `done` | 本轮已完成，但未生成策略文件。 |
| `auth_invalid` | Bearer API Key 无效 | HTTP 401 | API Key 无效 |
| `request_invalid` | 请求体、参数或 UUID 格式无效 | HTTP 422 | 请求参数无效 |
| `provider_not_configured` | 使用 DeepSeek 前没有配置 API Key | HTTP 409 | 请先配置 DeepSeek API Key |
| `provider_key_invalid` | DeepSeek API Key 校验失败或服务暂不可达 | HTTP 400 | DeepSeek API Key 验证失败，请检查 Key 或网络连接 |
| `conversation_not_found` | 会话不存在或不属于该账号 | HTTP 404 | 会话不存在 |
| `account_not_found` | 账号不存在 | HTTP 404 | 账号不存在 |
| `artifact_not_found` | 产物不存在或不属于该账号 | HTTP 404 | 产物不存在 |
| `model_service_unavailable` | 模型列表上游不可用 | HTTP 502 | 模型服务暂不可用，请稍后重试 |
| `agent_incomplete` | Agent 流结束但未发出 `done` | SSE `error` | 本轮生成未正常结束，请重试 |
| `agent_failed` | Agent、模型或工具运行中抛出异常 | SSE `error` | 本轮处理失败，请稍后重试 |
| `resource_not_found` | API 路由或其他资源不存在 | HTTP 404 | 请求的资源不存在 |
| `method_not_allowed` | API 请求方法不支持 | HTTP 405 | 请求方法不受支持 |
| `internal_error` | 未分类的 API 服务端异常 | HTTP 500 | 服务内部错误，请稍后重试 |

`GET /api/models` 返回 Ollama 的模型列表；`GET /api/models?provider=deepseek` 返回 DeepSeek API 当前可用模型。响应还会给出 `deepseek_configured` 状态和 `default_model`。`GET /api/providers` 返回 provider 配置状态。`PUT /api/providers/deepseek` 使用 `{"api_key":"..."}` 验证并保存 DeepSeek Key，`DELETE /api/providers/deepseek` 清除 Key；这两个配置接口同样要求服务 Bearer Key。网页优先选择 provider 返回的默认模型。

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

下载接口按 `artifact_id` 查找文件，并使用请求中的 `account_id` 核对 SQLite 产物归属。成功响应以 `application/octet-stream` 分块发送原始 UTF-8 文件字节，并通过 `Content-Disposition` 提供 `.py` 文件名；浏览器客户端可读取响应字节并生成 `.py` 文件。策略主类标识符固定为 `My_PythonObj`，其 `name` 属性只能使用英文 ASCII 字母、数字、下划线和连字符，并以英文字母开头；文件名由该 `name` 值决定，例如 `name = "RiskAlertStrategy"` 会保存为 `RiskAlertStrategy.py`。文件名只允许直接的 `.py` 文件名；接口读取和产物下载都拒绝路径穿越及符号链接。人工清理时删除 `artifacts/` 下对应的 `.py` 文件；迁移前旧产物若有 `.json` sidecar，也可一并清理。

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
