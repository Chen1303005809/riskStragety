# 风控策略 Agent

一个由 POST 请求触发的价格风控策略生成服务。浏览器负责聊天与下载，FastAPI 负责认证和请求调度，Agent 负责提示词、Ollama 工具调用和策略生成。服务在局域网运行；网页的生产构建由同一个 Python 进程托管。

## 目录与边界

| 目录 | 职责 |
| --- | --- |
| `frontend/` | React 聊天页面；`src/api.ts` 集中处理 HTTP、SSE、预览和下载请求。API Key 只在页面内存中。 |
| `backend/` | FastAPI 路由、Bearer 认证、环境配置、请求校验、SSE 映射和适配器装配。 |
| `agent/` | 原文迁移的系统提示词、按轮次添加上海日期、历史消息组织和有上限的工具调用循环。无 FastAPI 依赖。 |
| `interfaces/价格接口/` | 从旧项目迁入的 Python 接口源码。Agent 可列出和读取，运行时不写入。 |
| `artifacts/` | 生成的 Python 源码及其账户归属元数据。服务自动创建，持续保存至人工清理。 |

`AgentRunner.run(...)` 是后端调用 Agent 的边界，返回异步业务事件。模型、接口读取和产物存储都经接口注入。当前认证将共享 Key 映射为 `default` 账户，产物元数据已经记录 `account_id`；后续可替换认证和存储适配器来支持多账户及共享存储。每次聊天请求独立构造历史和工具状态，服务端不保存会话。

## 本地启动

需要 Conda、Node.js 20+，以及一个支持工具调用的 Ollama 模型。先安装并启动 Ollama、拉取所选模型；模型名称以 `GET /api/models` 实际返回为准。

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

编辑 `.env`：设置真实的长随机 `AGENT_API_KEY`、已安装的 `OLLAMA_MODEL`，必要时修改 `OLLAMA_BASE_URL`。默认服务监听 `0.0.0.0:8765`，可通过 `APP_HOST` 和 `APP_PORT` 调整。服务不会自动读取 `.env`，启动时把它导入当前 shell：

```bash
set -a
source .env
set +a
python -m backend
```

在浏览器打开 `http://<服务所在主机>:8765/`，输入同一个 API Key。前端开发时，在两个终端分别启动后端和 `cd frontend && npm run dev`；Vite 会将 `/api` 代理到本机的 8765 端口。生产部署时，请限制服务仅在可信局域网可达；跨网访问可在前面加 HTTPS 反向代理。

## HTTP 接口

所有 `/api` 接口都要求 `Authorization: Bearer <AGENT_API_KEY>`。每次请求携带此前完整的 user/assistant 历史，不接受系统消息。历史只存在客户端页面内存中，刷新后需要重新开始。

```bash
curl -N http://127.0.0.1:8765/api/agent/chat/stream \
  -H "Authorization: Bearer $AGENT_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"message":"SHFE rb2710 超过 3900 时预警","history":[],"model":"qwen3:8b","options":{"temperature":0.3,"top_p":0.8,"num_ctx":16384},"think":false}'
```

`options.temperature` 范围为 `0–2`，`options.top_p` 范围为 `0–1`，`options.num_ctx` 范围为 `512–131072` 的整数。省略 `options` 或 `think` 时分别使用 `0.3`、`0.8`、`16384` 和 `false`。

`POST /api/agent/chat/stream` 返回 `text/event-stream`：

| 事件 | 数据 |
| --- | --- |
| `status` | `phase`（开始或工具执行状态）、可选 `tool`。 |
| `delta` | `text`，回复的流式片段。 |
| `artifact` | `artifact_id`、`filename`，表示文件已保存。 |
| `done` | 完整 `content`、`model`、本轮 `artifacts`、`usage`。 |
| `error` | `code`、`message`；上游不可用或工具循环失败等。 |

`GET /api/models` 返回已安装的 `models` 列表和服务端配置的 `default_model`。网页优先选择已安装的默认模型。

必要参数不足时，Agent 按旧提示词追问；用户补充后将此前历史继续传入。生成文件只保存到 `artifacts/`，不覆盖接口源码或已有文件。Agent 对生成源码做 Python 3.7 语法检查，以符合迁入接口的策略规范；它不会执行生成代码。

```bash
curl http://127.0.0.1:8765/api/models \
  -H "Authorization: Bearer $AGENT_API_KEY"

curl -X POST http://127.0.0.1:8765/api/artifacts/download \
  -H "Authorization: Bearer $AGENT_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"artifact_id":"<artifact 事件里的 32 位 ID>"}' \
  -o strategy.py
```

下载接口按 `artifact_id` 查找文件，并核对当前账户身份。成功响应以 `application/octet-stream` 分块发送原始 UTF-8 文件字节，并通过 `Content-Disposition` 提供 `.py` 文件名；浏览器客户端可读取响应字节并生成 `.py` 文件。文件名只允许直接的 `.py` 文件名；接口读取和产物下载都拒绝路径穿越及符号链接。人工清理时可按 ID 删除 `artifacts/` 下对应的 `.py` 与 `.json` 文件。

Python 客户端示例见 [`examples/download_artifact_client.py`](examples/download_artifact_client.py)，它会分块接收二进制响应并增量解码为 UTF-8 源码文本。运行前，在文件末尾的 `if __name__ == "__main__"` 代码块中填写 API Key 和 artifact ID，并按需修改服务地址与输出文件名。

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
