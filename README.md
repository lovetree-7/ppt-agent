# pptAgennt - 多智能体 PPT 自动生成流水线

基于 **LangGraph** 编排 + **DeepSeek** 大模型的 Human-in-the-Loop 多 Agent PPT 自动生成系统。只需输入自然语言需求，即可端到端生成专业级 `.pptx` 演示文稿。

## 架构概览

```
用户输入需求
    │
    ▼
┌─────────────────────────────────────────────┐
│  Agent 1: Planner (大纲规划专家)             │
│  deepseek-v4-flash · temperature=0.7        │
│  拆解需求 → 结构化 JSON 大纲                │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────┐
│  Review 1: 人工审核  │  ← LangGraph interrupt 断点
│  approve / reject    │     reject → 回退 Planner 重生成
└─────────────────┘
    │ (approve)
    ▼
┌─────────────────────────────────────────────┐
│  Agent 2: Writer (文案视觉化专家)            │
│  deepseek-v4-flash · temperature=0.8        │
│  大纲 → 视觉化短文案 JSON                    │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────┐
│  Review 2: 人工审核  │  ← LangGraph interrupt 断点
│  approve / reject    │     reject → 回退 Writer 重生成
└─────────────────┘
    │ (approve)
    ▼
┌─────────────────────────────────────────────┐
│  Agent 3: Skill Adapter (格式翻译专家)       │
│  deepseek-v4-flash · temperature=0.3        │
│  翻译文案 → spec_lock.md + design_spec.md   │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Agent 4: SVG Executor (SVG 执行专家)       │
│  deepseek-v4-pro[1m] · temperature=0.2      │
│  逐页生成 SVG → 导出 .pptx                   │
└─────────────────────────────────────────────┘
    │
    ▼
  输出 .pptx 文件
```

## 技术栈

| 层级 | 技术 |
|------|------|
| 编排框架 | LangGraph (StateGraph + MemorySaver + interrupt) |
| LLM 客户端 | OpenAI SDK (兼容模式) |
| 快速模型 | DeepSeek V4 Flash (Agent 1-3) |
| 强力模型 | DeepSeek V4 Pro [1M context] (Agent 4) |
| PPT 引擎 | ppt-master skill (SVG → PPTX) |
| 语言 | Python 3 |

## 环境要求

- Python 3.10+
- DeepSeek API Key ([获取地址](https://platform.deepseek.com))
- ppt-master skill（SVG 渲染引擎，位于 `.workbuddy/skills/ppt-master/`）

## 快速开始

### 1. 安装依赖

```bash
pip install langgraph openai
```

### 2. 配置 API Key

```bash
# 方式一：环境变量
export DEEPSEEK_API_KEY="sk-xxxx"

# 方式二：.env 文件
cp .env.example .env
# 编辑 .env，填入你的 API Key
```

### 3. 配置 ppt-master 路径

编辑 `multi_agent_ppt_pipeline.py` 中的 `PPT_MASTER_SKILL_DIR` 变量，指向你的 ppt-master skill 目录。

### 4. 运行

```bash
python multi_agent_ppt_pipeline.py
```

启动后按提示输入 PPT 需求，在审核断点处输入 `approve`/`reject` 控制流水线。

## 支持的布局

| 布局名称 | 说明 |
|----------|------|
| single-column-centered | 单列居中布局 |
| symmetric-split | 对称分栏布局 |
| asymmetric-split | 非对称分栏布局 |
| top-bottom-split | 上下拆分布局 |
| three-column-cards | 三列卡片布局 |
| four-column-cards | 四列卡片布局 |
| matrix-grid | 矩阵网格布局 |
| z-pattern | Z 型视觉流布局 |
| center-radiating | 中心发散布局 |
| full-bleed | 全幅出血布局 |
| negative-space | 负空间布局 |

## 设计推断

当用户未指定设计偏好时，Agent 3 会根据主题自动推断配色方案：

| 主题领域 | 主色 | 强调色 |
|----------|------|--------|
| 科技/互联网 | 深蓝 | 青色 |
| 商务/金融 | 深灰 | 金色 |
| 教育/培训 | 紫色 | 珊瑚色 |
| 医疗/健康 | 青绿 | 蓝色 |

## 项目结构

```
pptAgennt/
├── multi_agent_ppt_pipeline.py   # 主入口，LangGraph 流水线
├── .env.example                  # 环境变量模板
├── .gitignore
└── README.md
```

## License

MIT
