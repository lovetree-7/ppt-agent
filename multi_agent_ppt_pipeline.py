"""
multi_agent_ppt_pipeline.py
============================
多 Agent 编排的 PPT 生成流水线。

架构：
  Agent 1 (Planner)       → 将原始需求拆解为结构化大纲
  Review 1                → 人工审核大纲 (approve / reject + 修改意见)
  Agent 2 (Writer)        → 将大纲扩展为视觉化短文案
  Review 2                → 人工审核文案 (approve / reject + 修改意见)
  Agent 3 (Skill Adapter) → 翻译文案为 ppt-master 的 spec_lock.md + design_spec.md
  Agent 4 (SVG Executor)  → 逐页生成 SVG，导出 PPTX

编排框架：LangGraph (支持 interrupt 审核断点 + 条件回退边)
LLM 客户端：OpenAI SDK (兼容 DeepSeek API)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt
from openai import OpenAI


# ============================================================================
# 第一部分：配置 (全部写死，仅 API Key 从环境变量读取)
# ============================================================================

# --- 模型配置 ---
# Agent 1-3 使用 deepseek-v4-flash (快速、低成本)
# Agent 4 使用 deepseek-v4-pro[1m] (最强模型，负责 SVG 代码生成)
MODEL_FAST = "deepseek-v4-flash"
MODEL_PRO = "deepseek-v4-pro[1m]"

MODEL_CONFIG = {
    "planner":  {"model": MODEL_FAST, "temperature": 0.7},
    "writer":   {"model": MODEL_FAST, "temperature": 0.8},
    "adapter":  {"model": MODEL_FAST, "temperature": 0.3},
    "executor": {"model": MODEL_PRO,  "temperature": 0.2},
}

# --- API 配置 ---
API_BASE_URL = "https://api.deepseek.com"
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if not API_KEY:
    # 尝试从 .env 文件加载 (用于本地开发)
    try:
        with open(Path(__file__).parent / ".env", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("DEEPSEEK_API_KEY="):
                    API_KEY = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    except FileNotFoundError:
        pass
if not API_KEY:
    raise RuntimeError(
        "请设置 DEEPSEEK_API_KEY 环境变量，或在项目根目录创建 .env 文件并写入:\n"
        "DEEPSEEK_API_KEY=your_api_key_here"
    )

# --- ppt-master skill 路径 ---
PPT_MASTER_SKILL_DIR = Path(
    r"C:\ProgramData\WorkBuddy\chromium-env\f0eyd0\.workbuddy\skills\ppt-master"
)

# --- 项目输出目录 ---
OUTPUT_DIR = Path(r"D:\ppt-agent\output")

# --- ppt-master 关键文件路径 (从 skill 目录派生) ---
SPEC_LOCK_REFERENCE = PPT_MASTER_SKILL_DIR / "templates" / "spec_lock_reference.md"
DESIGN_SPEC_REFERENCE = PPT_MASTER_SKILL_DIR / "templates" / "design_spec_reference.md"
EXECUTOR_BASE = PPT_MASTER_SKILL_DIR / "references" / "executor-base.md"
SHARED_STANDARDS = PPT_MASTER_SKILL_DIR / "references" / "shared-standards.md"
PROJECT_MANAGER_SCRIPT = PPT_MASTER_SKILL_DIR / "scripts" / "project_manager.py"
FINALIZE_SVG_SCRIPT = PPT_MASTER_SKILL_DIR / "scripts" / "finalize_svg.py"
SVG_TO_PPTX_SCRIPT = PPT_MASTER_SKILL_DIR / "scripts" / "svg_to_pptx.py"

# --- LLM 调用参数 ---
MAX_RETRIES = 3       # JSON 解析失败时的重试次数
LLM_TIMEOUT = 300     # 单次 LLM 调用超时 (秒)，Agent 4 的 SVG 生成可能较慢


# ============================================================================
# 第二部分：状态定义 (LangGraph State)
# ============================================================================

class PipelineState(TypedDict, total=False):
    """LangGraph 共享状态机。

    字段说明：
      user_prompt        : 用户原始需求 (必填)
      design_prefs       : 用户可选的设计偏好 (画布/颜色/字体/图标/风格)，不填则 Agent 3 推断
      outline            : Agent 1 输出的结构化大纲 (JSON)
      outline_feedback   : Review 1 的 reject 修改意见 (回退时 Planner 读取)
      outline_decision   : Review 1 的决策 ("approve" / "reject")
      page_contents      : Agent 2 输出的视觉化文案 (JSON)
      writer_feedback    : Review 2 的 reject 修改意见 (回退时 Writer 读取)
      writer_decision    : Review 2 的决策 ("approve" / "reject")
      spec_lock_content  : Agent 3 生成的 spec_lock.md 文件内容
      design_spec_content: Agent 3 生成的 design_spec.md 文件内容
      project_path       : ppt-master 项目目录路径
      svg_pages          : Agent 4 生成的 SVG 页面列表 [{"page_num": 1, "svg": "<svg>..."}, ...]
      pptx_path          : 最终导出的 PPTX 文件路径
      error              : 错误信息 (如果有)
    """
    user_prompt: str
    design_prefs: str
    outline: dict
    outline_feedback: str
    outline_decision: str
    page_contents: dict
    writer_feedback: str
    writer_decision: str
    spec_lock_content: str
    design_spec_content: str
    project_path: str
    svg_pages: list
    pptx_path: str
    error: str


# ============================================================================
# 第三部分：LLM 客户端工厂
# ============================================================================

def get_llm_client() -> OpenAI:
    """创建 OpenAI 兼容客户端 (连接 DeepSeek API)。"""
    if not API_KEY:
        print("[ERROR] API_KEY 未配置。请编辑代码第 52 行填入你的 DeepSeek API Key。")
        sys.exit(1)
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def call_llm(
    client: OpenAI,
    agent_key: str,
    system_prompt: str,
    user_message: str,
    json_output: bool = True,
) -> str:
    """调用 LLM 并返回文本响应。

    参数：
      agent_key    : MODEL_CONFIG 中的键 ("planner" / "writer" / "adapter" / "executor")
      system_prompt: 系统提示词
      user_message : 用户消息
      json_output  : 是否强制 JSON 输出 (Agent 4 生成 SVG 时设为 False)

    返回：
      LLM 的文本响应
    """
    config = MODEL_CONFIG[agent_key]
    kwargs = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": config["temperature"],
        "timeout": LLM_TIMEOUT,
    }
    if json_output:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def call_llm_json(
    client: OpenAI,
    agent_key: str,
    system_prompt: str,
    user_message: str,
) -> dict:
    """调用 LLM 并解析为 JSON dict。带重试机制。"""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = call_llm(client, agent_key, system_prompt, user_message, json_output=True)
            return json.loads(raw)
        except (json.JSONDecodeError, Exception) as e:
            last_error = e
            print(f"  [WARN] JSON 解析失败 (第 {attempt}/{MAX_RETRIES} 次): {e}")
            if attempt < MAX_RETRIES:
                # 在重试时追加提示
                user_message = user_message + "\n\n[注意] 上次输出不是合法 JSON，请严格输出 JSON 格式。"
    raise RuntimeError(f"LLM JSON 解析失败 {MAX_RETRIES} 次: {last_error}")


# ============================================================================
# 第四部分：Agent System Prompts
# ============================================================================

# --- Agent 1: Planner ---
PLANNER_SYSTEM_PROMPT = """\
你是一个顶级的 PPT 大纲规划专家。你的任务是将用户的原始需求拆解为一份结构严密、逻辑连贯的 PPT 大纲。

## 你的职责

1. 分析用户需求，确定 PPT 的主题、受众和演示目的
2. 规划总页数（建议 8-15 页，封面 + 正文 + 结尾）
3. 为每页设定标题和核心主旨（一句话概括该页要传达的核心信息）
4. 确保大纲具有清晰的叙事逻辑：引入 → 展开 → 论证 → 总结

## 大纲结构规则

- 第一页必须是封面 (role: "cover")
- 最后一页视情况为结尾页 (role: "closing")，如果没有明确的总结/CTA 需求则不加
- 中间页面按内容逻辑分组，可以有 "chapter" (章节过渡页) 和 "content" (内容页)
- 每页的 core_message 必须是一句完整的断言句，不能是标题的重复

## 用户反馈处理

如果提供了"用户修改意见"，你必须：
1. 认真分析用户的不满点
2. 在新大纲中针对性地调整
3. 不要完全推翻原大纲，而是在原基础上改进

## 输出格式 (严格 JSON)

```json
{
  "title": "PPT 整体标题",
  "total_pages": 10,
  "narrative_summary": "一句话概括整个 PPT 的叙事主线",
  "pages": [
    {
      "page_num": 1,
      "role": "cover",
      "title": "封面标题",
      "core_message": "一句话说明这个 PPT 要讲什么",
      "layout_hint": "single-column-centered"
    },
    {
      "page_num": 2,
      "role": "chapter",
      "title": "章节过渡标题",
      "core_message": "本章节要讨论的核心问题",
      "layout_hint": "single-column-centered"
    },
    {
      "page_num": 3,
      "role": "content",
      "title": "内容页标题",
      "core_message": "本页的核心断言句",
      "layout_hint": "three-column-cards"
    }
  ]
}
```

## layout_hint 可选值

- "single-column-centered" : 封面/结尾/单点强调
- "symmetric-split" : 左右对比 (5:5)
- "asymmetric-split" : 主次分栏 (3:7 或 2:8)
- "top-bottom-split" : 上下分栏 (流程/时间线)
- "three-column-cards" : 三列并列卡片
- "four-column-cards" : 四列并列卡片
- "matrix-grid" : 2x2 矩阵
- "z-pattern" : Z 型叙事
- "center-radiating" : 中心辐射
- "full-bleed" : 全图 + 浮动文字
- "negative-space" : 大留白单元素

## 注意

- 只输出 JSON，不要输出任何其他文字
- core_message 必须是完整的中文句子
- layout_hint 根据内容性质选择最合适的布局
"""


# --- Agent 2: Writer ---
WRITER_SYSTEM_PROMPT = """\
你是一个顶级的 PPT 文案视觉化专家。你的任务是将 PPT 大纲扩展为适合演示的精简视觉文案。

## 核心原则

1. **字数精简**：每页正文不超过 80 字，用短句和要点
2. **多用要点 (Bullet Points)**：每页 3-5 个要点，每个要点不超过 20 字
3. **规划视觉排版**：根据 layout_hint 设计具体的内容结构
4. **大数字凸显**：有关键数据时，标注为 hero_number 供排版突出
5. **左右对比/三列并列**：对比类内容要明确标注左右/三列的内容分配

## 文案结构规则

- 封面页：主标题 + 副标题 + 作者/日期信息
- 章节页：章节标题 + 一句过渡语
- 内容页：标题 + 核心断言 + 3-5 个要点 + 可选的视觉元素描述
- 结尾页：总结要点 / CTA / 联系方式

## 用户反馈处理

如果提供了"用户修改意见"，你必须：
1. 针对用户意见调整文案
2. 保持大纲结构不变（大纲已通过审核），只调整文案表达

## 输出格式 (严格 JSON)

```json
{
  "pages": [
    {
      "page_num": 1,
      "role": "cover",
      "title": "主标题",
      "subtitle": "副标题",
      "info": "作者 / 日期 / 机构",
      "layout": "single-column-centered",
      "content_blocks": [
        {"type": "title", "text": "主标题"},
        {"type": "subtitle", "text": "副标题"},
        {"type": "info", "text": "作者 / 日期"}
      ]
    },
    {
      "page_num": 2,
      "role": "content",
      "title": "页面标题",
      "core_message": "本页核心断言句",
      "layout": "three-column-cards",
      "content_blocks": [
        {"type": "heading", "text": "页面标题"},
        {"type": "lead", "text": "核心断言句"},
        {"type": "card", "text": "要点1标题: 要点1描述", "icon_hint": "target"},
        {"type": "card", "text": "要点2标题: 要点2描述", "icon_hint": "bolt"},
        {"type": "card", "text": "要点3标题: 要点3描述", "icon_hint": "shield"}
      ],
      "hero_number": null,
      "visual_elements": ["三列卡片网格", "每个卡片带图标"]
    }
  ]
}
```

## content_blocks 的 type 可选值

- "title"     : 主标题
- "subtitle"  : 副标题
- "info"      : 作者/日期/机构信息
- "heading"   : 页面标题
- "lead"      : 核心断言句 (引言/导语)
- "subheading": 小节标题
- "bullet"    : 要点条目
- "card"      : 卡片内容 (含标题和描述)
- "quote"     : 引用语
- "data"      : 数据/数字
- "footer"    : 页脚/来源标注

## 注意

- 只输出 JSON，不要输出任何其他文字
- 所有文案使用中文
- 每页的 content_blocks 数量不超过 8 个
- icon_hint 使用英文图标名 (如 target, bolt, shield, users, chart-bar, lightbulb)
"""


# --- Agent 3: Skill Adapter ---
ADAPTER_SYSTEM_PROMPT = """\
你是 ppt-master 的格式翻译专家。你的任务是将视觉化文案翻译为 ppt-master 能识别的两个核心文件：spec_lock.md 和 design_spec.md 的 §IX Content Outline 部分。

## 你的工作流程

1. 阅读下方提供的 "ppt-master spec_lock 模板参考" 和 "design_spec 模板参考"
2. 阅读下方提供的 "视觉化文案 (page_contents)"
3. 阅读下方提供的 "用户设计偏好 (design_prefs)"，如果为空则你自行推断
4. 生成符合 ppt-master 格式规范的 spec_lock.md 完整内容
5. 生成 design_spec.md 的 §IX Content Outline 部分

## spec_lock.md 生成规则

spec_lock.md 是机器可读的执行契约，格式为 Markdown 的 ## 分节 + - 列表项。必须包含以下分节：

- ## canvas    : 画布配置 (viewBox, format)
- ## mode      : 叙事模式 (pyramid / narrative / instructional / showcase / briefing)
- ## visual_style : 视觉风格预设名 (swiss-minimal / corporate-clean / editorial-elegant / tech-bold 等)
- ## colors    : 颜色方案 (bg, primary, accent, secondary_accent, text, text_secondary, border)
- ## typography: 字体方案 (font_family, title_family, body_family, body, title, subtitle 等)
- ## icons     : 图标库 (library, inventory)
- ## page_rhythm : 每页节奏 (anchor / dense / breathing)
- ## page_layouts : 每页布局模板 (如有)
- ## forbidden  : 禁止项

## design_spec §IX 生成规则

§IX Content Outline 是每页的内容详细描述，格式为：

### Part N: [章节名]

#### Slide NN - [页面名]

- **Layout**: [布局描述]
- **Title**: [页面标题]
- **Core message**: [核心断言]
- **Content**:
  - [内容块1]
  - [内容块2]

## 设计推断规则 (当用户未指定时)

1. **画布格式**：默认 PPT 16:9 (viewBox: 0 0 1280 720)
2. **颜色方案**：根据 PPT 主题推断
   - 科技/互联网 → 深蓝主色 (#185FA5) + 青色强调 (#1D9E75)
   - 商务/金融 → 深灰主色 (#444441) + 金色强调 (#BA7517)
   - 教育/培训 → 紫色主色 (#534AB7) + 珊瑚强调 (#D85A30)
   - 医疗/健康 → 青绿主色 (#0F6E56) + 蓝色强调 (#185FA5)
3. **字体**：中文默认 "Microsoft YaHei"，英文标题用 Georgia
4. **图标库**：默认 chunk-filled
5. **视觉风格**：默认 swiss-minimal
6. **叙事模式**：根据内容推断 (论述型→pyramid, 故事型→narrative, 教程型→instructional)

## 输出格式 (严格 JSON)

```json
{
  "spec_lock_content": "## canvas\\n- viewBox: 0 0 1280 720\\n- format: PPT 16:9\\n\\n## mode\\n- mode: pyramid\\n\\n...(完整 spec_lock.md 内容)...",
  "design_spec_section_ix": "## IX. Content Outline\\n\\n### Part 1: [章节名]\\n\\n#### Slide 01 - Cover\\n\\n- **Layout**: single-column-centered\\n- **Title**: [标题]\\n...(完整 §IX 内容)..."
}
```

## 注意

- 只输出 JSON，不要输出任何其他文字
- spec_lock_content 中的换行用 \\n 表示
- 颜色必须是 HEX 格式 (#RRGGBB)
- 字体大小是无单位纯数字 (如 24, 不是 24px)
- page_rhythm 每页必须有一个值
"""


# --- Agent 4: SVG Executor ---
EXECUTOR_SYSTEM_PROMPT = """\
你是 ppt-master 的 SVG 执行专家。你的任务是根据 spec_lock.md 的约束和 design_spec.md §IX 的内容，逐页生成符合 ppt-master 规范的 SVG 代码。

## 你的工作流程

1. 阅读下方提供的 "ppt-master 执行规范 (executor-base.md)" — 这是完整的 SVG 生成规范
2. 阅读下方提供的 "ppt-master 共享标准 (shared-standards.md)" — 这是 SVG 书写的硬性约束
3. 阅读下方提供的 "spec_lock.md 内容" — 这是当前 PPT 的机器可读执行契约
4. 阅读下方提供的 "当前页内容 (来自 design_spec §IX)" — 这是你要生成的具体页面内容
5. 生成一段完整的 <svg>...</svg> XML 代码

## SVG 生成硬性规则 (违反即失败)

1. viewBox 必须与 spec_lock.md canvas.viewBox 一致
2. 背景使用 <rect> 元素填充
3. 文本换行使用 <tspan>，禁止 <foreignObject>
4. 透明度使用 fill-opacity / stroke-opacity，禁止 rgba()
5. 禁止: <style>, class, <foreignObject>, textPath, @font-face, <animate*>, <script>, <iframe>
6. 禁止: <g opacity> (在子元素上单独设置 opacity)
7. 禁止: HTML 命名实体 (&nbsp; &mdash; 等)，使用原始 Unicode (— © → 等)
8. XML 保留字必须转义: & → &amp;, < → &lt;, > → &gt;
9. 颜色值必须来自 spec_lock.md 的 colors 分节
10. 字体必须来自 spec_lock.md 的 typography 分节
11. 字体大小必须来自 spec_lock.md 的 typography 分节
12. 图标必须来自 spec_lock.md 的 icons.inventory

## 布局规则

1. 安全边距：距画布边缘 40-60px
2. 内容块间距：24-40px
3. 图标与文字间距：8-16px
4. 卡片间距：20-32px，卡片内边距：20-32px
5. 卡片圆角：8-16px

## 输出格式

直接输出完整的 SVG XML 代码，从 <svg 开始到 </svg> 结束。
不要输出任何解释性文字，不要输出 markdown 代码块标记。
"""


# ============================================================================
# 第五部分：Agent 节点函数
# ============================================================================

def planner_node(state: PipelineState) -> dict:
    """Agent 1: Planner — 将原始需求拆解为结构化大纲。"""
    print("\n" + "=" * 60)
    print("  Agent 1: Planner — 正在规划大纲...")
    print("=" * 60)

    client = get_llm_client()

    # 构建 user message
    user_msg = f"用户需求：\n{state['user_prompt']}"

    # 如果有 reject 反馈，追加到 user message
    feedback = state.get("outline_feedback", "")
    if feedback:
        user_msg += f"\n\n--- 用户对上一版大纲的修改意见 ---\n{feedback}\n---\n"
        user_msg += "请根据以上修改意见重新生成大纲。"

    # 调用 LLM
    try:
        outline = call_llm_json(client, "planner", PLANNER_SYSTEM_PROMPT, user_msg)
        print(f"\n  [OK] 大纲生成完成，共 {outline.get('total_pages', '?')} 页")
        print(f"  叙事主线: {outline.get('narrative_summary', 'N/A')}")
        return {"outline": outline, "outline_feedback": ""}  # 清空 feedback
    except Exception as e:
        print(f"\n  [ERROR] Planner 失败: {e}")
        return {"error": str(e)}


def review_outline_node(state: PipelineState) -> dict:
    """Review 1: 大纲审核 — interrupt() 暂停，等待用户 approve/reject。"""
    outline = state.get("outline", {})
    pages = outline.get("pages", [])

    print("\n" + "=" * 60)
    print("  Review 1: 大纲审核")
    print("=" * 60)

    # 打印大纲
    print(f"\n  标题: {outline.get('title', 'N/A')}")
    print(f"  总页数: {outline.get('total_pages', 'N/A')}")
    print(f"  叙事主线: {outline.get('narrative_summary', 'N/A')}")
    print(f"\n  页面列表:")
    print(f"  {'─' * 56}")
    for page in pages:
        role_tag = {"cover": "封面", "chapter": "章节", "content": "内容", "closing": "结尾"}.get(
            page.get("role", ""), page.get("role", ""))
        print(f"  P{page.get('page_num', '?'):02d} [{role_tag}] {page.get('title', 'N/A')}")
        print(f"       核心主旨: {page.get('core_message', 'N/A')}")
        print(f"       布局: {page.get('layout_hint', 'N/A')}")
    print(f"  {'─' * 56}")

    # interrupt: 等待用户输入
    user_input = interrupt({
        "review_point": "outline",
        "message": "请审核以上大纲。输入 'approve' 确认，或输入修改意见来 reject 并重新生成。",
    })

    # 解析用户输入
    user_input = user_input.strip() if isinstance(user_input, str) else str(user_input).strip()

    if user_input.lower() in ("approve", "ok", "y", "yes", "确认", "通过", "可以"):
        print("\n  [OK] 大纲审核通过，继续下一步...")
        return {"outline_decision": "approve"}
    else:
        print(f"\n  [REJECT] 大纲需修改，修改意见: {user_input}")
        print("  回退到 Planner Agent 重新生成...")
        return {
            "outline_decision": "reject",
            "outline_feedback": user_input,
        }


def review_outline_router(state: PipelineState) -> str:
    """条件路由: Review 1 后决定走向 Writer 还是回退 Planner。"""
    if state.get("outline_decision") == "approve":
        return "writer"
    return "planner"


def writer_node(state: PipelineState) -> dict:
    """Agent 2: Writer — 将大纲扩展为视觉化短文案。"""
    print("\n" + "=" * 60)
    print("  Agent 2: Writer — 正在生成视觉化文案...")
    print("=" * 60)

    client = get_llm_client()
    outline = state.get("outline", {})

    # 构建 user message
    user_msg = f"PPT 大纲：\n{json.dumps(outline, ensure_ascii=False, indent=2)}"

    # 如果有 reject 反馈
    feedback = state.get("writer_feedback", "")
    if feedback:
        user_msg += f"\n\n--- 用户对上一版文案的修改意见 ---\n{feedback}\n---\n"
        user_msg += "请根据以上修改意见重新生成文案。注意保持大纲结构不变，只调整文案表达。"

    try:
        page_contents = call_llm_json(client, "writer", WRITER_SYSTEM_PROMPT, user_msg)
        pages = page_contents.get("pages", [])
        print(f"\n  [OK] 文案生成完成，共 {len(pages)} 页")
        for p in pages:
            blocks = p.get("content_blocks", [])
            print(f"  P{p.get('page_num', '?'):02d} [{p.get('role', '?')}] {p.get('title', 'N/A')} ({len(blocks)} blocks)")
        return {"page_contents": page_contents, "writer_feedback": ""}
    except Exception as e:
        print(f"\n  [ERROR] Writer 失败: {e}")
        return {"error": str(e)}


def review_content_node(state: PipelineState) -> dict:
    """Review 2: 文案审核 — interrupt() 暂停，等待用户 approve/reject。"""
    page_contents = state.get("page_contents", {})
    pages = page_contents.get("pages", [])

    print("\n" + "=" * 60)
    print("  Review 2: 文案审核")
    print("=" * 60)

    for page in pages:
        pnum = page.get("page_num", "?")
        role = page.get("role", "?")
        title = page.get("title", "N/A")
        core = page.get("core_message", "")
        layout = page.get("layout", "N/A")
        blocks = page.get("content_blocks", [])

        print(f"\n  ── P{pnum:02d} [{role}] {title} ──")
        print(f"  布局: {layout}")
        if core:
            print(f"  核心断言: {core}")
        for i, block in enumerate(blocks, 1):
            btype = block.get("type", "?")
            btext = block.get("text", "")
            icon = block.get("icon_hint", "")
            icon_str = f" [icon: {icon}]" if icon else ""
            print(f"  {i}. ({btype}){icon_str} {btext}")
        hero = page.get("hero_number")
        if hero:
            print(f"  ★ Hero number: {hero}")
        visuals = page.get("visual_elements", [])
        if visuals:
            print(f"  视觉元素: {', '.join(visuals)}")

    print("\n" + "─" * 56)

    user_input = interrupt({
        "review_point": "content",
        "message": "请审核以上文案。输入 'approve' 确认，或输入修改意见来 reject 并重新生成。",
    })

    user_input = user_input.strip() if isinstance(user_input, str) else str(user_input).strip()

    if user_input.lower() in ("approve", "ok", "y", "yes", "确认", "通过", "可以"):
        print("\n  [OK] 文案审核通过，继续下一步...")
        return {"writer_decision": "approve"}
    else:
        print(f"\n  [REJECT] 文案需修改，修改意见: {user_input}")
        print("  回退到 Writer Agent 重新生成...")
        return {
            "writer_decision": "reject",
            "writer_feedback": user_input,
        }


def review_content_router(state: PipelineState) -> str:
    """条件路由: Review 2 后决定走向 Adapter 还是回退 Writer。"""
    if state.get("writer_decision") == "approve":
        return "adapter"
    return "writer"


def adapter_node(state: PipelineState) -> dict:
    """Agent 3: Skill Adapter — 翻译文案为 ppt-master 的 spec 文件。"""
    print("\n" + "=" * 60)
    print("  Agent 3: Skill Adapter — 正在翻译为 ppt-master 格式...")
    print("=" * 60)

    client = get_llm_client()

    # 读取 ppt-master 模板参考文件
    print("  读取 spec_lock 模板参考...")
    spec_lock_ref = SPEC_LOCK_REFERENCE.read_text(encoding="utf-8")
    print("  读取 design_spec 模板参考...")
    design_spec_ref = DESIGN_SPEC_REFERENCE.read_text(encoding="utf-8")

    # 获取文案和设计偏好
    page_contents = state.get("page_contents", {})
    design_prefs = state.get("design_prefs", "未提供，请自行推断")

    # 构建 user message
    user_msg = textwrap.dedent(f"""\
        === ppt-master spec_lock 模板参考 ===
        {spec_lock_ref}

        === ppt-master design_spec 模板参考 ===
        {design_spec_ref}

        === 视觉化文案 (page_contents) ===
        {json.dumps(page_contents, ensure_ascii=False, indent=2)}

        === 用户设计偏好 (design_prefs) ===
        {design_prefs}
    """)

    try:
        result = call_llm_json(client, "adapter", ADAPTER_SYSTEM_PROMPT, user_msg)
        spec_lock_content = result.get("spec_lock_content", "")
        design_spec_ix = result.get("design_spec_section_ix", "")

        print(f"\n  [OK] spec_lock.md 生成完成 ({len(spec_lock_content)} 字符)")
        print(f"  [OK] design_spec §IX 生成完成 ({len(design_spec_ix)} 字符)")

        # 创建 ppt-master 项目
        print("\n  创建 ppt-master 项目...")
        import time
        project_name = f"ppt_{int(time.time())}"

        # 确保输出目录存在
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # 调用 project_manager.py init
        # 注意: project_manager.py 会自动追加 _ppt169_日期 后缀
        cmd = [
            sys.executable,
            str(PROJECT_MANAGER_SCRIPT),
            "init",
            project_name,
            "--format", "ppt169",
            "--dir", str(OUTPUT_DIR),
        ]
        print(f"  运行: {' '.join(cmd)}")
        result_proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        # 从 stdout 解析实际创建的项目路径
        # project_manager.py 输出格式: "Project created: D:\...\project_name_ppt169_20260624"
        project_path = None
        if result_proc.returncode == 0:
            for line in result_proc.stdout.split("\n"):
                if line.startswith("Project created:"):
                    project_path = Path(line.split(":", 1)[1].strip())
                    break

        if project_path and project_path.exists():
            print(f"  [OK] 项目创建成功: {project_path}")
        else:
            # 回退: 手动创建目录结构
            print(f"  [WARN] project_manager.py 未返回有效路径，手动创建目录")
            if result_proc.stderr:
                print(f"  stderr: {result_proc.stderr[:300]}")
            # 搜索 OUTPUT_DIR 下最新创建的匹配目录
            candidates = sorted(OUTPUT_DIR.glob(f"{project_name}*"), key=lambda p: p.stat().st_mtime, reverse=True)
            if candidates:
                project_path = candidates[0]
                print(f"  [OK] 找到项目目录: {project_path}")
            else:
                project_path = OUTPUT_DIR / project_name
                project_path.mkdir(parents=True, exist_ok=True)
                (project_path / "svg_output").mkdir(exist_ok=True)
                (project_path / "svg_final").mkdir(exist_ok=True)
                print(f"  [OK] 手动创建目录: {project_path}")

        # 写入 spec 文件
        spec_lock_path = project_path / "spec_lock.md"
        design_spec_path = project_path / "design_spec.md"

        spec_lock_path.write_text(spec_lock_content, encoding="utf-8")
        print(f"  [OK] 写入 {spec_lock_path}")

        # design_spec.md 需要包含基本结构 + §IX
        design_spec_full = f"""# {project_name} - Design Spec

## IX. Content Outline

{design_spec_ix}
"""
        design_spec_path.write_text(design_spec_full, encoding="utf-8")
        print(f"  [OK] 写入 {design_spec_path}")

        return {
            "spec_lock_content": spec_lock_content,
            "design_spec_content": design_spec_full,
            "project_path": str(project_path),
        }

    except Exception as e:
        print(f"\n  [ERROR] Adapter 失败: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def executor_node(state: PipelineState) -> dict:
    """Agent 4: SVG Executor — 逐页生成 SVG，导出 PPTX。"""
    print("\n" + "=" * 60)
    print("  Agent 4: SVG Executor — 正在逐页生成 SVG...")
    print("=" * 60)

    # 检查前置条件
    project_path_str = state.get("project_path", "")
    if not project_path_str:
        print("  [ERROR] project_path 为空，Agent 3 可能失败了")
        return {"error": "project_path 为空，Agent 3 可能失败", "svg_pages": [], "pptx_path": ""}

    project_path = Path(project_path_str)
    if not project_path.exists():
        print(f"  [ERROR] 项目目录不存在: {project_path}")
        return {"error": f"项目目录不存在: {project_path}", "svg_pages": [], "pptx_path": ""}

    # 检查是否有前置错误
    if state.get("error"):
        print(f"  [ERROR] 前置错误: {state['error']}")
        return {"svg_pages": [], "pptx_path": ""}

    client = get_llm_client()

    # 读取 ppt-master 规范文件
    print("  读取 executor-base.md (SVG 生成规范)...")
    executor_base = EXECUTOR_BASE.read_text(encoding="utf-8")
    print(f"  ({len(executor_base)} 字符)")

    print("  读取 shared-standards.md (共享标准)...")
    shared_standards = SHARED_STANDARDS.read_text(encoding="utf-8")
    print(f"  ({len(shared_standards)} 字符)")

    spec_lock = state.get("spec_lock_content", "")
    page_contents = state.get("page_contents", {})
    pages = page_contents.get("pages", [])

    # 确保 svg_output 目录存在
    svg_output_dir = project_path / "svg_output"
    svg_output_dir.mkdir(parents=True, exist_ok=True)

    # 逐页生成 SVG
    svg_pages = []
    for page in pages:
        pnum = page.get("page_num", 0)
        title = page.get("title", f"Page {pnum}")

        print(f"\n  生成 P{pnum:02d}: {title}...")

        # 构建当前页的 user message
        page_json = json.dumps(page, ensure_ascii=False, indent=2)
        user_msg = textwrap.dedent(f"""\
            === ppt-master 执行规范 (executor-base.md) ===
            {executor_base}

            === ppt-master 共享标准 (shared-standards.md) ===
            {shared_standards}

            === spec_lock.md 内容 ===
            {spec_lock}

            === 当前页内容 (来自 design_spec §IX) ===
            Page {pnum}:
            {page_json}

            请根据以上规范和约束，为第 {pnum} 页生成完整的 SVG 代码。
            viewBox 必须与 spec_lock.md 中的 canvas.viewBox 一致。
        """)

        try:
            # SVG 生成不需要 JSON 输出
            svg_content = call_llm(
                client, "executor", EXECUTOR_SYSTEM_PROMPT, user_msg, json_output=False
            )

            # 清理可能的 markdown 代码块标记
            svg_content = svg_content.strip()
            if svg_content.startswith("```"):
                lines = svg_content.split("\n")
                svg_content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

            # 确保以 <svg 开头
            svg_start = svg_content.find("<svg")
            if svg_start > 0:
                svg_content = svg_content[svg_start:]

            # 写入 SVG 文件
            svg_filename = f"{pnum:02d}_{page.get('role', 'content')}.svg"
            svg_path = svg_output_dir / svg_filename
            svg_path.write_text(svg_content, encoding="utf-8")
            print(f"  [OK] 写入 {svg_path} ({len(svg_content)} 字符)")

            svg_pages.append({"page_num": pnum, "svg": svg_content, "filename": svg_filename})

        except Exception as e:
            print(f"  [ERROR] P{pnum:02d} SVG 生成失败: {e}")
            svg_pages.append({"page_num": pnum, "error": str(e)})

    print(f"\n  SVG 生成完成: {len(svg_pages)} 页")

    # 后处理: finalize_svg.py (svg_output/ → svg_final/)
    print("\n  运行 SVG 后处理 (finalize_svg.py)...")
    if FINALIZE_SVG_SCRIPT.exists():
        cmd = [sys.executable, str(FINALIZE_SVG_SCRIPT), str(project_path)]
        print(f"  运行: {' '.join(cmd)}")
        result_proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result_proc.returncode != 0:
            print(f"  [WARN] finalize_svg.py 返回非零:")
            print(f"  {result_proc.stderr[:500]}")
        else:
            print("  [OK] SVG 后处理完成 (svg_output/ → svg_final/)")
    else:
        print(f"  [WARN] finalize_svg.py 不存在: {FINALIZE_SVG_SCRIPT}")

    # 导出 PPTX: svg_to_pptx.py
    print("\n  导出 PPTX (svg_to_pptx.py)...")
    pptx_path = project_path / "exports" / f"{project_path.name}.pptx"
    pptx_path.parent.mkdir(exist_ok=True)

    if SVG_TO_PPTX_SCRIPT.exists():
        cmd = [
            sys.executable, str(SVG_TO_PPTX_SCRIPT),
            str(project_path),
            "-o", str(pptx_path),
        ]
        print(f"  运行: {' '.join(cmd)}")
        result_proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result_proc.returncode != 0:
            print(f"  [WARN] svg_to_pptx.py 返回非零:")
            print(f"  {result_proc.stderr[:500]}")
            # 尝试无 -o 参数的默认输出
            print("  尝试默认输出路径...")
            cmd_default = [sys.executable, str(SVG_TO_PPTX_SCRIPT), str(project_path)]
            result_proc2 = subprocess.run(cmd_default, capture_output=True, text=True, timeout=300)
            if result_proc2.returncode == 0:
                # 搜索默认输出的 pptx
                pptx_files = list(project_path.rglob("*.pptx"))
                if pptx_files:
                    pptx_path = pptx_files[0]
                    print(f"  [OK] PPTX 导出成功 (默认路径): {pptx_path}")
                else:
                    print(f"  [WARN] 未找到导出的 PPTX 文件")
            else:
                print(f"  [WARN] {result_proc2.stderr[:500]}")
        else:
            print(f"  [OK] PPTX 导出成功: {pptx_path}")
    else:
        print(f"  [WARN] svg_to_pptx.py 不存在: {SVG_TO_PPTX_SCRIPT}")
        print(f"  SVG 文件已生成在: {svg_output_dir}")
        print(f"  请手动运行 ppt-master 的导出流程。")

    return {
        "svg_pages": svg_pages,
        "pptx_path": str(pptx_path) if pptx_path.exists() else "",
    }


# ============================================================================
# 第六部分：LangGraph 图组装
# ============================================================================

def build_graph():
    """构建 LangGraph 状态图。"""
    graph = StateGraph(PipelineState)

    # 添加节点
    graph.add_node("planner", planner_node)
    graph.add_node("review_outline", review_outline_node)
    graph.add_node("writer", writer_node)
    graph.add_node("review_content", review_content_node)
    graph.add_node("adapter", adapter_node)
    graph.add_node("executor", executor_node)

    # 设置入口
    graph.set_entry_point("planner")

    # 边: planner → review_outline
    graph.add_edge("planner", "review_outline")

    # 条件边: review_outline → writer (approve) 或 planner (reject)
    graph.add_conditional_edges(
        "review_outline",
        review_outline_router,
        {"writer": "writer", "planner": "planner"},
    )

    # 边: writer → review_content
    graph.add_edge("writer", "review_content")

    # 条件边: review_content → adapter (approve) 或 writer (reject)
    graph.add_conditional_edges(
        "review_content",
        review_content_router,
        {"adapter": "adapter", "writer": "writer"},
    )

    # 边: adapter → executor → END
    graph.add_edge("adapter", "executor")
    graph.add_edge("executor", END)

    return graph.compile()


# ============================================================================
# 第七部分：主入口
# ============================================================================

def main():
    """主入口函数。"""
    print("=" * 60)
    print("  多 Agent PPT 生成流水线")
    print("  LangGraph + DeepSeek + ppt-master")
    print("=" * 60)

    # 获取用户输入
    print("\n请输入 PPT 需求 (输入空行结束):")
    user_prompt = input("> ").strip()
    if not user_prompt:
        print("需求不能为空。")
        return

    print("\n请输入设计偏好 (可选，直接回车跳过):")
    print("  示例: 画布16:9, 深蓝科技风, 微软雅黑字体, chunk-filled图标库")
    design_prefs = input("> ").strip()
    if not design_prefs:
        design_prefs = ""
        print("  (未提供，将由 Agent 3 自动推断)")

    # 初始化状态
    initial_state = {
        "user_prompt": user_prompt,
        "design_prefs": design_prefs,
    }

    # 构建并运行图 (带 MemorySaver 支持 interrupt 恢复)
    print("\n启动 LangGraph 流水线...\n")
    app = build_graph_with_checkpoint()
    config = {"configurable": {"thread_id": "ppt-pipeline-1"}}

    try:
        # 第一次运行 (会在 review_outline 处 interrupt)
        result = app.invoke(initial_state, config=config)

        # 如果在 interrupt 处暂停，处理用户输入
        while result and "__interrupt__" in result:
            interrupt_info = result["__interrupt__"]
            review_point = interrupt_info[0].value.get("review_point", "unknown")
            message = interrupt_info[0].value.get("message", "请审核")

            print(f"\n{'─' * 56}")
            print(f"  ⏸ {message}")
            print(f"{'─' * 56}")

            user_response = input("\n请输入 (approve 或 修改意见): ").strip()
            if not user_response:
                user_response = "approve"

            # 恢复执行
            result = app.invoke(Command(resume=user_response), config=config)

    except Exception as e:
        print(f"\n[ERROR] 流水线执行失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 输出结果
    print("\n" + "=" * 60)
    print("  流水线执行完成!")
    print("=" * 60)

    if result.get("error"):
        print(f"\n  [错误] {result['error']}")
    else:
        pptx_path = result.get("pptx_path", "")
        project_path = result.get("project_path", "")
        svg_pages = result.get("svg_pages", [])

        print(f"\n  项目目录: {project_path}")
        print(f"  SVG 页数: {len(svg_pages)}")
        if pptx_path:
            print(f"  PPTX 文件: {pptx_path}")
        else:
            print(f"  PPTX 导出: 未成功 (请检查 svg_output/ 目录手动导出)")

    print()


def build_graph_with_checkpoint():
    """构建带 MemorySaver 的 LangGraph 图 (支持 interrupt 恢复)。"""
    from langgraph.checkpoint.memory import MemorySaver

    graph = StateGraph(PipelineState)

    graph.add_node("planner", planner_node)
    graph.add_node("review_outline", review_outline_node)
    graph.add_node("writer", writer_node)
    graph.add_node("review_content", review_content_node)
    graph.add_node("adapter", adapter_node)
    graph.add_node("executor", executor_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "review_outline")
    graph.add_conditional_edges(
        "review_outline",
        review_outline_router,
        {"writer": "writer", "planner": "planner"},
    )
    graph.add_edge("writer", "review_content")
    graph.add_conditional_edges(
        "review_content",
        review_content_router,
        {"adapter": "adapter", "writer": "writer"},
    )
    graph.add_edge("adapter", "executor")
    graph.add_edge("executor", END)

    return graph.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    main()
