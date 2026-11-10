# 准 AGI 基准测试（Quasi-AGI Benchmark）

> **用「工具自主认知」测 AGI 准入门槛的开源基准。**
>
> 本仓库包含出卷器、考场门控插件、论文源码与全部实测数据。

---

## 这是什么

**一个基准测试系统。它给被测模型出题，观察它能不能在给定边界内自主完成任务。**

具体形态：

```
题目：A000 × [ ] × [ ] × … × [ ] = Z

已知      A000、Z
未知      10 个空位
选项      60 个，每个是「函数(变量) ÷ 系数」的组合
数据      每个选项的变量值藏在图里（科学计数法，需逐位数图案）

工具      90 个计算工具 —— 编号 T-001…T-090
          其中 58 个是选项里用得到的函数
              16 个是它们的反函数
              16 个完全用不上（干扰项）

要求      从 60 个选项里挑出 10 个正确的，填进空位
```

**测的不是算得对不对，是：**

```
① 能不能只靠行为描述认出工具（描述里没有函数名）
② 会不会自己配系数（卷面写除法，工具做乘法）
③ 能不能守住边界（沙箱内随便翻，翻出去算作弊）
④ 会不会把单步串成长链
```

---

## 快速开始

### 依赖

```
Python   3.11+，带 torch（GPU 可选，CPU 也能跑但慢）
Node.js  20+（跑 DSH）
DSH      >= 0.2.0-rc.2      https://github.com/deepseek-ai/deepseek-harness
```

**⚠️ 本仓库不含 DSH。** DSH 是 MIT 开源项目，请自行安装：

```bash
git clone https://github.com/deepseek-ai/deepseek-harness.git
cd deepseek-harness
pnpm install && pnpm build
```

### 路径配置（必读）

**源码与数据里的路径已脱敏为占位符：**

| 占位符 | 含义 | 你需要换成 |
|---|---|---|
| `<BENCH>` | 项目根目录（含 `generator_v2/`、`dsh-env/`） | 你自己 clone 后的路径 |
| `<HOME>` | 该根目录的上级 | 同上 |
| `<DSH>` | DSH 安装位置 | 你的 DSH 路径 |

**例如源码里：**

```python
sys.path.insert(0, r'<BENCH>\generator_v2')
ENV_ROOT = r'<BENCH>\dsh-env'
```

**换成（Linux 示例）：**

```python
sys.path.insert(0, '/opt/pagi/generator_v2')
ENV_ROOT = '/opt/pagi/dsh-env'
```

**⚠️ 我们只做文本替换，没有改语法。** 所以改动是「把占位符换成你的路径」这一件事，
不需要理解代码结构。

详见 [INSTALL.md](INSTALL.md)。

### 三步跑通

```bash
# 1. 出卷
python bench/generator_v2/assemble.py

# 2. 打包并交付到沙箱
python -c "import sys; sys.path.insert(0,'bench/generator_v2'); \
           import assemble, pack, handoff; \
           pp=assemble.assemble(); r=pack.pack_paper(pp); \
           handoff.deliver(r)"

# 3. 开考
python bench/generator_v2/hold.py
```

**⚠️ 第 3 步会启动 DSH 子进程并向模型发题。**
**主控不注入任何提示，模型通过 MCP 接口自己调用工具。**

---

## 目录结构

```
bench/
├── generator_v2/          出卷器 + 答卷器（35 个 Python 模块）
│   ├── assemble.py        主流程总装
│   ├── optbase.py         60 个选项的构造
│   ├── midlayer.py        选项渲染成中文式子
│   ├── figs.py            出图（按展示顺序命名 001.png…）
│   ├── packs.py           74 个数理包
│   ├── registry.py        74 个函数登记
│   ├── toolsplus.py       90 个工具（58 正向 + 16 反向 + 16 干扰）
│   ├── toolconsole.py     控制台，全量记录
│   ├── mcpserver.py       MCP 服务，7 个工具
│   ├── examfile.py        卷子读写、特征、卷子选择
│   ├── handoff.py         交付到沙箱
│   ├── checkall.py        一次跑完 20 项复核
│   ├── hold.py            起考试并保持父进程存活
│   └── runexam.py         DSH 握手（JSON-RPC over stdio）
├── plugin/
│   └── dsh-exam-gate/     考场门控插件（DSH 插件，MIT）
└── dsh-env/profiles/exam/
    └── cordis.patch.yml   插件组合配置

docs/
└── chapters/              论文正文（按章节拆分）
```

**⚠️ 实测数据与开发日志不在本仓库。**

**它们在私有仓库 `pagi-benchmark-data`：**

```
pagi-benchmark-data/
├── measurements/raw/      会话日志、门控记录、卷子、封包
└── devlog/                开发日志（20 篇，含全部返工记录）
```

**本仓库只放「怎么跑」，不放「跑出了什么」。**

---

## 门控插件

**`@sroada/dsh-exam-gate` 是一个 DSH 插件，挂在 `tools/post-execute` 和 `session/event` 上。**

**它做三件事：**

```
① 观察 —— 每次工具调用记一行，判定 in_scope / out_scope_*
            ⚠️ 只旁观，物理上不可能拦截（用 post-execute 而非 pre-execute）
② 挂起 —— 模型没交卷就停了，注入提示把它拉回来（最多 8 次）
③ 终止 —— 收到 submit 就 cancel 会话
```

**它不做的事：**

```
✗ 不注入任何解题提示
✗ 不在正常答题时打断
✗ 不改工具返回的内容
```

**安装：**

```bash
# 把插件放进 profile 的 node_modules
mkdir -p $DSH_HOME/profiles/exam/node_modules/@sroada
cp -r bench/plugin/dsh-exam-gate \
      $DSH_HOME/profiles/exam/node_modules/@sroada/

# 用配套的 cordis 配置
cp bench/dsh-env/profiles/exam/cordis.patch.yml \
   $DSH_HOME/profiles/exam/
```

**⚠️ 插件不能 `export const Config`。** cordis 看到这个导出会去读
`Config['~standard'].validate`，`undefined` 会抛 TypeError。

---

## 数据

**实测数据（会话日志、门控记录、卷子、封包）与开发日志
单独存放在私有仓库 `pagi-benchmark-data`，不随代码公开。**

**原因：**

```
① 体积 —— 数据 4.25 MB，与代码 1.16 MB 节奏不同
② 性质 —— 代码是「怎么跑」，数据是「跑出了什么」
③ Zenodo 不支持私有仓库自动存档，公开仓库才是存档对象
```

**论文相关的统计结果见 `docs/chapters/`。**

---

## ⚠️ 已知缺陷（重要）

**v1.0 实现中，`toolconsole.show_paper()` 曾把 90 个工具的列表
标注为「选项」返回给被测模型。**

**后果：**

```
模型收到 90 个编号叫 T-xxx 的「选项」
⇒ 它把工具编号填进答案格
⇒ 它的推理写着 "they say 60 but list 90" —— 这个观察是准确的
```

**⚠️ 因此第 7 场（`-090544`）的数据不能用于评估模型能力 ——
这不是「模型抽象能力不足」，是「基准设计的纰漏」。**

**已修复：`show_paper` 改为从 `pp.ops` 取 60 个选项，
工具只在 `list_tools` 里列（90 个，`T-` 开头）。**

**凡是引用 2026-11-10 09:00 之前的场次，必须先核对该场用的是哪个版本。**

**⚠️ 另外：本项目由单人完成，开发过程中存在未备份的情况。**

**⇒ 能提供：用最终代码可复现出同样的卷子和同样的判定逻辑。**
**⇒ 不能提供：每一个中间版本都能重放。**

---

## 许可

| 对象 | 许可 |
|---|---|
| 代码（`bench/`） | **MIT** — 见 [LICENSE](LICENSE) |
| 数据 | **CC-BY-4.0** — 见 [LICENSE-DATA](LICENSE-DATA) |
| 论文（`docs/`） | **CC-BY-4.0** |

**简言之：代码随便用，数据和论文用了请署名。**

---

## 引用

**见 [CITATION.cff](CITATION.cff)。**

**⚠️ 正式发表前请引用 GitHub 仓库；Zenodo DOI 生成后以其为准。**

---

## 联系

**GitHub Issues。**（仓库公开后可提交）
