# 安装与接入 DSH

> **这份文档假设你从零开始，手上只有本仓库和一个能跑的 DSH。**

---

## 一、装 DSH

**本仓库不含 DSH。** 它是 MIT 开源项目，从官方仓库安装：

```bash
git clone https://github.com/deepseek-ai/deepseek-harness.git
cd deepseek-harness
pnpm install
pnpm build
```

**⚠️ 兼容版本（见插件 `package.json` 的 peerDependencies）：**

```
@deepseek-ai/dsh        >= 0.2.0-rc.2
@deepseek-ai/dsh-tools  >= 0.2.0-rc.2
@deepseek-ai/cordis     >= 4.0.4
```

**我们用过的版本：`@deepseek-ai/dsh` 0.2.2-rc.1。**

---

## 二、建目录

**所有路径都基于一个根目录，下文用 `<BENCH>` 表示：**

```bash
export BENCH=/opt/pagi            # Linux
# 或  set BENCH=D:\pagi           # Windows

mkdir -p $BENCH
cp -r bench/ $BENCH/
cp -r docs/  $BENCH/
cp -r data/  $BENCH/
```

**建好后长这样：**

```
$BENCH/
├── generator_v2/          出卷器
├── plugin/                门控插件
└── dsh-env/               运行环境（下面建）
```

---

## 三、替换占位符（唯一需要改代码的地方）

**源码里的路径已脱敏为占位符：**

| 占位符 | 你要换成 |
|---|---|
| `<BENCH>` | 你的项目根目录 |
| `<HOME>` | 上级目录 |
| `<DSH>` | DSH 安装路径 |

**批量替换（Linux / macOS）：**

```bash
cd $BENCH/generator_v2
sed -i "s|<BENCH>|$BENCH|g" *.py
sed -i "s|<HOME>|$(dirname $BENCH)|g" *.py
sed -i "s|<DSH>|$DSH_PATH|g" *.py
```

**Windows PowerShell：**

```powershell
$BENCH = "D:\pagi"
Get-ChildItem "$BENCH\generator_v2\*.py" | ForEach-Object {
    (Get-Content $_.FullName -Raw) `
        -replace '<BENCH>', $BENCH `
        -replace '<HOME>', (Split-Path $BENCH -Parent) `
        -replace '<DSH>', 'D:\dsh' |
    Set-Content $_.FullName -NoNewline
}
```

**⚠️ 反斜杠问题：源码里是 Windows 风格路径 `r'<BENCH>\generator_v2'`。
在 Linux 上替换后 `\` 不会自动变成 `/`，需要额外一步：**

```bash
sed -i 's|\\|<SLASH>|g' *.py    # 临时
sed -i "s|<SLASH>|/|g" *.py
```

**或者更稳妥：手动检查一遍 `sys.path.insert` 和 `ENV_ROOT` 那几行。**

---

## 四、建 dsh-env

**DSH 需要一个独立的 HOME（不要把考场混进你的日常 DSH 环境）：**

```bash
export DSH_HOME_ENV=$BENCH/dsh-env/dsh-home
mkdir -p $DSH_HOME_ENV/profiles/exam/node_modules/@sroada
mkdir -p $DSH_HOME_ENV/sessions
```

### 装门控插件

```bash
cp -r $BENCH/plugin/dsh-exam-gate \
      $DSH_HOME_ENV/profiles/exam/node_modules/@sroada/
```

**⚠️ 插件只有两个文件（`package.json` + `lib/index.js`），
`dependencies` 为空，所有运行时依赖由 DSH 提供。**

### 放 cordis 配置

```bash
cp $BENCH/plugin/../dsh-env/profiles/exam/cordis.patch.yml \
   $DSH_HOME_ENV/profiles/exam/
```

**这份配置声明了插件怎么挂、挂在哪里。**

---

## 五、装 DSH CLI

**出卷器要调用 DSH 的 JSON-RPC 接口，需要它的 CLI 入口：**

```bash
mkdir -p $BENCH/dsh-env/node_modules/@deepseek-ai
ln -s $DSH_PATH/apps/cli \
      $BENCH/dsh-env/node_modules/@deepseek-ai/dsh
```

**⚠️ 出卷器期望的路径是：**

```
$BENCH/dsh-env/node_modules/@deepseek-ai/dsh/lib/bin.js
```

**如果实际路径不同，改 `runexam.py` 里的 `CLI` 常量。**

---

## 六、配置模型

**考试要真的调模型。设置环境变量：**

```bash
export DEEPSEEK_API_KEY=sk-xxxxxxxxxxxx
```

**Windows（用户级）：**

```powershell
[Environment]::SetEnvironmentVariable(
    'DEEPSEEK_API_KEY', 'sk-xxxxxxxxxxxx', 'User')
```

**⚠️ `runexam.py` 会从环境变量读 key 并传给 DSH 子进程。
`dshlink.py` 会主动清空它（那是给子代理用的），只有真考试才注入。**

---

## 七、验证

### 第一步：复核

```bash
python $BENCH/generator_v2/checkall.py
```

**应该看到 20 项全过，约 180 秒。**

**⚠️ 如果 A 组 `mathpack` 就失败，说明 torch 没装好。**

### 第二步：出卷

```bash
cd $BENCH/generator_v2
python assemble.py
```

**⚠️ 这一步会 GPU 生成数值、出 60 张图，约 1–2 分钟。**

### 第三步：交付

```python
import sys
sys.path.insert(0, '/opt/pagi/generator_v2')
import assemble, pack, handoff

pp = assemble.assemble()
root = pack.pack_paper(pp)
r = handoff.deliver(root)
print(r['dst'])
```

**应该看到：**

```
交付 ok = True
沙箱 = ['exam_public.json', 'figs', 'paper.md', 'prompt.md']
```

**⚠️ 沙箱里**不能**出现 `answer.json` / `exam_secret.json`。
`verify_delivery` 会检查，不通过就报错。**

### 第四步：开考

```bash
python $BENCH/generator_v2/hold.py
```

**应该看到：**

```
EXAM_PICKED EXAM-xxxxxxxx-xxxxxx-xxxxxx
STARTED EXAM-xxxxxxxx-xxxxxx-xxxxxx
```

**然后：**

```bash
# 看门控有没有在工作
tail -f $BENCH/dsh-env/logs/cheat/cheat-<EXAM_ID>.jsonl
```

**⚠️ 如果这个文件不生成，说明门控没读到卷子特征 —— 检查
`exam_public.json` 是不是叫这个名字（不是 `exam.json`）。**

### 第五步：停止

```bash
touch $BENCH/dsh-env/logs/STOP
```

**`hold.py` 每 2 秒检查一次，看到就停。**

---

## 八、常见问题

### 门控没记录

**检查顺序：**

```
① logs/cheat/ 目录有没有新建 jsonl
② exam_public.json 里有没有 features 段
③ 插件有没有真被 cordis 加载（看 stderr 有没有插件名）
```

### `session "..." already exists`

**一套卷子只能考一次。** session id 是 `exam-<EXAM_ID>`。

**⇒ 换新卷子。不要复用。**

### DSH 进程起来了但没动静

**`initialize` 之后要等 MCP 注册完成（stderr 出现 `[mcp]`），
`runexam.py` 里等了 30 秒。**

**如果还不行，看 `logs/run-<EXAM_ID>.err.log`。**

### 图读不出来

**卷面里的图路径是绝对路径。** 如果换了机器，卷子要重新交付，
因为 `handoff.deliver` 会把路径改成新位置的绝对路径。

### `<BENCH>` / `<BENCH>` 找不到

**这是脱敏占位符。** 按第三节替换即可。

**@sroada/dsh-exam-gate 里的 sroada 是 npm 作用域名，不要改。**

---

## 九、跑一场完整考试的时长

| 阶段 | 时长 |
|---|---|
| 出卷（含出图） | 1–2 分钟 |
| 打包交付 | 10 秒 |
| 考试（模型答题） | 8–45 分钟 |
| 判分 | 1 秒 |

**⚠️ 考试时长取决于模型。我们观察到 460–2480 秒。**

---

**最后更新：2026-11-10**
