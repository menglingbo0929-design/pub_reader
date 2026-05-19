# Pub Reader

Pub Reader 是一个用于英文论文阅读的桌面应用。它可以导入 PDF 论文，调用 DeepSeek API 将正文、图名、表格文本翻译为中文 Markdown，并生成一份按论文结构组织的中文 brief summary。

## 主要功能

- 桌面端资料库：新建、删除、重命名文件夹。
- 项目内输出：所有论文资料默认存放在项目根目录的 `output/`。
- 每篇论文独立存储：以导入的 PDF 文件名创建论文文件夹，里面保存原 PDF、`translated.md` 和 `summary.md`。
- PDF 导入：从 Abstract / Introduction 附近开始处理正文内容。
- 领域感知翻译：先依据 Abstract / Introduction 判断研究领域，再要求模型按领域术语习惯翻译。
- Markdown 输出：保留原论文阅读顺序，图片会抽取到 `assets/` 并在 Markdown 中引用。
- API key 弹窗：点击生成时输入 DeepSeek API key，不在项目中硬编码密钥。

## 从 GitHub 下载项目

建议把项目 clone 到一个固定目录，例如 `D:\pub_reader_version_1`：

```powershell
cd D:\
mkdir pub_reader_version_1
cd pub_reader_version_1
git clone https://github.com/menglingbo0929-design/pub_reader.git
cd pub_reader
git checkout codex/pub-reader-desktop
```

如果 GitHub 连接超时，通常是 Git 没有走本机代理。可以先设置：

```powershell
git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890
```

## 找到 clone 下来的项目路径

如果忘了项目放在哪里，可以在 PowerShell 中搜索：

```powershell
Get-ChildItem D:\ -Directory -Filter pub_reader -Recurse -ErrorAction SilentlyContinue | Select-Object FullName
```

找到后进入项目目录，例如：

```powershell
cd D:\pub_reader_version_1\pub_reader
```

## 直接运行源码

PowerShell 可能会禁止执行 `.venv\Scripts\Activate.ps1`。不用激活也可以运行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pub_reader
```

## 打包 Windows 应用

在项目根目录运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_windows.ps1
```

打包产物会生成在：

```text
dist\PubReader\PubReader.exe
```

## 创建桌面快捷方式

打包完成后，运行安装脚本：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\dist\PubReaderSetup.ps1
```

它会把应用安装到：

```text
C:\Users\你的用户名\AppData\Local\Programs\PubReader
```

并创建桌面快捷方式：

```text
C:\Users\你的用户名\Desktop\Pub Reader.lnk
```

如果只是想测试程序，也可以直接双击：

```text
dist\PubReader\PubReader.exe
```

## 输出目录结构

应用默认在项目根目录创建并使用：

```text
output\
  默认文件夹\
```

在应用中创建的新文件夹都会放在 `output/` 下。处理某篇论文时，会按 PDF 文件名创建论文目录，例如导入 `GAD.pdf` 后：

```text
output\
  test\
    GAD\
      GAD.pdf
      translated.md
      summary.md
      assets\
```

## DeepSeek 配置

默认 API 地址是：

```text
https://api.deepseek.com/chat/completions
```

默认模型名是：

```text
deepseek-v4-pro
```

如果 DeepSeek 账号中的模型名不同，可以在应用配置文件中修改，或在 `pub_reader/config.py` 中调整默认值。
