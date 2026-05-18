# Pub Reader

Pub Reader 是一个用于英文论文阅读的桌面应用。它可以导入 PDF 论文，调用 DeepSeek API 将正文、图名、表格文本翻译为中文 Markdown，并生成一份按论文结构组织的中文 brief summary。

## 主要功能

- 桌面端资料库：新建、删除、重命名文件夹。
- 每篇论文独立存储：原始 PDF、中文译文 Markdown、中文 summary Markdown。
- PDF 导入：从 Abstract / Introduction 附近开始处理正文内容。
- 领域感知翻译：先依据 Abstract / Introduction 判断研究领域，再要求模型按领域术语习惯翻译。
- Markdown 输出：保留原论文阅读顺序，图片会抽取到 `assets/` 并在 Markdown 中引用。
- API key 弹窗：点击生成时输入 DeepSeek API key，不在项目中硬编码密钥。

## 快速运行

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
python -m pub_reader
```

## 打包 Windows 应用

```powershell
.\scripts\build_windows.ps1
```

打包产物会生成在 `dist\PubReader\`。如果你安装了 Inno Setup，还可以用 `installer\pub_reader.iss` 生成 `PubReaderSetup.exe`。

## DeepSeek 配置

默认 API 地址是：

```text
https://api.deepseek.com/chat/completions
```

默认模型名是：

```text
deepseek-v4-pro
```

如果 DeepSeek 账号中的模型名不同，可以在应用目录下的 `config.json` 中修改，或者在代码中的 `pub_reader/config.py` 调整默认值。
