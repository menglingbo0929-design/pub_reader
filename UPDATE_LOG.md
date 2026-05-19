# Update Log

## 2026-05-19

- Added README instructions for finding the cloned project path and creating the desktop shortcut with PowerShell.
- Changed the default library location from `~/.pub_reader/library` to the project-local `output/` directory.
- Added a tracked `output/默认文件夹/` starter folder.
- Changed paper output folders to use the uploaded PDF file name.
- Kept the original PDF file name inside each paper folder, alongside `translated.md` and `summary.md`.
- Added translation chunk progress updates so long DeepSeek calls do not appear frozen at 42%.
- Added small code comments around project-root detection, Markdown skeleton replacement, and DeepSeek request payloads.
- Updated the setup script so the desktop shortcut installation keeps `library_dir` pointed at the cloned project's `output/` folder.
- Added explanatory comments across configuration, library management, PDF extraction, LLM calls, and the UI worker flow.
