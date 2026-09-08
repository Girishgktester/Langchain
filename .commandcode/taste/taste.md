# Taste

- Prefers a commit-and-push to `main` workflow after each batch of work, frequently requesting "push latest changes" with a meaningful commit message. Confidence: 0.9
- Prefers descriptive, meaningful commit messages that reflect the actual diff (e.g. "Add parallel and conditional runnable examples"), and explicitly asks for detailed/explanatory commit messages when work is more involved. Confidence: 0.9
- Works in a Windows environment using PowerShell and Jupyter notebooks (`.ipynb`) for LangChain learning; expects git/build commands to run via PowerShell. Confidence: 0.8
- Prefers staging specific changed files (using `git add <specific paths>`) and combining stage + commit in one command rather than staging everything blindly. Confidence: 0.7
- Willing to knowingly publish potentially sensitive files to an unverified/destination repo when explicitly asked, after being informed of the risk (e.g. approving push of a profile PDF). Confidence: 0.6