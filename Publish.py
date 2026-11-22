# -- IMPORTS --
import json
import re
import subprocess
from pathlib import Path

# -- PATHS --
Root = Path(__file__).resolve().parent
WebsiteFolder = Root / "website"
PagePath = WebsiteFolder / "index.html"

# -- DATA --
Blocks: dict[str, Path] = {
	"SimData": Root / "SimResults.json",
	"MarksData": Root / "SimMarks.json",
	"MarkerData": Root / "MarkerResults.json",
	"LiveData": Root / "LiveTestSummary.json",
}
CommitMessage = "Update test results"
GitTimeout: int = 90


# -- FUNCTIONS --
def Fill() -> bool:
	Page = PagePath.read_text(encoding="utf-8")
	Fresh = Page
	for Id, Source in Blocks.items():
		if not Source.exists():
			continue
		Data = json.dumps(json.loads(Source.read_text(encoding="utf-8")), separators=(",", ":"))
		Pattern = re.compile(rf'(<script type="application/json" id="{Id}">)(.*?)(</script>)', re.S)
		Fresh = Pattern.sub(lambda Match: Match.group(1) + Data + Match.group(3), Fresh, count=1)
	if Fresh == Page:
		return False
	PagePath.write_text(Fresh, encoding="utf-8")
	return True


def Git(*Arguments: str) -> subprocess.CompletedProcess[str]:
	return subprocess.run(["git", "-C", str(WebsiteFolder), *Arguments], capture_output=True, text=True, timeout=GitTimeout)


def Upload() -> str:
	try:
		if Git("rev-parse", "--is-inside-work-tree").returncode != 0:
			return "Not uploaded: this folder is not a Git repository yet. See README, Upload the website."
		Git("add", ".")
		if Git("diff", "--cached", "--quiet", "--", ".").returncode == 0:
			return "Nothing new to upload."
		Commit = Git("commit", "-m", CommitMessage, "--", ".")
		if Commit.returncode != 0:
			return f"Not uploaded: commit failed. {Commit.stderr.strip() or Commit.stdout.strip()}"
		Push = Git("push")
		if Push.returncode != 0:
			return f"Saved locally, but the push failed. {Push.stderr.strip()}"
		return "Uploaded. The live site updates in about a minute."
	except FileNotFoundError:
		return "Not uploaded: Git is not installed on this computer."
	except subprocess.TimeoutExpired:
		return "Not uploaded: Git did not answer in time. Check the internet connection and run python Publish.py."


def Sync() -> None:
	if not PagePath.exists():
		print("[PUBLISH] No website folder found, nothing to update")
		return
	Changed = Fill()
	print(f"[PUBLISH] Website numbers {'updated' if Changed else 'already up to date'} | {Upload()}")


# -- INIT --
if __name__ == "__main__":
	Sync()
