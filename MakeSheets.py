# -- IMPORTS --
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import Layout

# -- PATHS --
Root = Path(__file__).resolve().parent
PrintFolder = Root / "Print"
TruthPath = Root / "Truth.json"
KeysPath = Root / "Keys.json"

# -- DATA --
PrintDpi: float = 300.0
Font = cv2.FONT_HERSHEY_DUPLEX
Ink: int = 20
Ring: int = 120
Soft: int = 110
MarkStyles: tuple[str, ...] = ("Solid", "Pencil", "Scribble", "Partial")


# -- FUNCTIONS --
def Px(Value: float, Density: float) -> int:
	return int(round(Value * Density))


def Write(Image: np.ndarray, Text: str, X: float, Y: float, Height: float, Density: float, Shade: int = Ink, Align: str = "Left") -> None:
	FontScale = cv2.getFontScaleFromHeight(Font, max(Px(Height, Density), 1), 1)
	Thickness = max(1, Px(Height * 0.1, Density))
	(Width, _), _ = cv2.getTextSize(Text, Font, FontScale, Thickness)
	Left = Px(X, Density)
	if Align == "Right":
		Left -= Width
	elif Align == "Centre":
		Left -= Width // 2
	cv2.putText(Image, Text, (Left, Px(Y, Density)), Font, FontScale, Shade, Thickness, cv2.LINE_AA)


def DrawSheet(TestId: int, Density: float, Form: Layout.Form) -> np.ndarray:
	Image = np.full((Px(Layout.PageHeight, Density), Px(Layout.PageWidth, Density)), 255, dtype=np.uint8)
	Dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
	Side = Px(Layout.MarkerSize, Density)

	for Id, (X, Y) in Layout.MarkerOrigins(TestId).items():
		Marker = cv2.aruco.generateImageMarker(Dictionary, Id, Side)
		Left, Top = Px(X, Density), Px(Y, Density)
		Image[Top:Top + Side, Left:Left + Side] = Marker

	Marks = Form.Mode == "Marks"
	Write(Image, "GLANCE", 30, 16.5, 7.0, Density)
	Write(Image, f"{'MARKS' if Marks else 'ANSWER'} SHEET  /  TEST {TestId:02d}", 30, 22.5, 2.6, Density, Soft)
	if Marks:
		Write(Image, "One bubble per question for the marks given. Add .5 for a half mark.", 180, 13.5, 2.3, Density, Soft, "Right")
	else:
		Write(Image, "One bubble per question. Fill it fully, dark pen or pencil.", 180, 13.5, 2.3, Density, Soft, "Right")
	Write(Image, "Keep all four corner squares visible to the camera.", 180, 18.5, 2.3, Density, Soft, "Right")

	Line = max(1, Px(0.3, Density))
	HeadY = Layout.RowTop - 5.2
	RollRadius = Px(Layout.BubbleRadius, Density)

	Write(Image, "ROLL NO.", 28.5, HeadY - 4.4, 2.2, Density, Soft)
	for Digit in range(Layout.RollDigits):
		Write(Image, "HTO"[Digit], Layout.RollFirstX + Digit * Layout.RollPitch, HeadY, 2.6, Density, Soft, "Centre")
	for Value in range(10):
		Write(Image, str(Value), Layout.RollLabelX, Layout.RowY(Value) + 1.4, 2.8, Density, Soft, "Centre")
		for Digit in range(Layout.RollDigits):
			X, Y = Layout.RollBubble(Digit, Value)
			cv2.circle(Image, (Px(X, Density), Px(Y, Density)), RollRadius, Ring, Line, cv2.LINE_AA)

	Divider = Px(47.5, Density)
	cv2.line(Image, (Divider, Px(Layout.RowTop - 9, Density)), (Divider, Px(Layout.RowY(9) + 4, Density)), 190, Line, cv2.LINE_AA)

	HeadHeight = 2.6 if Form.Radius >= 2.6 else 2.1
	for Text, X in Form.Heads:
		Write(Image, Text, X, HeadY, HeadHeight, Density, Soft, "Centre")
	Radius = Px(Form.Radius, Density)
	for Row, Centres in enumerate(Form.Rows):
		Y = Centres[0][1]
		Write(Image, Form.Labels[Row], Form.LabelXs[Row], Y + 1.4, 2.8, Density, Ink, "Right")
		for X, _ in Centres:
			cv2.circle(Image, (Px(X, Density), Px(Y, Density)), Radius, Ring, Line, cv2.LINE_AA)

	Write(Image, "HOLD THE SHEET HERE", Layout.PageWidth / 2, 134.5, 2.6, Density, Soft, "Centre")
	Footer = Form.Name if Form.Name else "bubble sheets read by a webcam with OpenCV"
	Write(Image, f"glance  /  {Footer}", Layout.PageWidth / 2, 139.5, 2.0, Density, 160, "Centre")
	return Image


def DrawMark(Image: np.ndarray, Centre: Layout.Point, Density: float, Style: str, Random: np.random.Generator, Size: float = Layout.BubbleRadius) -> None:
	X, Y = Px(Centre[0], Density), Px(Centre[1], Density)
	Radius = Size * Density

	if Style == "Solid":
		cv2.circle(Image, (X, Y), int(Radius * Random.uniform(0.86, 1.0)), int(Random.integers(25, 60)), -1, cv2.LINE_AA)
	elif Style == "Pencil":
		Span = int(Radius * 2.4)
		Patch = np.zeros((Span, Span), dtype=np.uint8)
		cv2.circle(Patch, (Span // 2, Span // 2), int(Radius * Random.uniform(0.86, 0.98)), 255, -1, cv2.LINE_AA)
		Grain = Random.normal(Random.uniform(95, 135), 14, Patch.shape)
		Top, Left = Y - Span // 2, X - Span // 2
		Region = Image[Top:Top + Span, Left:Left + Span].astype(np.float32)
		Alpha = Patch.astype(np.float32) / 255.0
		Image[Top:Top + Span, Left:Left + Span] = np.clip(Region * (1 - Alpha) + Grain * Alpha, 0, 255).astype(np.uint8)
	elif Style == "Scribble":
		Thickness = max(1, int(Density * 0.55))
		for _ in range(int(Random.integers(11, 16))):
			AngleA, AngleB = Random.uniform(0, 2 * np.pi, 2)
			ReachA, ReachB = Random.uniform(0.3, 0.92, 2) * Radius
			Start = (int(X + np.cos(AngleA) * ReachA), int(Y + np.sin(AngleA) * ReachA))
			End = (int(X + np.cos(AngleB) * ReachB), int(Y + np.sin(AngleB) * ReachB))
			cv2.line(Image, Start, End, int(Random.integers(30, 70)), Thickness, cv2.LINE_AA)
	elif Style == "Partial":
		Shift = Random.uniform(-0.22, 0.22, 2) * Radius
		cv2.circle(Image, (int(X + Shift[0]), int(Y + Shift[1])), int(Radius * Random.uniform(0.7, 0.8)), int(Random.integers(30, 70)), -1, cv2.LINE_AA)
	elif Style == "Dot":
		cv2.circle(Image, (X, Y), max(1, int(Radius * 0.2)), 50, -1, cv2.LINE_AA)
	elif Style == "Erased":
		cv2.circle(Image, (X, Y), int(Radius * 0.9), int(Random.integers(205, 226)), -1, cv2.LINE_AA)


def BubblesFor(Form: Layout.Form, Row: int, Answer: str) -> list[int]:
	if Form.Mode == "Choice":
		return [Layout.Options.index(Letter) for Letter in Answer if Letter in Layout.Options]
	if Answer in ("-", "*", ""):
		return []
	Value = float(Answer)
	Whole = int(Value)
	Bubbles = [Whole] if Whole > 0 or Value == 0 else []
	if Value != Whole:
		Bubbles.append(len(Form.Rows[Row]) - 1)
	return Bubbles


def FillSheet(Image: np.ndarray, Form: Layout.Form, Answers: list[str], Roll: str, Density: float, Random: np.random.Generator, Style: str | None = None) -> None:
	for Digit, Value in enumerate(Roll):
		DrawMark(Image, Layout.RollBubble(Digit, int(Value)), Density, Style or str(Random.choice(MarkStyles)), Random)
	for Row, Answer in enumerate(Answers):
		for Bubble in BubblesFor(Form, Row, Answer):
			DrawMark(Image, Form.Rows[Row][Bubble], Density, Style or str(Random.choice(MarkStyles)), Random, Form.Radius)


def RandomKey(Random: np.random.Generator) -> str:
	return "".join(Random.choice(list(Layout.Options), Layout.QuestionCount))


def StudentAnswers(Key: str, Skill: float, Hardness: np.ndarray, Random: np.random.Generator) -> list[str]:
	Answers: list[str] = []
	for Question, Right in enumerate(Key):
		Roll = Random.random()
		if Roll < 0.03:
			Answers.append("-")
		elif Roll < 0.03 + Skill * (1.0 - Hardness[Question]):
			Answers.append(Right)
		else:
			Answers.append(str(Random.choice([Letter for Letter in Layout.Options if Letter != Right])))
	return Answers


def StudentMarks(Form: Layout.Form, Skill: float, Strength: dict[str, float], Random: np.random.Generator) -> list[str]:
	Marks: list[str] = []
	for Row, Highest in enumerate(Form.Maxes):
		Share = float(np.clip(Skill * Strength[Form.Topics[Row]] + Random.normal(0, 0.15), 0, 1))
		Step = 0.5 if Form.Half else 1.0
		Marks.append(f"{round(Share * Highest / Step) * Step:g}")
	return Marks


def TwoUp(Sheets: list[np.ndarray], Density: float) -> np.ndarray:
	Page = np.full((Px(297.0, Density), Px(210.0, Density)), 255, dtype=np.uint8)
	for Index, Sheet in enumerate(Sheets[:2]):
		Top = Index * Px(148.5, Density)
		Page[Top:Top + Sheet.shape[0], :Sheet.shape[1]] = Sheet
	Middle = Px(148.5, Density)
	for Left in range(0, Page.shape[1], Px(4.0, Density)):
		cv2.line(Page, (Left, Middle), (Left + Px(1.5, Density), Middle), 170, 1)
	return Page


def LoadJson(FilePath: Path) -> dict:
	if FilePath.exists():
		return json.loads(FilePath.read_text())
	return {}


# -- INIT --
def Main() -> None:
	Parser = argparse.ArgumentParser(description="Make printable Glance sheets")
	Parser.add_argument("--test", type=int, default=1)
	Parser.add_argument("--samples", type=int, default=0)
	Parser.add_argument("--seed", type=int, default=263)
	Args = Parser.parse_args()

	if not 1 <= Args.test <= Layout.MaxTestId:
		raise SystemExit(f"Test id must be 1 to {Layout.MaxTestId}")

	PrintFolder.mkdir(exist_ok=True)
	Density = PrintDpi / 25.4
	Random = np.random.default_rng(Args.seed + Args.test)
	Form = Layout.FormFor(Args.test, Layout.LoadPapers())
	Kind = "Marks" if Form.Mode == "Marks" else "Answer"
	Blank = DrawSheet(Args.test, Density, Form)
	cv2.imwrite(str(PrintFolder / f"{Kind}_Blank_Test{Args.test:02d}_A4.png"), TwoUp([Blank, Blank], Density))

	if Args.samples > 0:
		Truth = LoadJson(TruthPath)
		Sheets: list[np.ndarray] = []
		if Form.Mode == "Marks":
			Strength = {Topic: float(Random.uniform(0.45, 1.05)) for Topic in sorted(set(Form.Topics))}
		else:
			Keys = LoadJson(KeysPath)
			Entry = Keys.get(str(Args.test))
			if Entry is None:
				Entry = {"Name": f"Test {Args.test:02d}", "Key": RandomKey(Random)}
				Keys[str(Args.test)] = Entry
				KeysPath.write_text(json.dumps(Keys, indent="\t"))
			Hardness = Random.uniform(0.0, 0.4, Layout.QuestionCount)

		for Index in range(Args.samples):
			Roll = f"{101 + Index:03d}"
			Skill = float(Random.uniform(0.62, 0.99))
			Answers = StudentMarks(Form, Skill, Strength, Random) if Form.Mode == "Marks" else StudentAnswers(Entry["Key"], Skill, Hardness, Random)
			Sheet = Blank.copy()
			FillSheet(Sheet, Form, Answers, Roll, Density, Random)
			Truth[f"{Args.test}-{Roll}"] = "|".join(Answers)
			Sheets.append(Sheet)

		for Index in range(0, len(Sheets), 2):
			Pair = Sheets[Index:Index + 2]
			cv2.imwrite(str(PrintFolder / f"{Kind}_Sample_Test{Args.test:02d}_Page{Index // 2 + 1:02d}_A4.png"), TwoUp(Pair, Density))
		TruthPath.write_text(json.dumps(Truth, indent="\t"))

	print(f"[MAKESHEETS] {Kind} sheet for test {Args.test:02d} written to {PrintFolder}")


if __name__ == "__main__":
	Main()
