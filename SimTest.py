# -- IMPORTS --
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

import Layout
import MakeSheets
import Publish
import Vision

# -- DATA --
Root = Path(__file__).resolve().parent
ChoicePath = Root / "SimResults.json"
MarksPath = Root / "SimMarks.json"
FrameWidth: int = 1280
FrameHeight: int = 720
SheetDensity: float = 8.0
Focal: float = 1050.0


# -- FUNCTIONS --
def RandomChoices(Random: np.random.Generator) -> tuple[list[str], list[list[int]], list[tuple[int, int, str]]]:
	Truth: list[str] = []
	Drawn: list[list[int]] = []
	Traps: list[tuple[int, int, str]] = []
	for Question in range(Layout.QuestionCount):
		Roll = Random.random()
		if Roll < 0.08:
			Truth.append(Vision.Blank)
			Drawn.append([])
		elif Roll < 0.14:
			Pair = Random.choice(list(Layout.Options), 2, replace=False)
			Truth.append(Vision.Multiple)
			Drawn.append([Layout.Options.index(Letter) for Letter in Pair])
		else:
			Letter = str(Random.choice(list(Layout.Options)))
			Truth.append(Letter)
			Drawn.append([Layout.Options.index(Letter)])
		if Random.random() < 0.12:
			Free = [Option for Option in range(Layout.OptionCount) if Option not in Drawn[-1]]
			Traps.append((Question, int(Random.choice(Free)), str(Random.choice(["Dot", "Erased"]))))
	return Truth, Drawn, Traps


def RandomMarks(Form: Layout.Form, Random: np.random.Generator) -> tuple[list[str], list[list[int]], list[tuple[int, int, str]]]:
	Truth: list[str] = []
	Drawn: list[list[int]] = []
	Traps: list[tuple[int, int, str]] = []
	for Row, Highest in enumerate(Form.Maxes):
		Last = len(Form.Rows[Row]) - 1
		Roll = Random.random()
		if Roll < 0.08:
			Truth.append(Vision.Blank)
			Drawn.append([])
		elif Roll < 0.14 and Highest >= 1:
			Pair = Random.choice(Highest + 1, 2, replace=False)
			Truth.append(Vision.Multiple)
			Drawn.append([int(Value) for Value in Pair])
		else:
			Whole = int(Random.integers(0, Highest + 1))
			Extra = Form.Half and Whole < Highest and Random.random() < 0.3
			Truth.append(f"{Whole + (0.5 if Extra else 0):g}")
			Bubbles = [Whole] if Whole > 0 or not Extra or Random.random() < 0.5 else []
			Drawn.append(Bubbles + ([Last] if Extra else []))
		if Random.random() < 0.12:
			Free = [Bubble for Bubble in range(Last + 1) if Bubble not in Drawn[-1]]
			if Free:
				Traps.append((Row, int(Random.choice(Free)), str(Random.choice(["Dot", "Erased"]))))
	return Truth, Drawn, Traps


def WorstMarksForm() -> Layout.Form:
	Questions = [{"Label": str(Index + 1), "Max": 10, "Topic": ""} for Index in range(20)]
	return Layout.MarksForm("stress test", Questions, True)


def BuildSheet(TestId: int, Random: np.random.Generator, Style: str | None, Form: Layout.Form) -> tuple[np.ndarray, list[str], str, list[list[int]]]:
	Sheet = MakeSheets.DrawSheet(TestId, SheetDensity, Form)
	Truth, Drawn, Traps = RandomMarks(Form, Random) if Form.Mode == "Marks" else RandomChoices(Random)
	Roll = f"{int(Random.integers(0, 1000)):03d}"
	for Digit, Value in enumerate(Roll):
		MakeSheets.DrawMark(Sheet, Layout.RollBubble(Digit, int(Value)), SheetDensity, Style or str(Random.choice(MakeSheets.MarkStyles)), Random)
	for Row, Bubbles in enumerate(Drawn):
		for Bubble in Bubbles:
			MakeSheets.DrawMark(Sheet, Form.Rows[Row][Bubble], SheetDensity, Style or str(Random.choice(MakeSheets.MarkStyles)), Random, Form.Radius)
	for Row, Bubble, Trap in Traps:
		MakeSheets.DrawMark(Sheet, Form.Rows[Row][Bubble], SheetDensity, Trap, Random, Form.Radius)
	return Sheet, Truth, Roll, Drawn


def Curl(Sheet: np.ndarray, Amount: float) -> np.ndarray:
	if Amount <= 0:
		return Sheet
	Height, Width = Sheet.shape
	Xs, Ys = np.meshgrid(np.arange(Width, dtype=np.float32), np.arange(Height, dtype=np.float32))
	Shift = Amount * SheetDensity * np.sin(np.pi * Xs / Width)
	return cv2.remap(Sheet, Xs, Ys + Shift.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def Backdrop(Random: np.random.Generator) -> np.ndarray:
	Base = Random.uniform(40, 200, 3)
	Ramp = np.linspace(0.7, 1.15, FrameWidth, dtype=np.float32)[None, :, None]
	Frame = np.clip(np.ones((FrameHeight, FrameWidth, 3), dtype=np.float32) * Base * Ramp, 0, 255).astype(np.uint8)
	for _ in range(int(Random.integers(6, 16))):
		X, Y = int(Random.integers(0, FrameWidth)), int(Random.integers(0, FrameHeight))
		W, H = int(Random.integers(40, 400)), int(Random.integers(40, 300))
		Colour = tuple(int(V) for V in Random.integers(0, 256, 3))
		if Random.random() < 0.5:
			cv2.rectangle(Frame, (X, Y), (X + W, Y + H), Colour, -1)
		else:
			cv2.circle(Frame, (X, Y), W // 3, Colour, -1)
	return cv2.GaussianBlur(Frame, (0, 0), float(Random.uniform(1.5, 6)))


def Project(Tilt: float, Random: np.random.Generator) -> tuple[np.ndarray, float]:
	Half = np.array([Layout.PageWidth / 2, Layout.PageHeight / 2])
	Corners = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=np.float64) * np.array([Half[0], Half[1], 1])

	Direction = Random.uniform(0, 2 * np.pi)
	Yaw = np.radians(Tilt) * np.cos(Direction)
	Pitch = np.radians(Tilt) * np.sin(Direction)
	Spin = np.radians(Random.uniform(-18, 18) + (180 if Random.random() < 0.15 else 0))

	RotationX = np.array([[1, 0, 0], [0, np.cos(Pitch), -np.sin(Pitch)], [0, np.sin(Pitch), np.cos(Pitch)]])
	RotationY = np.array([[np.cos(Yaw), 0, np.sin(Yaw)], [0, 1, 0], [-np.sin(Yaw), 0, np.cos(Yaw)]])
	RotationZ = np.array([[np.cos(Spin), -np.sin(Spin), 0], [np.sin(Spin), np.cos(Spin), 0], [0, 0, 1]])
	Turned = Corners @ (RotationZ @ RotationY @ RotationX).T

	Fill = Random.uniform(0.5, 0.85)
	Distance = Focal * Layout.PageWidth / (Fill * FrameWidth)
	Depth = Turned[:, 2] + Distance
	Points = np.stack([Focal * Turned[:, 0] / Depth, Focal * Turned[:, 1] / Depth], axis=1)

	Low, High = Points.min(axis=0), Points.max(axis=0)
	Room = np.array([FrameWidth, FrameHeight]) - (High - Low) - 16
	if Room.min() < 0:
		Points *= float(np.min((np.array([FrameWidth, FrameHeight]) - 16) / (High - Low)))
		Low, High = Points.min(axis=0), Points.max(axis=0)
		Room = np.array([FrameWidth, FrameHeight]) - (High - Low) - 16
	Offset = 8 + Random.uniform(0, 1, 2) * np.maximum(Room, 0) - Low
	Quad = (Points + Offset).astype(np.float32)
	Span = float(np.linalg.norm(Quad[1] - Quad[0]))
	return Quad, Span


def Shoot(Sheet: np.ndarray, Random: np.random.Generator, Tilt: float, Blur: float) -> tuple[np.ndarray, dict[str, float]]:
	Bend = float(Random.uniform(0.0, 1.2))
	Paper = Curl(Sheet, Bend)
	Height, Width = Paper.shape
	Quad, Span = Project(Tilt, Random)
	Source = np.array([[0, 0], [Width, 0], [Width, Height], [0, Height]], dtype=np.float32)
	Matrix = cv2.getPerspectiveTransform(Source, Quad)

	Soft = cv2.GaussianBlur(Paper, (0, 0), max(0.6, Width / max(Span, 1.0) * 0.45))
	Colour = cv2.cvtColor(Soft, cv2.COLOR_GRAY2BGR)
	Frame = Backdrop(Random)
	Layer = cv2.warpPerspective(Colour, Matrix, (FrameWidth, FrameHeight), flags=cv2.INTER_LINEAR)
	Mask = cv2.warpPerspective(np.full((Height, Width), 255, dtype=np.uint8), Matrix, (FrameWidth, FrameHeight))
	Frame[Mask > 127] = Layer[Mask > 127]

	Xs, Ys = np.meshgrid(np.linspace(-1, 1, FrameWidth, dtype=np.float32), np.linspace(-1, 1, FrameHeight, dtype=np.float32))
	Angle = Random.uniform(0, 2 * np.pi)
	Shadow = float(Random.uniform(0.0, 0.45))
	Light = 1.0 - Shadow * (0.5 + 0.5 * (Xs * np.cos(Angle) + Ys * np.sin(Angle)))
	Spot = (float(Random.uniform(-1, 1)), float(Random.uniform(-1, 1)))
	Light *= 1.0 - float(Random.uniform(0.0, 0.3)) * np.exp(-((Xs - Spot[0]) ** 2 + (Ys - Spot[1]) ** 2) / 0.18)
	Gain = float(Random.uniform(0.6, 1.1))
	Shot = Frame.astype(np.float32) * Light[..., None] * Gain

	if Blur > 0:
		Shot = cv2.GaussianBlur(Shot, (0, 0), Blur)
	Smear = 0
	if Random.random() < 0.2:
		Smear = int(Random.integers(3, 9))
		Kernel = np.zeros((Smear, Smear), dtype=np.float32)
		Kernel[Smear // 2, :] = 1.0 / Smear
		Turn = cv2.getRotationMatrix2D((Smear / 2 - 0.5, Smear / 2 - 0.5), float(Random.uniform(0, 180)), 1.0)
		Kernel = cv2.warpAffine(Kernel, Turn, (Smear, Smear))
		Shot = cv2.filter2D(Shot, -1, Kernel / max(float(Kernel.sum()), 1e-6))

	Shot += Random.normal(0, float(Random.uniform(1.5, 7.0)), Shot.shape)
	Shot = np.clip(Shot, 0, 255).astype(np.uint8)
	_, Packed = cv2.imencode(".jpg", Shot, [cv2.IMWRITE_JPEG_QUALITY, int(Random.integers(55, 96))])
	Shot = cv2.imdecode(Packed, cv2.IMREAD_COLOR)
	return Shot, {"Tilt": Tilt, "Blur": Blur, "Bend": Bend, "Span": Span, "Shadow": Shadow, "Smear": float(Smear)}


def Bucket(Value: float, Edges: list[float], Names: list[str]) -> str:
	for Edge, Name in zip(Edges, Names):
		if Value < Edge:
			return Name
	return Names[-1]


def Tally(Table: dict[str, dict[str, int]], Name: str, Found: bool, Right: int, Fields: int, Perfect: bool) -> None:
	Row = Table.setdefault(Name, {"Shots": 0, "Found": 0, "Right": 0, "Fields": 0, "Perfect": 0})
	Row["Shots"] += 1
	Row["Found"] += int(Found)
	Row["Right"] += Right
	Row["Fields"] += Fields
	Row["Perfect"] += int(Perfect)


# -- INIT --
def Main() -> None:
	Parser = argparse.ArgumentParser(description="Stress test Glance on simulated camera shots")
	Parser.add_argument("--shots", type=int, default=600)
	Parser.add_argument("--seed", type=int, default=2026)
	Parser.add_argument("--save", type=int, default=0)
	Parser.add_argument("--dry", action="store_true")
	Parser.add_argument("--mode", choices=["choice", "marks"], default="choice")
	Args = Parser.parse_args()

	Random = np.random.default_rng(Args.seed)
	Form = WorstMarksForm() if Args.mode == "marks" else Layout.ChoiceForm()
	Fields = len(Form.Rows) + Layout.RollDigits
	Tables: dict[str, dict[str, dict[str, int]]] = {"All": {}, "Tilt": {}, "Blur": {}, "Marks": {}, "Size": {}}
	MarkedScores: list[float] = []
	EmptyScores: list[float] = []
	Times: list[float] = []
	Misses: list[dict[str, float]] = []
	SaveFolder = Root / "SimShots"
	if Args.save:
		SaveFolder.mkdir(exist_ok=True)

	for Index in range(Args.shots):
		TestId = int(Random.integers(1, Layout.MaxTestId + 1))
		Style = None if Random.random() < 0.4 else str(Random.choice(MakeSheets.MarkStyles))
		Sheet, Truth, Roll, Drawn = BuildSheet(TestId, Random, Style, Form)
		Tilt = float(Random.uniform(0, 40))
		Blur = float(Random.uniform(0, 1.8))
		Shot, Info = Shoot(Sheet, Random, Tilt, Blur)
		if Index < Args.save:
			cv2.imwrite(str(SaveFolder / f"Shot{Index:03d}_T{TestId:02d}_R{Roll}.jpg"), Shot)

		Start = time.perf_counter()
		Result = Vision.ReadSheet(Shot, lambda _: Form)
		Times.append(time.perf_counter() - Start)

		Found = Result is not None and Result[0].TestId == TestId
		Right = 0
		if Found:
			_, _, Reading = Result
			Right = sum(A == B for A, B in zip(Reading.Answers, Truth)) + sum(A == B for A, B in zip(Reading.Roll, Roll))
			for Row, Answer in enumerate(Truth):
				Line = Vision.RowScores(Form, Reading.Scores, Row)
				for Bubble, Score in enumerate(Line):
					if Bubble in Drawn[Row]:
						if Answer != Vision.Multiple:
							MarkedScores.append(float(Score))
					elif Answer != Vision.Multiple:
						EmptyScores.append(float(Score))
		Perfect = Found and Right == Fields
		if Found and not Perfect:
			Misses.append({**Info, "Wrong": float(Fields - Right)})

		Tally(Tables["All"], "All shots", Found, Right, Fields, Perfect)
		Tally(Tables["Tilt"], Bucket(Tilt, [15, 30], ["0 to 15 deg", "15 to 30 deg", "30 to 40 deg"]), Found, Right, Fields, Perfect)
		Tally(Tables["Blur"], Bucket(Blur, [0.6, 1.2], ["Sharp", "Soft", "Blurry"]), Found, Right, Fields, Perfect)
		Tally(Tables["Marks"], Style or "Mixed", Found, Right, Fields, Perfect)
		Tally(Tables["Size"], Bucket(Info["Span"], [700, 900], ["Small in frame", "Medium in frame", "Large in frame"]), Found, Right, Fields, Perfect)

	Whole = Tables["All"]["All shots"]
	Summary = {
		"Shots": Args.shots,
		"Seed": Args.seed,
		"Mode": Args.mode,
		"Rows": len(Form.Rows),
		"BubblesPerSheet": sum(len(Row) for Row in Form.Rows) + 10 * Layout.RollDigits,
		"BubbleRadius": Form.Radius,
		"FieldsPerSheet": Fields,
		"SheetsFound": Whole["Found"],
		"FieldsRead": Whole["Found"] * (Fields),
		"FieldsRight": Whole["Right"],
		"PerfectSheets": Whole["Perfect"],
		"MedianMilliseconds": round(float(np.median(Times)) * 1000, 1),
		"Tables": Tables,
		"MarkedHistogram": np.histogram(MarkedScores, bins=50, range=(0, 1))[0].tolist(),
		"EmptyHistogram": np.histogram(EmptyScores, bins=50, range=(0, 1))[0].tolist(),
		"Threshold": Vision.MarkThreshold,
		"OpenCv": cv2.__version__,
	}
	if not Args.dry:
		(MarksPath if Args.mode == "marks" else ChoicePath).write_text(json.dumps(Summary, indent="\t"))

	Read = Whole["Found"] * (Fields)
	print(f"[SIMTEST] {Args.mode} sheet | {Args.shots} shots | found {Whole['Found']} | fields right {Whole['Right']}/{Read} ({100 * Whole['Right'] / max(Read, 1):.2f}%) | perfect sheets {Whole['Perfect']} | median {Summary['MedianMilliseconds']} ms")
	for Name, Table in Tables.items():
		for Label, Row in Table.items():
			Seen = Row["Found"] * (Fields)
			print(f"  {Name:6s} {Label:18s} shots {Row['Shots']:4d} found {Row['Found']:4d} right {100 * Row['Right'] / max(Seen, 1):6.2f}% perfect {Row['Perfect']:4d}")
	if not Args.dry:
		Publish.Sync()
	if Misses:
		print(f"  misses: {len(Misses)} sheets, mean tilt {np.mean([M['Tilt'] for M in Misses]):.1f}, mean blur {np.mean([M['Blur'] for M in Misses]):.2f}, mean bend {np.mean([M['Bend'] for M in Misses]):.2f}, mean span {np.mean([M['Span'] for M in Misses]):.0f}")


if __name__ == "__main__":
	Main()
