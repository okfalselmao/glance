import time

InitTime = time.perf_counter()
Updated = "21/11/2026"

# -- IMPORTS --
import argparse
import csv
import json
import sys
import threading
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import Layout
import Publish
import Vision

# -- PATHS --
Root = Path(__file__).resolve().parent
KeysPath = Root / "Keys.json"
ResultsPath = Root / "Results.csv"
TruthPath = Root / "Truth.json"
LiveTestPath = Root / "LiveTest.csv"
LiveSummaryPath = Root / "LiveTestSummary.json"

# -- DATA --
WindowName = "Glance"
CanvasWidth: int = 1280
CanvasHeight: int = 720
ViewWidth: int = 960
ViewHeight: int = 540
PanelLeft: int = 960
LockFrames: int = 6
AverageFrames: int = 5
LostFrames: int = 10
ShieldBlock: int = 22
Joiner: str = "|"

Ink = (15, 11, 11)
Panel = (28, 21, 21)
Line = (56, 43, 43)
White = (248, 244, 244)
Mute = (184, 169, 169)
Pink = (154, 46, 255)
Blue = (255, 123, 108)
Gold = (61, 200, 255)
Green = (128, 222, 74)
Red = (84, 84, 255)

StatusColours: dict[str, tuple[int, int, int]] = {
	"Correct": Green,
	"Wrong": Red,
	"Blank": Mute,
	"Multiple": Gold,
	"Read": Pink,
}

Font = cv2.FONT_HERSHEY_DUPLEX
Colour = tuple[int, int, int]
KeyBook = dict[str, dict[str, str]]


@dataclass
class Insight:
	Count: int
	Average: float
	Shares: list[float]
	Traps: list[str]
	Topics: list[tuple[str, float]]


@dataclass
class Shown:
	TestId: int
	Form: Layout.Form
	Roll: str
	Answers: list[str]
	Score: float | None
	Out: float
	Status: list[str]
	Note: str
	NoteColour: Colour


# -- CLASSES --
class Forms:
	def __init__(self) -> None:
		self.Papers = Layout.LoadPapers()
		self.Cache: dict[int, Layout.Form] = {}

	def Get(self, TestId: int) -> Layout.Form:
		if TestId not in self.Cache:
			self.Cache[TestId] = Layout.FormFor(TestId, self.Papers)
		return self.Cache[TestId]


class ResultStore:
	def __init__(self, FilePath: Path) -> None:
		self.FilePath = FilePath
		self.Rows: dict[tuple[int, str], list[str]] = {}
		if FilePath.exists():
			with FilePath.open(newline="") as File:
				for Row in csv.DictReader(File):
					self.Rows[(int(Row["TestId"]), Row["Roll"])] = Row["Answers"].split(Joiner)

	def Put(self, TestId: int, Roll: str, Answers: list[str]) -> None:
		self.Rows[(TestId, Roll)] = list(Answers)

	def Remove(self, TestId: int, Roll: str) -> None:
		self.Rows.pop((TestId, Roll), None)

	def ForTest(self, TestId: int) -> dict[str, list[str]]:
		return {Roll: Answers for (Id, Roll), Answers in self.Rows.items() if Id == TestId}

	def Save(self, Keys: KeyBook, Book: Forms) -> None:
		with self.FilePath.open("w", newline="") as File:
			Writer = csv.writer(File)
			Writer.writerow(["TestId", "Roll", "Answers", "Score", "Total"])
			for (TestId, Roll), Answers in sorted(self.Rows.items()):
				Score, Out = Mark(Book.Get(TestId), Answers, Keys.get(str(TestId)))
				Writer.writerow([TestId, Roll, Joiner.join(Answers), "" if Score is None else f"{Score:g}", f"{Out:g}"])

		for TestId in sorted({Id for Id, _ in self.Rows}):
			Form = Book.Get(TestId)
			if Form.Mode != "Marks":
				continue
			with (self.FilePath.parent / f"Marks_Test{TestId:02d}.csv").open("w", newline="") as File:
				Writer = csv.writer(File)
				Writer.writerow(["Roll", *[f"Q{Label}" for Label in Form.Labels], "Total", "Out of"])
				for Roll, Answers in sorted(self.ForTest(TestId).items()):
					Writer.writerow([Roll, *Answers, f"{Vision.Total(Answers):g}", sum(Form.Maxes)])


class Tracker:
	def __init__(self) -> None:
		self.History: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=AverageFrames)
		self.TestId: int | None = None
		self.Signature: str = ""
		self.Stable: int = 0
		self.Missing: int = 0
		self.Handled: bool = False

	def Reset(self) -> None:
		self.History.clear()
		self.TestId = None
		self.Signature = ""
		self.Stable = 0
		self.Handled = False

	def Lost(self) -> None:
		self.Missing += 1
		if self.Missing > LostFrames:
			self.Reset()

	def Update(self, TestId: int, Form: Layout.Form, Scores: np.ndarray, RollScores: np.ndarray) -> Vision.Reading:
		self.Missing = 0
		if TestId != self.TestId:
			self.Reset()
			self.TestId = TestId
		self.History.append((Scores, RollScores))
		MeanScores = np.mean([Item[0] for Item in self.History], axis=0)
		MeanRoll = np.mean([Item[1] for Item in self.History], axis=0)
		Reading = Vision.Decide(Form, MeanScores, MeanRoll)
		Signature = Reading.Roll + Joiner.join(Reading.Answers)
		if Signature == self.Signature:
			self.Stable += 1
		else:
			self.Signature = Signature
			self.Stable = 0
			self.Handled = False
		return Reading

	def Locked(self) -> bool:
		return self.Stable >= LockFrames


class LiveTest:
	def __init__(self, Fresh: bool) -> None:
		self.Truth: dict[str, str] = json.loads(TruthPath.read_text()) if TruthPath.exists() else {}
		self.Sheets: int = 0
		self.Right: int = 0
		self.Fields: int = 0
		self.Perfect: int = 0
		self.Unmatched: int = 0
		self.Rows: list[list[str]] = []
		if LiveTestPath.exists() and not Fresh:
			with LiveTestPath.open(newline="") as File:
				for Row in list(csv.reader(File))[1:]:
					self.Rows.append(Row)
					if Row[3] == "":
						self.Unmatched += 1
						continue
					self.Sheets += 1
					self.Right += int(Row[3])
					self.Fields += int(Row[4])
					self.Perfect += int(Row[3] == Row[4])

	def Record(self, TestId: int, Roll: str, Answers: list[str]) -> str:
		Stored = self.Truth.get(f"{TestId}-{Roll}")
		Read = Joiner.join(Answers)
		if Stored is None:
			self.Unmatched += 1
			self.Rows.append([time.strftime("%H:%M:%S"), str(TestId), Roll, "", "", Read, ""])
			return "NOT IN TRUTH.JSON"
		Expected = Stored.split(Joiner) if Joiner in Stored else list(Stored)
		Right = sum(A == B for A, B in zip(Answers, Expected))
		self.Sheets += 1
		self.Right += Right
		self.Fields += len(Expected)
		self.Perfect += int(Right == len(Expected))
		self.Rows.append([time.strftime("%H:%M:%S"), str(TestId), Roll, str(Right), str(len(Expected)), Read, Stored])
		return f"TEST {Right}/{len(Expected)} RIGHT"

	def Line(self) -> str:
		return f"{self.Right}/{self.Fields} rows right, {self.Sheets} reads"

	def Finish(self) -> None:
		with LiveTestPath.open("w", newline="") as File:
			Writer = csv.writer(File)
			Writer.writerow(["Time", "TestId", "Roll", "Right", "Fields", "Read", "Truth"])
			Writer.writerows(self.Rows)
		Summary = {
			"Date": time.strftime("%d/%m/%Y"),
			"Reads": self.Sheets,
			"AnswersRight": self.Right,
			"AnswersRead": self.Fields,
			"PerfectReads": self.Perfect,
			"Unmatched": self.Unmatched,
		}
		LiveSummaryPath.write_text(json.dumps(Summary, indent="\t"))
		print(f"[LIVETEST] {self.Right}/{self.Fields} rows right in {self.Sheets} reads | {self.Perfect} perfect | {self.Unmatched} unmatched")
		Publish.Sync()


# -- FUNCTIONS --
def LoadKeys() -> KeyBook:
	if KeysPath.exists():
		return json.loads(KeysPath.read_text())
	return {}


def Beep(Pitch: int) -> None:
	try:
		import winsound
		threading.Thread(target=winsound.Beep, args=(Pitch, 110), daemon=True).start()
	except ImportError:
		print("\a", end="", flush=True)


def Label(Canvas: np.ndarray, Text: str, X: int, Y: int, Size: float, Shade: Colour, Thick: int = 1, Align: str = "Left") -> None:
	if Align != "Left":
		(Width, _), _ = cv2.getTextSize(Text, Font, Size, Thick)
		X -= Width if Align == "Right" else Width // 2
	cv2.putText(Canvas, Text, (X, Y), Font, Size, Shade, Thick, cv2.LINE_AA)


def Mark(Form: Layout.Form, Answers: list[str], Entry: dict[str, str] | None) -> tuple[float | None, float]:
	if Form.Mode == "Marks":
		return Vision.Total(Answers), float(sum(Form.Maxes))
	if Entry is None:
		return None, float(len(Form.Rows))
	return float(Vision.Grade(Answers, Entry["Key"])[0]), float(len(Form.Rows))


def ClassInsight(Form: Layout.Form, Rows: dict[str, list[str]], Entry: dict[str, str] | None) -> Insight | None:
	Count = len(Rows)
	if Form.Mode == "Choice" and Entry is None:
		return None
	Shares: list[float] = []
	Traps: list[str] = []
	Earned: dict[str, float] = {}
	Possible: dict[str, float] = {}
	Sum = 0.0
	for Row in range(len(Form.Rows)):
		Picks = [Answers[Row] for Answers in Rows.values() if Row < len(Answers)]
		if Form.Mode == "Marks":
			Got = sum(Vision.MarkValue(Pick) for Pick in Picks)
			Shares.append(Got / (Count * Form.Maxes[Row]) if Count else 0.0)
			Traps.append("")
			Topic = Form.Topics[Row]
			Earned[Topic] = Earned.get(Topic, 0.0) + Got
			Possible[Topic] = Possible.get(Topic, 0.0) + Count * Form.Maxes[Row]
		else:
			Right = Entry["Key"][Row]
			Got = float(sum(Pick == Right for Pick in Picks))
			Shares.append(Got / Count if Count else 0.0)
			Wrong = Counter(Pick for Pick in Picks if Pick in Layout.Options and Pick != Right)
			Traps.append(Wrong.most_common(1)[0][0] if Wrong else "")
		Sum += Got
	Topics = [(Topic, Earned[Topic] / Possible[Topic] if Possible[Topic] else 0.0) for Topic in Earned if Topic]
	return Insight(Count, Sum / Count if Count else 0.0, Shares, Traps, Topics)


def Shield(Frame: np.ndarray, Quad: np.ndarray | None) -> np.ndarray:
	Height, Width = Frame.shape[:2]
	Small = cv2.resize(Frame, (max(Width // ShieldBlock, 1), max(Height // ShieldBlock, 1)), interpolation=cv2.INTER_AREA)
	Hidden = cv2.resize(Small, (Width, Height), interpolation=cv2.INTER_NEAREST)
	Hidden = cv2.convertScaleAbs(Hidden, alpha=0.5)
	if Quad is None:
		return Hidden
	Mask = np.zeros((Height, Width), dtype=np.uint8)
	cv2.fillConvexPoly(Mask, np.rint(Quad).astype(np.int32), 255)
	cv2.copyTo(Frame, Mask, Hidden)
	cv2.polylines(Hidden, [np.rint(Quad).astype(np.int32)], True, Pink, max(2, Width // 480), cv2.LINE_AA)
	return Hidden


def DrawMarks(View: np.ndarray, Sheet: Vision.Located, Current: Shown, Reading: Vision.Reading, Key: str | None) -> None:
	Form = Current.Form
	Points = Vision.FormPoints(Form)
	Centres = Vision.ToFrame(Points, Sheet.Inverse)
	Edge = Points.copy()
	Edge[:, 0] += Form.Radius * Layout.Scale
	Edges = Vision.ToFrame(Edge, Sheet.Inverse)
	Thick = max(2, View.shape[1] // 420)
	Starts = Vision.RowStarts(Form)

	for Row in range(len(Form.Rows)):
		Shade = StatusColours[Current.Status[Row]]
		for Bubble in range(Starts[Row + 1] - Starts[Row]):
			Index = Starts[Row] + Bubble
			Centre = Centres[Index]
			Radius = int(round(float(np.linalg.norm(Edges[Index] - Centre)) * 1.25))
			Spot = (int(round(Centre[0])), int(round(Centre[1])))
			if Reading.Scores[Index] >= Vision.MarkThreshold:
				cv2.circle(View, Spot, Radius, Shade, Thick, cv2.LINE_AA)
			elif Key is not None and Key[Row] == Layout.Options[Bubble] and Current.Status[Row] != "Correct":
				cv2.circle(View, Spot, Radius, Blue, max(1, Thick - 1), cv2.LINE_AA)

	RollCentres = Vision.ToFrame(Vision.RollPoints, Sheet.Inverse)
	RollEdge = Vision.RollPoints.copy()
	RollEdge[..., 0] += Layout.BubbleRadius * Layout.Scale
	RollEdges = Vision.ToFrame(RollEdge, Sheet.Inverse)
	for Digit, Value in enumerate(Reading.Roll):
		if Value == Vision.Unknown:
			continue
		Centre = RollCentres[int(Value), Digit]
		Radius = int(round(float(np.linalg.norm(RollEdges[int(Value), Digit] - Centre)) * 1.25))
		cv2.circle(View, (int(round(Centre[0])), int(round(Centre[1]))), Radius, Pink, Thick, cv2.LINE_AA)


def DebugView(Flat: np.ndarray, Form: Layout.Form, Reading: Vision.Reading) -> np.ndarray:
	View = cv2.cvtColor(Flat, cv2.COLOR_GRAY2BGR)
	Groups = (
		(Vision.FormPoints(Form), Reading.Scores, Form.Radius),
		(Vision.RollPoints.reshape(-1, 2), Reading.RollScores.reshape(-1), Layout.BubbleRadius),
	)
	for Points, Scores, Size in Groups:
		Radius = int(Size * Layout.Scale)
		for Point, Score in zip(Points, Scores):
			Spot = (int(Point[0]), int(Point[1]))
			Marked = Score >= Vision.MarkThreshold
			cv2.circle(View, Spot, Radius + 3, Pink if Marked else Blue, 3 if Marked else 1, cv2.LINE_AA)
			if Score >= 0.05:
				Label(View, f"{int(Score * 100)}", Spot[0], Spot[1] - Radius - 7, 0.55, Pink if Marked else Blue, 1, "Centre")
	Label(View, f"FLATTENED SHEET  /  fill score per bubble, marked at {int(Vision.MarkThreshold * 100)} or more", 190, Layout.WarpHeight - 28, 0.7, Pink, 1)
	return View


def Place(Canvas: np.ndarray, View: np.ndarray) -> None:
	Height, Width = View.shape[:2]
	Fit = min(ViewWidth / Width, ViewHeight / Height)
	Size = (int(Width * Fit), int(Height * Fit))
	Resized = cv2.resize(View, Size, interpolation=cv2.INTER_AREA if Fit < 1 else cv2.INTER_LINEAR)
	Left = (ViewWidth - Size[0]) // 2
	Top = (ViewHeight - Size[1]) // 2
	Canvas[Top:Top + Size[1], Left:Left + Size[0]] = Resized


def DrawStrip(Canvas: np.ndarray, Current: Shown | None) -> None:
	Label(Canvas, "THIS SHEET", 16, 566, 0.5, Pink)
	if Current is None:
		Label(Canvas, "Hold a sheet up to the camera. Keep all four corner squares in view.", 16, 640, 0.62, Mute)
		return
	CellWidth = 62
	for Row, Answer in enumerate(Current.Answers):
		Left = 16 + (Row % 15) * CellWidth
		Top = 580 + (Row // 15) * 66
		Shade = StatusColours[Current.Status[Row]]
		cv2.rectangle(Canvas, (Left, Top), (Left + CellWidth - 6, Top + 58), Panel, -1)
		cv2.rectangle(Canvas, (Left, Top), (Left + CellWidth - 6, Top + 4), Shade, -1)
		Label(Canvas, Current.Form.Labels[Row][:5], Left + 6, Top + 22, 0.4, Mute)
		Label(Canvas, Answer, Left + 28, Top + 46, 0.85 if len(Answer) < 3 else 0.6, Shade, 2, "Centre")


def DrawInsight(Canvas: np.ndarray, Form: Layout.Form | None, Stats: Insight | None) -> None:
	if Form is None or Stats is None or Stats.Count == 0:
		Label(Canvas, "CLASS INSIGHT", 16, 566, 0.5, Pink)
		Label(Canvas, "No saved sheets for this test yet.", 16, 640, 0.62, Mute)
		return
	Base = 690
	Tall = 92
	if Form.Mode == "Marks" and Stats.Topics:
		Label(Canvas, "CLASS INSIGHT  /  share of the marks the class earned in each topic", 16, 566, 0.5, Pink)
		BarWidth = min(150, 928 // len(Stats.Topics))
		for Index, (Topic, Share) in enumerate(Stats.Topics):
			Left = 16 + Index * BarWidth
			Shade = Green if Share >= 0.7 else Gold if Share >= 0.5 else Red
			cv2.rectangle(Canvas, (Left, Base - Tall), (Left + BarWidth - 10, Base), Panel, -1)
			cv2.rectangle(Canvas, (Left, Base - int(Tall * Share)), (Left + BarWidth - 10, Base), Shade, -1)
			Label(Canvas, f"{int(round(100 * Share))}%", Left + 8, Base - 8, 0.6, Ink, 2)
			Label(Canvas, Topic[:BarWidth // 10], Left, Base + 18, 0.42, White)
		return

	Title = "share of the marks the class earned on each question" if Form.Mode == "Marks" else "share of the class that got each question right, and the wrong option most of them picked"
	Label(Canvas, f"CLASS INSIGHT  /  {Title}", 16, 566, 0.5, Pink)
	BarWidth = 31
	for Row, Share in enumerate(Stats.Shares):
		Left = 16 + Row * BarWidth
		Shade = Green if Share >= 0.7 else Gold if Share >= 0.4 else Red
		cv2.rectangle(Canvas, (Left, Base - Tall), (Left + BarWidth - 6, Base), Panel, -1)
		cv2.rectangle(Canvas, (Left, Base - int(Tall * Share)), (Left + BarWidth - 6, Base), Shade, -1)
		Label(Canvas, Form.Labels[Row].lstrip("0")[:3], Left + 12, Base + 18, 0.38, Mute, 1, "Centre")
		if Stats.Traps[Row] and Share < 0.7:
			Label(Canvas, Stats.Traps[Row], Left + 12, Base - Tall - 6, 0.45, White, 1, "Centre")


def DrawPanel(Canvas: np.ndarray, Current: Shown | None, Keys: KeyBook, Stats: Insight | None, Progress: float, Tester: LiveTest | None, Milliseconds: float) -> None:
	cv2.rectangle(Canvas, (PanelLeft, 0), (CanvasWidth, CanvasHeight), Panel, -1)
	cv2.line(Canvas, (PanelLeft, 0), (PanelLeft, CanvasHeight), Line, 1)
	Left = PanelLeft + 24
	Label(Canvas, "GLANCE", Left, 52, 1.25, Pink, 2)
	Label(Canvas, "bubble sheets, read by a webcam", Left, 76, 0.45, Mute)
	cv2.line(Canvas, (Left, 96), (CanvasWidth - 24, 96), Line, 1)
	Marks = Current is not None and Current.Form.Mode == "Marks"

	if Current is None:
		Label(Canvas, "SEARCHING", Left, 140, 0.8, Blue, 1)
		Label(Canvas, "Show a sheet to begin.", Left, 168, 0.5, Mute)
	else:
		Entry = Keys.get(str(Current.TestId))
		Name = Current.Form.Name if Marks else Entry["Name"] if Entry else "No answer key yet"
		Label(Canvas, f"TEST {Current.TestId:02d}  /  {'MARKS' if Marks else 'ANSWERS'}", Left, 128, 0.55, Blue)
		Label(Canvas, Name[:26], Left, 152, 0.55, White)
		Label(Canvas, "ROLL NO.", Left, 196, 0.45, Mute)
		Label(Canvas, Current.Roll, Left, 240, 1.3, White, 2)
		Label(Canvas, "TOTAL MARKS" if Marks else "SCORE", Left, 286, 0.45, Mute)
		if Current.Score is None:
			Label(Canvas, f"--/{Current.Out:g}", Left, 352, 2.0, Mute, 3)
		else:
			Text = f"{Current.Score:g}/{Current.Out:g}"
			Label(Canvas, Text, Left, 352, 2.0 if len(Text) <= 5 else 1.5, Green, 3)
			if len(Text) <= 5 and Current.Out:
				Label(Canvas, f"{int(100 * Current.Score / Current.Out)}%", CanvasWidth - 24, 352, 0.8, White, 1, "Right")
		cv2.rectangle(Canvas, (Left, 372), (CanvasWidth - 24, 378), Line, -1)
		cv2.rectangle(Canvas, (Left, 372), (Left + int((CanvasWidth - 24 - Left) * min(Progress, 1.0)), 378), Current.NoteColour, -1)
		Label(Canvas, Current.Note[:30], Left, 406, 0.52, Current.NoteColour)

	cv2.line(Canvas, (Left, 430), (CanvasWidth - 24, 430), Line, 1)
	Label(Canvas, "CLASS SO FAR", Left, 458, 0.45, Mute)
	if Current is not None and Stats is not None and Stats.Count > 0:
		Label(Canvas, f"{Stats.Count} sheets, average {Stats.Average:.1f}/{Current.Out:g}", Left, 484, 0.52, White)
		if Marks and Stats.Topics:
			for Index, (Topic, Share) in enumerate(sorted(Stats.Topics, key=lambda Item: Item[1])[:3]):
				Label(Canvas, f"{Topic[:16]}: {int(round(100 * Share))}% of marks", Left, 512 + Index * 24, 0.47, Gold)
		else:
			Hardest = sorted(range(len(Stats.Shares)), key=lambda Row: Stats.Shares[Row])[:3]
			for Index, Row in enumerate(Hardest):
				Extra = f", most chose {Stats.Traps[Row]}" if Stats.Traps[Row] else ""
				Label(Canvas, f"Q{Current.Form.Labels[Row].lstrip('0')}: {int(100 * Stats.Shares[Row])}%{' right' if not Marks else ''}{Extra}", Left, 512 + Index * 24, 0.47, Gold)
	else:
		Label(Canvas, "No sheets saved yet.", Left, 484, 0.5, Mute)

	if Tester is not None:
		Label(Canvas, "LIVE ACCURACY TEST", Left, 600, 0.45, Pink)
		Label(Canvas, Tester.Line(), Left, 622, 0.45, White)

	Label(Canvas, "K set key   I insight   D debug   Q quit", Left, 662, 0.42, Mute)
	Label(Canvas, "Camera frames are never saved.", Left, 684, 0.42, Mute)
	Label(Canvas, f"{Milliseconds:.0f} ms per frame", Left, 706, 0.42, Line)


def Describe(TestId: int, Form: Layout.Form, Reading: Vision.Reading, Keys: KeyBook) -> Shown:
	Entry = Keys.get(str(TestId))
	Score, Out = Mark(Form, Reading.Answers, Entry)
	if Form.Mode == "Choice" and Entry is not None:
		Status = Vision.Grade(Reading.Answers, Entry["Key"])[1]
	else:
		Status = ["Blank" if Answer == Vision.Blank else "Multiple" if Answer == Vision.Multiple else "Read" for Answer in Reading.Answers]
	return Shown(TestId, Form, Reading.Roll, Reading.Answers, Score, Out, Status, "READING", Blue)


def Compose(Frame: np.ndarray, Sheet: Vision.Located | None, Reading: Vision.Reading | None, Current: Shown | None, Keys: KeyBook, Stats: Insight | None, Progress: float, Tester: LiveTest | None, ShowInsight: bool, ShowDebug: bool, Milliseconds: float) -> np.ndarray:
	Canvas = np.full((CanvasHeight, CanvasWidth, 3), Ink, dtype=np.uint8)
	if Sheet is not None and Reading is not None and Current is not None:
		if ShowDebug:
			View = DebugView(Vision.Flatten(Vision.ToGray(Frame), Sheet.Homography), Current.Form, Reading)
		else:
			Quad = Vision.ToFrame(Vision.PageCorners, Sheet.Inverse)[0]
			View = Shield(Frame, Quad)
			Entry = Keys.get(str(Sheet.TestId))
			DrawMarks(View, Sheet, Current, Reading, Entry["Key"] if Entry and Current.Form.Mode == "Choice" else None)
	else:
		View = Shield(Frame, None)
	Place(Canvas, View)
	cv2.line(Canvas, (0, ViewHeight), (ViewWidth, ViewHeight), Line, 1)
	if ShowInsight:
		DrawInsight(Canvas, Current.Form if Current else None, Stats)
	else:
		DrawStrip(Canvas, Current)
	DrawPanel(Canvas, Current, Keys, Stats, Progress, Tester, Milliseconds)
	return Canvas


def StatsFor(TestId: int | None, Store: ResultStore, Keys: KeyBook, Book: Forms) -> Insight | None:
	if TestId is None:
		return None
	return ClassInsight(Book.Get(TestId), Store.ForTest(TestId), Keys.get(str(TestId)))


def HandleLock(Current: Shown, Store: ResultStore, Keys: KeyBook, Book: Forms, Tester: LiveTest | None) -> None:
	if Vision.Unknown in Current.Roll:
		Current.Note, Current.NoteColour = "ROLL NUMBER UNCLEAR", Gold
		return
	if Tester is not None:
		Current.Note, Current.NoteColour = Tester.Record(Current.TestId, Current.Roll, Current.Answers), Pink
		Beep(1175)
		return
	if Current.Score is None:
		Current.Note, Current.NoteColour = "NO KEY: SHOW KEY SHEET, PRESS K", Gold
		return
	if Current.Form.Mode == "Marks" and Vision.Multiple in Current.Answers:
		Row = Current.Answers.index(Vision.Multiple)
		Current.Note, Current.NoteColour = f"Q{Current.Form.Labels[Row]}: TWO MARKS, FIX IT", Gold
		return
	Store.Put(Current.TestId, Current.Roll, Current.Answers)
	Store.Save(Keys, Book)
	Empty = Current.Answers.count(Vision.Blank)
	if Current.Form.Mode == "Marks" and Empty:
		Current.Note, Current.NoteColour = f"SAVED, {Empty} ROW{'S' if Empty > 1 else ''} LEFT BLANK", Gold
	else:
		Current.Note, Current.NoteColour = "SAVED", Green
	Beep(988)


def SetKey(Current: Shown | None, Store: ResultStore, Keys: KeyBook, Book: Forms) -> None:
	if Current is None or Current.Form.Mode != "Choice":
		return
	if any(Answer not in Layout.Options for Answer in Current.Answers):
		Current.Note, Current.NoteColour = "KEY NEEDS ALL 30 ANSWERS", Gold
		return
	Entry = Keys.setdefault(str(Current.TestId), {"Name": f"Test {Current.TestId:02d}", "Key": ""})
	Entry["Key"] = "".join(Current.Answers)
	KeysPath.write_text(json.dumps(Keys, indent="\t"))
	Store.Remove(Current.TestId, Current.Roll)
	Store.Save(Keys, Book)
	Current.Note, Current.NoteColour = "ANSWER KEY SAVED", Green
	Beep(1319)


def GradeImage(ImagePath: str, OutputPath: str | None, Keys: KeyBook, Book: Forms, ShowDebug: bool) -> None:
	Frame = cv2.imread(ImagePath)
	if Frame is None:
		raise SystemExit(f"Could not open {ImagePath}")
	Start = time.perf_counter()
	Result = Vision.ReadSheet(Frame, Book.Get)
	Milliseconds = (time.perf_counter() - Start) * 1000
	Store = ResultStore(ResultsPath)
	Current: Shown | None = None
	Sheet: Vision.Located | None = None
	Reading: Vision.Reading | None = None
	if Result is not None:
		Sheet, Form, Reading = Result
		Current = Describe(Sheet.TestId, Form, Reading, Keys)
		Current.Note, Current.NoteColour = ("READ", Green) if Current.Score is not None else ("NO ANSWER KEY", Gold)
		Score = "none" if Current.Score is None else f"{Current.Score:g}/{Current.Out:g}"
		print(f"[GLANCE] Test {Sheet.TestId:02d} | Roll {Reading.Roll} | Answers {Joiner.join(Reading.Answers)} | Score {Score}")
	else:
		print("[GLANCE] No sheet found in the image")
	Stats = StatsFor(Sheet.TestId if Sheet else None, Store, Keys, Book)
	Canvas = Compose(Frame, Sheet, Reading, Current, Keys, Stats, 1.0, None, False, ShowDebug, Milliseconds)
	if OutputPath:
		cv2.imwrite(OutputPath, Canvas)
	else:
		cv2.imshow(WindowName, Canvas)
		cv2.waitKey(0)


def RunCamera(CameraIndex: int, Keys: KeyBook, Book: Forms, Tester: LiveTest | None) -> None:
	Backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
	Capture = cv2.VideoCapture(CameraIndex, Backend)
	Capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
	Capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
	if not Capture.isOpened():
		raise SystemExit(f"Camera {CameraIndex} did not open. Try --camera 1")

	Store = ResultStore(ResultsPath)
	Track = Tracker()
	Current: Shown | None = None
	Stats: Insight | None = None
	StatsTestId: int | None = None
	ShowInsight = False
	ShowDebug = False
	Milliseconds = 0.0
	cv2.namedWindow(WindowName, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(WindowName, CanvasWidth, CanvasHeight)

	while True:
		Okay, Frame = Capture.read()
		if not Okay:
			break
		Start = time.perf_counter()
		Gray = Vision.ToGray(Frame)
		Sheet = Vision.LocateSheet(Gray)
		Reading: Vision.Reading | None = None

		if Sheet is None:
			Track.Lost()
		else:
			Form = Book.Get(Sheet.TestId)
			Scores, RollScores = Vision.MeasureBubbles(Gray, Sheet.Homography, Form)
			Reading = Track.Update(Sheet.TestId, Form, Scores, RollScores)
			if not Track.Handled:
				Current = Describe(Sheet.TestId, Form, Reading, Keys)
				if Track.Locked():
					HandleLock(Current, Store, Keys, Book, Tester)
					Track.Handled = True
					Stats = StatsFor(Sheet.TestId, Store, Keys, Book)
			if Sheet.TestId != StatsTestId:
				Stats = StatsFor(Sheet.TestId, Store, Keys, Book)
				StatsTestId = Sheet.TestId

		Showing = Sheet is not None and Current is not None and Current.TestId == Sheet.TestId
		Progress = 1.0 if Track.Handled else Track.Stable / LockFrames
		Milliseconds = Milliseconds * 0.9 + (time.perf_counter() - Start) * 100
		Canvas = Compose(Frame, Sheet if Showing else None, Reading if Showing else None, Current, Keys, Stats, Progress, Tester, ShowInsight, ShowDebug, Milliseconds)
		cv2.imshow(WindowName, Canvas)

		Pressed = cv2.waitKey(1) & 0xFF
		if Pressed in (ord("q"), 27):
			break
		if Pressed == ord("i"):
			ShowInsight = not ShowInsight
		elif Pressed == ord("d"):
			ShowDebug = not ShowDebug
		elif Pressed == ord("k") and Track.Locked():
			SetKey(Current, Store, Keys, Book)
			Stats = StatsFor(Current.TestId if Current else None, Store, Keys, Book)
		if cv2.getWindowProperty(WindowName, cv2.WND_PROP_VISIBLE) < 1:
			break

	Capture.release()
	cv2.destroyAllWindows()
	if Tester is not None:
		Tester.Finish()


# -- INIT --
def Main() -> None:
	Parser = argparse.ArgumentParser(description="Glance: read bubble sheets with a webcam")
	Parser.add_argument("--camera", type=int, default=0)
	Parser.add_argument("--image", type=str, default=None)
	Parser.add_argument("--out", type=str, default=None)
	Parser.add_argument("--debug", action="store_true")
	Parser.add_argument("--test", action="store_true")
	Parser.add_argument("--fresh", action="store_true")
	Args = Parser.parse_args()

	Keys = LoadKeys()
	Book = Forms()
	print(f"[GLANCE] Loaded in {(time.perf_counter() - InitTime) * 1000:.1f} ms | OpenCV {cv2.__version__} | {len(Keys)} answer keys, {len(Book.Papers)} marks papers | Version {Updated}")

	if Args.image:
		GradeImage(Args.image, Args.out, Keys, Book, Args.debug)
	else:
		RunCamera(Args.camera, Keys, Book, LiveTest(Args.fresh) if Args.test else None)


if __name__ == "__main__":
	Main()
