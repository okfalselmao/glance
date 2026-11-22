# -- IMPORTS --
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import cv2
import numpy as np

import Layout

# -- DATA --
Dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
Parameters = cv2.aruco.DetectorParameters()
Parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
Detector = cv2.aruco.ArucoDetector(Dictionary, Parameters)

MarkThreshold: float = 0.28
RivalRatio: float = 0.45
InnerRatio: float = 0.62
BackgroundShrink: int = 4
BackgroundKernel: int = 15

Blank: str = "-"
Multiple: str = "*"
Unknown: str = "?"

RollPoints = np.array(Layout.RollCentres(), dtype=np.float32) * Layout.Scale
PageCorners = np.array([[
	[0.0, 0.0],
	[Layout.PageWidth, 0.0],
	[Layout.PageWidth, Layout.PageHeight],
	[0.0, Layout.PageHeight],
]], dtype=np.float32) * Layout.Scale

CloseKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (BackgroundKernel, BackgroundKernel))
FormLookup = Callable[[int], Layout.Form]


@dataclass
class Located:
	TestId: int
	Homography: np.ndarray
	Inverse: np.ndarray


@dataclass
class Reading:
	Answers: list[str]
	Roll: str
	Scores: np.ndarray
	RollScores: np.ndarray


# -- FUNCTIONS --
@lru_cache(maxsize=8)
def Disc(Radius: float) -> np.ndarray:
	Inner = int(round(Radius * Layout.Scale * InnerRatio))
	Size = Inner * 2 + 1
	Kernel = np.zeros((Size, Size), dtype=np.float32)
	cv2.circle(Kernel, (Inner, Inner), Inner, 1.0, -1)
	return Kernel / Kernel.sum()


@lru_cache(maxsize=64)
def FormPoints(Form: Layout.Form) -> np.ndarray:
	return np.array([Point for Row in Form.Rows for Point in Row], dtype=np.float32) * Layout.Scale


@lru_cache(maxsize=64)
def RowStarts(Form: Layout.Form) -> tuple[int, ...]:
	Starts = [0]
	for Row in Form.Rows:
		Starts.append(Starts[-1] + len(Row))
	return tuple(Starts)


def RowScores(Form: Layout.Form, Scores: np.ndarray, Row: int) -> np.ndarray:
	Starts = RowStarts(Form)
	return Scores[Starts[Row]:Starts[Row + 1]]


def ToGray(Frame: np.ndarray) -> np.ndarray:
	if Frame.ndim == 2:
		return Frame
	return cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)


def LocateSheet(Gray: np.ndarray) -> Located | None:
	Corners, Ids, _ = Detector.detectMarkers(Gray)
	if Ids is None:
		return None

	Found: dict[int, np.ndarray] = {}
	for Corner, Id in zip(Corners, Ids.flatten()):
		Found[int(Id)] = Corner.reshape(4, 2)

	TestIds = [Layout.TestMarkers.index(Id) + 1 for Id in Found if Id in Layout.TestMarkers]
	if len(TestIds) != 1:
		return None

	TestId = TestIds[0]
	Origins = Layout.MarkerOrigins(TestId)
	if any(Id not in Found for Id in Origins):
		return None

	Source: list[np.ndarray] = []
	Target: list[list[float]] = []
	for Id, Origin in Origins.items():
		Source.append(Found[Id])
		for X, Y in Layout.MarkerCorners(Origin):
			Target.append([X * Layout.Scale, Y * Layout.Scale])

	SourcePoints = np.concatenate(Source).astype(np.float32)
	TargetPoints = np.array(Target, dtype=np.float32)
	Homography, _ = cv2.findHomography(SourcePoints, TargetPoints, 0)
	if Homography is None:
		return None

	Okay, Inverse = cv2.invert(Homography)
	if Okay == 0:
		return None
	return Located(TestId, Homography, Inverse)


def Flatten(Gray: np.ndarray, Homography: np.ndarray) -> np.ndarray:
	return cv2.warpPerspective(
		Gray, Homography, (Layout.WarpWidth, Layout.WarpHeight),
		flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
	)


def Darkness(Flat: np.ndarray) -> np.ndarray:
	Small = cv2.resize(
		Flat, (Layout.WarpWidth // BackgroundShrink, Layout.WarpHeight // BackgroundShrink),
		interpolation=cv2.INTER_AREA,
	)
	Paper = cv2.morphologyEx(Small, cv2.MORPH_CLOSE, CloseKernel)
	Paper = cv2.GaussianBlur(Paper, (0, 0), 5)
	Paper = cv2.resize(Paper, (Layout.WarpWidth, Layout.WarpHeight), interpolation=cv2.INTER_LINEAR)
	Ratio = Flat.astype(np.float32) / np.maximum(Paper.astype(np.float32), 1.0)
	return 1.0 - np.clip(Ratio, 0.0, 1.0)


def Sample(FillMap: np.ndarray, Points: np.ndarray) -> np.ndarray:
	Xs = np.clip(np.rint(Points[..., 0]).astype(np.intp), 0, Layout.WarpWidth - 1)
	Ys = np.clip(np.rint(Points[..., 1]).astype(np.intp), 0, Layout.WarpHeight - 1)
	return FillMap[Ys, Xs]


def MeasureBubbles(Gray: np.ndarray, Homography: np.ndarray, Form: Layout.Form) -> tuple[np.ndarray, np.ndarray]:
	Ink = Darkness(Flatten(Gray, Homography))
	RollMap = cv2.filter2D(Ink, -1, Disc(Layout.BubbleRadius), borderType=cv2.BORDER_REPLICATE)
	FormMap = RollMap if Form.Radius == Layout.BubbleRadius else cv2.filter2D(Ink, -1, Disc(Form.Radius), borderType=cv2.BORDER_REPLICATE)
	return Sample(FormMap, FormPoints(Form)), Sample(RollMap, RollPoints)


def MarkedBubbles(Scores: np.ndarray) -> np.ndarray:
	Top = float(Scores.max())
	if Top < MarkThreshold:
		return np.empty(0, dtype=np.intp)
	return np.flatnonzero((Scores >= MarkThreshold) & (Scores >= Top * RivalRatio))


def PickOne(Scores: np.ndarray) -> int:
	Marked = MarkedBubbles(Scores)
	if len(Marked) == 0:
		return -1
	if len(Marked) > 1:
		return -2
	return int(Marked[0])


def PickMarks(Scores: np.ndarray, Highest: int, Half: bool) -> str:
	Marked = MarkedBubbles(Scores)
	if len(Marked) == 0:
		return Blank
	Whole = [int(Index) for Index in Marked if Index <= Highest]
	Extra = Half and (len(Scores) - 1) in Marked
	if len(Whole) > 1:
		return Multiple
	Value = (Whole[0] if Whole else 0) + (0.5 if Extra else 0.0)
	if Value > Highest:
		return Multiple
	return f"{Value:g}"


def Decide(Form: Layout.Form, Scores: np.ndarray, RollScores: np.ndarray) -> Reading:
	Answers: list[str] = []
	for Row in range(len(Form.Rows)):
		Line = RowScores(Form, Scores, Row)
		if Form.Mode == "Marks":
			Answers.append(PickMarks(Line, Form.Maxes[Row], Form.Half))
		else:
			Pick = PickOne(Line)
			Answers.append(Blank if Pick == -1 else Multiple if Pick == -2 else Layout.Options[Pick])

	Roll = ""
	for Digit in range(Layout.RollDigits):
		Pick = PickOne(RollScores[:, Digit])
		Roll += Unknown if Pick < 0 else str(Pick)

	return Reading(Answers, Roll, Scores, RollScores)


def ReadSheet(Frame: np.ndarray, Lookup: FormLookup) -> tuple[Located, Layout.Form, Reading] | None:
	Gray = ToGray(Frame)
	Sheet = LocateSheet(Gray)
	if Sheet is None:
		return None
	Form = Lookup(Sheet.TestId)
	Scores, RollScores = MeasureBubbles(Gray, Sheet.Homography, Form)
	return Sheet, Form, Decide(Form, Scores, RollScores)


def ToFrame(Points: np.ndarray, Inverse: np.ndarray) -> np.ndarray:
	Shape = Points.shape
	Flat = Points.reshape(-1, 1, 2).astype(np.float32)
	return cv2.perspectiveTransform(Flat, Inverse).reshape(Shape)


def Grade(Answers: list[str], Key: str) -> tuple[int, list[str]]:
	Score = 0
	Status: list[str] = []
	for Answer, Right in zip(Answers, Key):
		if Answer == Right:
			Score += 1
			Status.append("Correct")
		elif Answer == Blank:
			Status.append("Blank")
		elif Answer == Multiple:
			Status.append("Multiple")
		else:
			Status.append("Wrong")
	return Score, Status


def MarkValue(Answer: str) -> float:
	return 0.0 if Answer in (Blank, Multiple) else float(Answer)


def Total(Answers: list[str]) -> float:
	return sum(MarkValue(Answer) for Answer in Answers)
