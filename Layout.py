# -- IMPORTS --
import json
import math
from dataclasses import dataclass
from pathlib import Path

# -- PATHS --
Root = Path(__file__).resolve().parent
PapersPath = Root / "Papers.json"

# -- DATA --
PageWidth: float = 210.0
PageHeight: float = 148.0
Scale: float = 6.0
WarpWidth: int = int(PageWidth * Scale)
WarpHeight: int = int(PageHeight * Scale)

MarkerSize: float = 18.0
MarkerInset: float = 6.0
MarkerTopLeft: int = 2
MarkerTopRight: int = 6
MarkerBottomLeft: int = 8
TestMarkers: tuple[int, ...] = (
	10, 13, 15, 17, 21, 22, 23, 26, 29, 30,
	31, 33, 37, 38, 40, 41, 43, 45, 49, 3,
	4, 5, 7, 16, 18, 24, 25, 32, 42, 48,
)
MaxTestId: int = len(TestMarkers)

BubbleRadius: float = 3.0
Options: str = "ABCD"
OptionCount: int = 4
QuestionCount: int = 30
RowsPerBlock: int = 10
RowTop: float = 36.0
RowPitch: float = 9.0

RollDigits: int = 3
RollLabelX: float = 11.0
RollFirstX: float = 21.0
RollPitch: float = 9.5

BlockLefts: tuple[float, ...] = (54.0, 104.0, 154.0)
BlockLabelOffset: float = 6.5
BlockFirstOffset: float = 13.0
OptionPitch: float = 10.0

MarksLeft: float = 51.0
MarksRight: float = 201.0
MarksLabelWidth: float = 11.0
MarksMaxPitch: float = 10.0
MarksRadiusShare: float = 0.42
HalfHead: str = ".5"

Point = tuple[float, float]


@dataclass(frozen=True)
class Form:
	Mode: str
	Name: str
	Labels: tuple[str, ...]
	Topics: tuple[str, ...]
	Maxes: tuple[int, ...]
	Rows: tuple[tuple[Point, ...], ...]
	Heads: tuple[tuple[str, float], ...]
	LabelXs: tuple[float, ...]
	Radius: float
	Half: bool


# -- FUNCTIONS --
def MarkerOrigins(TestId: int) -> dict[int, Point]:
	Far = PageWidth - MarkerInset - MarkerSize
	Low = PageHeight - MarkerInset - MarkerSize
	return {
		MarkerTopLeft: (MarkerInset, MarkerInset),
		MarkerTopRight: (Far, MarkerInset),
		MarkerBottomLeft: (MarkerInset, Low),
		TestMarkers[TestId - 1]: (Far, Low),
	}


def MarkerCorners(Origin: Point) -> list[Point]:
	X, Y = Origin
	return [
		(X, Y),
		(X + MarkerSize, Y),
		(X + MarkerSize, Y + MarkerSize),
		(X, Y + MarkerSize),
	]


def RowY(Row: int) -> float:
	return RowTop + Row * RowPitch


def QuestionBubble(Question: int, Option: int) -> Point:
	Block = Question // RowsPerBlock
	Row = Question % RowsPerBlock
	return (BlockLefts[Block] + BlockFirstOffset + Option * OptionPitch, RowY(Row))


def RollBubble(Digit: int, Value: int) -> Point:
	return (RollFirstX + Digit * RollPitch, RowY(Value))


def QuestionCentres() -> list[list[Point]]:
	return [[QuestionBubble(Q, O) for O in range(OptionCount)] for Q in range(QuestionCount)]


def RollCentres() -> list[list[Point]]:
	return [[RollBubble(D, V) for D in range(RollDigits)] for V in range(10)]


def ChoiceForm(Name: str = "") -> Form:
	Heads: list[tuple[str, float]] = []
	for Left in BlockLefts:
		for Option, Letter in enumerate(Options):
			Heads.append((Letter, Left + BlockFirstOffset + Option * OptionPitch))
	return Form(
		"Choice",
		Name,
		tuple(f"{Question + 1:02d}" for Question in range(QuestionCount)),
		tuple("" for _ in range(QuestionCount)),
		tuple(1 for _ in range(QuestionCount)),
		tuple(tuple(Row) for Row in QuestionCentres()),
		tuple(Heads),
		tuple(BlockLefts[Question // RowsPerBlock] + BlockLabelOffset for Question in range(QuestionCount)),
		BubbleRadius,
		False,
	)


def MarksForm(Name: str, Questions: list[dict], Half: bool) -> Form:
	Count = len(Questions)
	if not 1 <= Count <= QuestionCount:
		raise ValueError(f"A marks paper needs 1 to {QuestionCount} rows")
	Blocks = math.ceil(Count / RowsPerBlock)
	Highest = max(int(Question["Max"]) for Question in Questions)
	Columns = Highest + 1 + int(Half)
	BlockWidth = (MarksRight - MarksLeft) / Blocks
	Pitch = min(MarksMaxPitch, (BlockWidth - MarksLabelWidth - 1.0) / Columns)
	Radius = min(BubbleRadius, round(Pitch * MarksRadiusShare, 2))

	def ColumnX(Block: int, Column: int) -> float:
		return MarksLeft + Block * BlockWidth + MarksLabelWidth + Pitch * (Column + 0.5)

	Rows: list[tuple[Point, ...]] = []
	for Index, Question in enumerate(Questions):
		Block = Index // RowsPerBlock
		Y = RowY(Index % RowsPerBlock)
		Row = [(ColumnX(Block, Mark), Y) for Mark in range(int(Question["Max"]) + 1)]
		if Half:
			Row.append((ColumnX(Block, Columns - 1), Y))
		Rows.append(tuple(Row))

	Heads: list[tuple[str, float]] = []
	for Block in range(Blocks):
		for Mark in range(Highest + 1):
			Heads.append((str(Mark), ColumnX(Block, Mark)))
		if Half:
			Heads.append((HalfHead, ColumnX(Block, Columns - 1)))

	return Form(
		"Marks",
		Name,
		tuple(str(Question["Label"]) for Question in Questions),
		tuple(str(Question.get("Topic", "")) for Question in Questions),
		tuple(int(Question["Max"]) for Question in Questions),
		tuple(Rows),
		tuple(Heads),
		tuple(MarksLeft + (Index // RowsPerBlock) * BlockWidth + MarksLabelWidth - 2.0 for Index in range(Count)),
		Radius,
		Half,
	)


def LoadPapers() -> dict[str, dict]:
	if PapersPath.exists():
		return json.loads(PapersPath.read_text())
	return {}


def FormFor(TestId: int, Papers: dict[str, dict]) -> Form:
	Paper = Papers.get(str(TestId))
	if Paper is None:
		return ChoiceForm()
	return MarksForm(Paper.get("Name", ""), Paper["Questions"], bool(Paper.get("Half", False)))
