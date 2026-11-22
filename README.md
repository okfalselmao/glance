# Glance

A class's marks, read by a laptop webcam with OpenCV.

After checking a copy, the teacher bubbles the marks for each question once on a Glance marks sheet and holds it up to the camera. Glance reads the roll number and every row, adds the total, saves it, and shows which topics the class is weak in. The same program also grades multiple-choice answer sheets against a key.

Built for VISION.EXE at NEON SYNAPSE 2026.

## Set up (once)

1. Install Python 3.10 or newer.
2. In this folder run:

```
pip install opencv-python numpy
```

## Run

```
python Glance.py
```

Use `python Glance.py --camera 1` if the laptop has more than one camera.

| Key | What it does |
| --- | --- |
| I | Switches the bottom strip to class insight (topics, or questions) |
| D | Shows the flattened sheet with the fill score of every bubble |
| K | Answer sheets only: saves the sheet on screen as the answer key |
| Q | Quits |

## Marks sheets (for checked copies)

1. Describe the paper in `Papers.json`. Each test number gets a name, whether half marks are allowed, and one line per question or sub-part:

```
"3": {
	"Name": "Class 9 Physics UT2",
	"Half": true,
	"Questions": [
		{"Label": "1a", "Max": 2, "Topic": "Motion"},
		{"Label": "1b", "Max": 3, "Topic": "Motion"},
		{"Label": "2", "Max": 5, "Topic": "Force"}
	]
}
```

   Up to 30 rows, up to 10 marks per row, test numbers 1 to 30.

2. Print the sheet: `python MakeSheets.py --test 3` writes `Print/Marks_Blank_Test03_A4.png`, two sheets per A4 page.
3. For each copy: fill the roll number, one bubble per question for the marks given, and the `.5` bubble for a half mark.
4. Hold it up to the camera. When the bar fills, the total is shown and the sheet is saved.

What is saved: `Marks_Test03.csv` with roll number, marks for every question and the total, plus `Results.csv` for all tests.

If a row has two marks, Glance refuses the sheet and names the row. Blank rows are counted as 0 and flagged.

`Papers.json` already holds test 2, a sample science paper, and `Print/` has its blank sheet and 12 filled samples.

## Answer sheets (multiple choice)

Any test number that is not in `Papers.json` is a 30-question answer sheet: `python MakeSheets.py --test 1`.

Fill one sheet with the correct answers, hold it up, wait for the bar to fill and press K. The key is stored in `Keys.json`. Then show the students' sheets.

## Measure accuracy with the real camera

1. Print the sample sheets in `Print/` (their correct readings are in `Truth.json`).
2. Run `python Glance.py --test`.
3. Show each sheet several times: straight, tilted, far, near. Take the sheet fully out of view between reads.
4. Press Q.

When you quit, Glance adds up every read, writes the totals into the website and uploads it. Reads from earlier runs are kept and added to. Start again from zero with `python Glance.py --test --fresh`.

For hand-filled sheets, add a line to `Truth.json` first, with the rows separated by `|`: `"2-205": "3|2.5|4|-|1"` (test number, roll number, then each row; `-` for blank, `*` for two marks).

## Upload the website

Do this once:

1. Create an empty repository on GitHub.
2. In this folder run:

```
git init -b main
git add .
git commit -m "Glance"
git remote add origin https://github.com/YOUR-NAME/YOUR-REPO.git
git push -u origin main
```

3. In Vercel, import that repository and set the Root Directory to `website`.

From then on the numbers on the site fill themselves in and upload themselves whenever you finish a live test, a simulation test or a marker test. Only the `website` folder is uploaded by this step. Student marks (`Results.csv`, `Marks_Test*.csv`) are listed in `.gitignore` and never leave the laptop.

To update and upload by hand: `python Publish.py`.

## Rerun the simulation tests

```
python SimTest.py --shots 1000
python SimTest.py --shots 1000 --mode marks --seed 2027
python MarkerTest.py
```

All three update the website too. Add `--dry` to `SimTest.py` for a quick trial that changes nothing.

## Files

| File | Job |
| --- | --- |
| `Layout.py` | Where every marker and bubble sits on a sheet, for both sheet types |
| `Vision.py` | All the computer vision: find, flatten, measure, decide |
| `Glance.py` | The live app: camera loop, tracker, screen, totals, topic report, saving |
| `MakeSheets.py` | Draws printable sheets and sample sheets |
| `SimTest.py` | Stress test on simulated camera shots |
| `MarkerTest.py` | Tests which ArUco markers survive blur |
| `Publish.py` | Writes the test numbers into the website and pushes it to GitHub |
| `Papers.json` | The pattern of each marks paper |
| `Keys.json` | Answer keys for multiple-choice tests |
| `EXPLAIN.md` | Plain-language walkthrough of the code and likely judge questions |
| `website/` | The project website |

## Privacy

Camera frames are never written to disk. Everything outside the sheet is pixelated on screen. The only things saved are the test number, roll number, the rows read and the total, in files on this laptop.

## Sources

- OpenCV (Apache 2.0 licence), including its built-in ArUco dictionary `DICT_4X4_50`.
- NumPy.
- No dataset and no pre-trained model are used.
