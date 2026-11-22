# Glance: how to explain it

Both team members should be able to say everything on this page without reading it.

## The 30-second pitch

We asked our teacher how marks get recorded. After checking the copies she writes each question's marks on the front page, then fills an OMR sheet with the same marks, the sheets go to a centre, a machine adds them up, and she rechecks the machine's totals. She does it because marks kept question by question show which topics the class is weak in. Glance does the same job on her own laptop: she bubbles the marks once, holds the sheet up to the webcam, and gets the total, a saved marks file and a topic-wise class report straight away. It uses only OpenCV, no trained model, and it never saves a camera frame.

Ask her before the event whether you may name her and quote her. It is the strongest thing you can say to a judge.

## The pipeline, one step at a time

All of this is in `Vision.py`.

1. **Find the sheet** (`LocateSheet`). The four black squares are ArUco markers. `detectMarkers` returns the four corner points and the ID number of each marker. IDs 2, 6 and 8 are always top-left, top-right and bottom-left. The fourth ID tells us the test number, and the test number tells us which sheet this is: a marks paper from `Papers.json`, or a multiple-choice answer sheet.
2. **Flatten it** (`LocateSheet`, `Flatten`). Four markers with four corners each give 16 points. `Layout.py` knows where those 16 points are on the printed page in millimetres. `cv2.findHomography` finds the 3x3 matrix that maps camera pixels to page pixels, and `cv2.warpPerspective` uses it to produce a straight 1260x888 image, 6 pixels per millimetre. Because the IDs give the order of the points, the sheet can even be upside down.
3. **Cancel shadows** (`Darkness`). A morphological close (dilate, then erode) with a kernel bigger than a bubble removes dark ink and leaves the paper brightness. Dividing the image by that paper brightness gives a value near 1 for paper and lower for ink, whatever the lighting. `1 - ratio` is the ink map: 0 is paper, 1 is black.
4. **Measure bubbles** (`MeasureBubbles`). `cv2.filter2D` with a small disc-shaped kernel replaces every pixel with the average ink around it. Reading that image at each bubble centre gives the fill score of the bubble. The disc is 62% of the bubble radius so the printed outline is not counted.
5. **Decide each row** (`MarkedBubbles`, `PickMarks`, `PickOne`). A bubble is marked if its score is at least 0.28 and at least 45% of the darkest bubble in its row. On a marks sheet: no marked bubble is blank, one number bubble is that many marks, the `.5` bubble adds a half, two number bubbles is `*` and the sheet is refused. On an answer sheet: one bubble is the answer, two is `*`.
6. **Total or grade** (`Total`, `Grade`). Marks are added up. Answers are compared with the key.

`Glance.py` adds the live part:

- `Tracker` averages the fill scores of the last 5 frames and locks a reading only after it has been identical for 6 frames in a row. A shaking hand cannot give a wrong total.
- `Shield` pixelates everything outside the sheet before the frame is shown.
- `ResultStore` saves `Results.csv` and, for each marks test, `Marks_TestNN.csv`.
- `ClassInsight` adds up, per topic, the marks the class earned out of the marks possible.

`Layout.py` builds the sheet from the paper pattern: the number of bubble columns is the highest marks plus one (plus one more for `.5`), and the bubbles shrink to fit, down to about 2.2 mm radius for 20 rows marked out of 10 with half marks.

## Numbers to know

- Sheet: A5 landscape, up to 30 rows, marks up to 10 per row, 3-digit roll number.
- Markers: 18 mm, OpenCV dictionary `DICT_4X4_50`.
- Flattened image: 1260x888 pixels, 6 px per mm.
- Mark threshold: 0.28. Second-mark ratio: 0.45.
- Lock: 6 identical readings in a row, scores averaged over 5 frames.
- Simulation, most crowded marks sheet (270 bubbles): 994 of 1,000 found, 0 wrong readings in 22,862.
- Simulation, answer sheet: 995 of 1,000 found, 1 wrong reading in 32,835.
- Marker test: marker 0 was found 67% of the time under blur, so the sheet uses only markers that scored 98% or more.
- Run the tests on your own laptop and quote the numbers they print.

## Questions judges are likely to ask

**How is this different from what the school already does?**
Today the marks are written twice, the sheets travel to a centre, and the totals are rechecked afterwards. With Glance the marks are bubbled once and read at the teacher's desk, and she sees the total and every row it read while the copy is still in her hand.

**Has the teacher used it?**
Answer truthfully. If she has tried it before the event, say what she said. If not, say it is built from her description and a trial is the next step.

**Can it replace the school's system?**
Not yet. It is a prototype that saves a marks file. Connecting to the school's system and a trial with real copies are the next steps.

**Why ArUco markers and not just finding the paper's edges?**
Edges fail when the background is light or a hand covers a side. Markers are found from any angle, give 16 exact points instead of 4, tell us which way up the sheet is, and carry the test number.

**Why no machine learning?**
The task does not need it. Position is solved by the markers and a filled bubble is simply darker than paper. A rule we can explain is better than a model we cannot, and it needs no dataset.

**How did you choose 0.28?**
From the histogram on the website. Unmarked bubbles, including rubbed-out marks, stay under 0.22. Real marks start around 0.32. 0.28 sits in the gap.

**Is zero errors the real accuracy?**
No. That is a simulation, and we tuned the threshold on simulated sheets. The real figure is whatever the live test at the counter shows. Say the live number, whatever it is.

**What if she bubbles the wrong mark?**
Glance cannot know what she meant, but it shows every mark it read beside the sheet, so she checks it against the copy in two seconds. Two marks in one row are refused and blank rows are flagged.

**What about privacy?**
No frame is saved. The screen pixelates everything outside the sheet. Only test number, roll number, the rows and the total are stored, on the laptop. Student marks are kept out of the GitHub upload.

**Doesn't this already exist?**
Phone apps such as ZipGrade grade multiple-choice sheets. Look one up before the event so you can compare honestly. Ours is built for marks from checked copies with a topic report, runs offline on a laptop a school already has, and we can change anything in it.

**Did you use AI?**
Answer truthfully. Claude helped write the code and the website, and the website says so. Then prove you own it: explain any function they point at.

## Demo plan for the counter

1. Before the judges arrive: about ten filled marks sheets already scanned so the topic report has data.
2. Hand a judge a blank marks sheet and a pen. Ask them to act as the teacher: a roll number and marks for each question.
3. Hold it up. Let them watch the bar fill, the rings appear and the total come up. Ask them to add it up themselves.
4. Press I for the topic report. Press D for the debug view and explain fill scores.
5. Have them fill two marks in one row, and show the sheet being refused.
6. Tilt the sheet, turn it upside down, cover a corner with a thumb to show an honest failure.
7. Open the website for the results.

## Before the event

- Run the app on the laptop and camera you will use.
- Put your teacher's real paper pattern in `Papers.json` and print that sheet. Show it to her if you can.
- Clip sheets to a clipboard or stiff card so they stay flat. Hold the sheet close enough to fill most of the view.
- Run `python Glance.py --test` with the printed sample sheets, at least 30 reads.
- Hand-fill real sheets with the pens and pencils teachers actually use. If light marks read as blank, press D, look at the scores and lower `MarkThreshold` in `Vision.py`.
- Bring spare printed sheets, pens, and the charger.
