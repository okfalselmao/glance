# -- IMPORTS --
import json
from pathlib import Path

import cv2
import numpy as np

import Layout
import Publish
import Vision

# -- DATA --
Root = Path(__file__).resolve().parent
OutputPath = Root / "MarkerResults.json"
Trials: int = 60
Seed: int = 1
Stage: int = 200


# -- FUNCTIONS --
def BlurryMarker(Id: int, Random: np.random.Generator) -> np.ndarray:
	Side = int(Random.integers(34, 52))
	Big = Side * 4
	Marker = cv2.aruco.generateImageMarker(Vision.Dictionary, Id, Big)
	Card = np.full((Big * 2, Big * 2), 235, dtype=np.uint8)
	Card[Big // 2:Big // 2 + Big, Big // 2:Big // 2 + Big] = Marker
	Turn = cv2.getRotationMatrix2D((Big, Big), float(Random.uniform(-25, 25)), 0.25)
	Turn[0, 2] = Stage / 2 - (Turn[0, 0] * Big + Turn[0, 1] * Big)
	Turn[1, 2] = Stage / 2 - (Turn[1, 0] * Big + Turn[1, 1] * Big)
	Small = cv2.warpAffine(cv2.GaussianBlur(Card, (0, 0), 2), Turn, (Stage, Stage), borderValue=235).astype(np.float32)
	Small = cv2.GaussianBlur(Small, (0, 0), float(Random.uniform(1.2, 2.0)))
	Small = Small * float(Random.uniform(0.6, 1.0)) + Random.normal(0, 5, Small.shape)
	return np.clip(Small, 0, 255).astype(np.uint8)


# -- INIT --
def Main() -> None:
	Random = np.random.default_rng(Seed)
	Rates: dict[int, float] = {}
	for Id in range(50):
		Seen = 0
		for _ in range(Trials):
			_, Ids, _ = Vision.Detector.detectMarkers(BlurryMarker(Id, Random))
			Seen += int(Ids is not None and Id in Ids.flatten())
		Rates[Id] = Seen / Trials

	Ranked = sorted(Rates.items(), key=lambda Item: -Item[1])
	Used = [Layout.MarkerTopLeft, Layout.MarkerTopRight, Layout.MarkerBottomLeft, *Layout.TestMarkers]
	Summary = {
		"Trials": Trials,
		"Rates": {str(Id): Rate for Id, Rate in Ranked},
		"WorstId": Ranked[-1][0],
		"WorstRate": Ranked[-1][1],
		"MarkerZeroRate": Rates[0],
		"Used": Used,
		"UsedLowest": min(Rates[Id] for Id in Used),
		"OpenCv": cv2.__version__,
	}
	OutputPath.write_text(json.dumps(Summary, indent="\t"))
	print(f"[MARKERTEST] {Trials} blurry trials per marker | marker 0 seen {100 * Rates[0]:.0f}% | worst is marker {Ranked[-1][0]} at {100 * Ranked[-1][1]:.0f}% | lowest marker we use {100 * Summary['UsedLowest']:.0f}%")
	Publish.Sync()


if __name__ == "__main__":
	Main()
