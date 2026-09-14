import cv2

CAMERA_INDEX = 0
OUTPUT_FILE = "mv-sua231gm.jpg"
EXPOSURE = -6
GAIN = 0


def set_manual_exposure_and_gain(camera, exposure=EXPOSURE, gain=GAIN):
	# DirectShow commonly uses 0.25 for manual exposure and 0.75 for auto.
	camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
	camera.set(cv2.CAP_PROP_EXPOSURE, exposure)
	camera.set(cv2.CAP_PROP_GAIN, gain)

	actual_exposure = camera.get(cv2.CAP_PROP_EXPOSURE)
	actual_gain = camera.get(cv2.CAP_PROP_GAIN)
	if actual_exposure == -1:
		print("Warning: the current driver does not expose manual exposure through OpenCV.")
	else:
		print(f"Exposure: requested {exposure}, actual {actual_exposure}")
	print(f"Gain: requested {gain}, actual {actual_gain}")


def take_photo(camera_index=CAMERA_INDEX, output_file=OUTPUT_FILE):
	camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

	if not camera.isOpened():
		raise RuntimeError(
			f"Could not open camera index {camera_index}. "
			"Try another index, such as 1 or 2."
		)

	try:
		set_manual_exposure_and_gain(camera)
		success, frame = camera.read()
		if not success or frame is None:
			raise RuntimeError("The camera opened, but no image was received.")

		if not cv2.imwrite(output_file, frame):
			raise RuntimeError(f"Could not save the photo to {output_file!r}.")
	finally:
		camera.release()

	print(f"Photo saved to {output_file}")


if __name__ == "__main__":
	take_photo()

