import cv2

# Open default webcam
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Check camera connection
if not camera.isOpened():
    print("Error: Could not access webcam.")
    exit()

# Set camera resolution
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Camera started successfully!")
print("Press Q to close the camera.")

while True:
    ret, frame = camera.read()

    if not ret:
        print("Error: Unable to capture frame.")
        break

    # Display camera preview
    cv2.imshow("IV Drip Detection - Camera Preview", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
camera.release()
cv2.destroyAllWindows()

print("Camera closed.")