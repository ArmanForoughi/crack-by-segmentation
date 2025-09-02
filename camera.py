from picamera2 import Picamera2, Preview
import time

start_time = 12

picam2 = Picamera2()
camera_config = picam2.create_preview_configuration()
picam2.configure(camera_config)	


num_photos = 6
path = "pic_camera/"
while True:
	if time.localtime().tm_hour == start_time:
		try:
			for i in range(num_photos):
				picam2.start_preview(Preview.QTGL)
				picam2.start()
				time.sleep(2)
				picam2.capture_file(f"{path}photo_{i}.jpg")
				picam2.stop_preview()		
			picam2.stop_preview()
		except:
			pass
	time.sleep(60*60)

