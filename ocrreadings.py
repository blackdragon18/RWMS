#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import pytesseract
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
# Set the rectangular region of interest (change these values based on your requirements)
x, y, w, h = 800, 10, 300, 1050


# Configure your email details
sender_email = "lostanonymous777@gmail.com"  # Replace with your email
receiver_email = "lostanonymous777@gmail.com"  # Replace with recipient's email
email_password = "estt gdgn vlcj mmvd"   # Replace with your email password
smtp_server = "smtp.gmail.com"
smtp_port = 587

def preprocess_frame(frame, x, y, w, h):
    
    # Initialize numeric_text
    numeric_text = ""
    nt=0
    if frame is not None:
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply a smaller Gaussian blur to enhance edges
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Use Canny edge detector with lower thresholds for sharper edges
        edges = cv2.Canny(blurred, 10, 20, L2gradient=True)

        roi_vertices = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)

        # Draw an orange rectangle around the ROI
        roi_highlighted = frame.copy()
        cv2.rectangle(roi_highlighted, (x, y), (x + w, y + h), (0, 165, 255), 2)

        # Mask for the ROI
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, [roi_vertices], 255)
        roi_edges = cv2.bitwise_and(edges, mask)

        # Use HoughLinesP to detect lines in the region of interest
        lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=10)

        # Draw the detected lines on a black image
        line_image = np.zeros_like(frame)
        topmost_line = None
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                # Filter out vertical lines (lines with similar x-coordinates)
                if abs(x1 - x2) > abs(y1 - y2):
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 5)  # Red lines
                    if topmost_line is None or y1 < topmost_line[1]:
                        topmost_line = line[0]

        # Draw the topmost Hough line in a different color
        if topmost_line is not None:
            cv2.line(line_image, (topmost_line[0], topmost_line[1]), (topmost_line[2], topmost_line[3]), (0, 255, 0), 5)

            # Highlight the new ROI in yellow above the top-most Hough line
            roi_height = 40
            wd = 80
            xd = x + 90
            xfd = xd + wd
            # Set the desired height of the ROI
            y_top = topmost_line[1]
            y_start = max(0, y_top - roi_height * 2)
            y_end = min(frame.shape[0], y_top)
            
            if y_start < 680:
                xnd = xd+20
                xfid = xfd+40
                roi_height2 = roi_height + 20
                y_start2 = max(0, y_top - roi_height2 * 2)
                y_end2 = min(frame.shape[0], y_top)
                cv2.rectangle(roi_highlighted, (xnd, y_start2), (xfid, y_end2), (0, 255, 255), 2)
                new_roi = frame[y_start2:y_end2, xnd:xfid]
            
            elif y_start >= 650:
                xd = x + 90
                roi_height = 40
                wd = 80
                xfd = xd + wd
                # Set the desired height of the ROI
                y_top = topmost_line[1]
                y_start = max(0, y_top - roi_height * 2)
                y_end = min(frame.shape[0], y_top)
                cv2.rectangle(roi_highlighted, (xd, y_start), (xfd, y_end), (0, 255, 255), 2)
                new_roi = frame[y_start:y_end, xd:xfd]
            #print(xd, y_start,xfd, y_end, end=" ")

            # Define a new ROI based on the topmost Hough line
            

            # Apply OCR to the new ROI (detect only numbers)
            custom_config = r'--oem 3 --psm 6 outputbase digits'
            text = pytesseract.image_to_string(new_roi, config=custom_config)
            
            # Extract only numeric characters from the OCR result
            numeric_text = ''.join(filter(str.isdigit, text))[:2]
            if numeric_text == '':
                nt = 0
            else:
                nt = int(numeric_text) - 2
            
            # Print OCR result on the Hough lines frame
            cv2.putText(line_image, f'OCR Result: {nt}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(line_image, f'point A: ({xd},{y_start})\npoint B: ({xd+wd},{y_end})\n', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        # Combine the edge frame and the Hough lines frame
        combined_frame = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
        
    

    return frame, edges, roi_highlighted, combined_frame, topmost_line, nt

def send_email(subject, body, sender_email, receiver_email, email_password, smtp_server, smtp_port):


    # Create the email message
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    
    # Attach the body of the email with date and time stamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    body_with_timestamp = f"\n\n{timestamp}"
    message.attach(MIMEText(body_with_timestamp, "plain"))

    # Attach the body of the email
    message.attach(MIMEText(body, "plain"))

    # Create a secure connection with the SMTP server
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(sender_email, email_password)
        server.sendmail(sender_email, receiver_email, message.as_string())
        server.quit()

def main():
    # Replace the camera index or RTSP URL accordingly
    rtsp_url = f'rtsp://admin:preetham77@192.168.0.20:554/cam/realmonitor?channel=1&subtype=0'
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("Error: Could not open RTSP stream.")
        return

    # Set the window names
    #cv2.namedWindow('Original Frame', cv2.WINDOW_NORMAL)
    #cv2.namedWindow('Edges', cv2.WINDOW_NORMAL)
    #cv2.namedWindow('ROI Highlighted', cv2.WINDOW_NORMAL)
    #cv2.namedWindow('Hough Lines', cv2.WINDOW_NORMAL)

    # Set the window sizes
    #cv2.resizeWindow('Original Frame', 960, 540)
    #cv2.resizeWindow('Edges', 960, 540)
    #cv2.resizeWindow('ROI Highlighted', 960, 540)
    #cv2.resizeWindow('Hough Lines', 960, 540)

    # Create a list to store OCR readings
    ocr_readings = []
    recording_duration = 15  # seconds

    # Start time of the recording
    start_time = cv2.getTickCount()


    while True:
        # Capture a frame from the RTSP stream
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Preprocess the frame and draw Hough lines
        original_frame, edges, roi_highlighted, hough_lines, topmost_line, numeric_text = preprocess_frame(frame, x, y, w, h)

        if original_frame is not None:
            # Display the frames separately
            #cv2.imshow('Original Frame', original_frame)
            #cv2.imshow('Edges', edges)
            #cv2.imshow('ROI Highlighted', roi_highlighted)
            #cv2.imshow('Hough Lines', hough_lines)

            # Extract OCR reading for the current frame
            if topmost_line is not None:
                if numeric_text > 28:
                    numeric_text=numeric_text%10 
                    ocr_readings.append(int(numeric_text))
                else:
                    ocr_readings.append(int(numeric_text))
                    
                
            elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            if elapsed_time >= recording_duration:
                print(f"Recording duration of {recording_duration} seconds reached. Exiting.")
                break

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the RTSP stream and close all windows
    cap.release()
    cv2.destroyAllWindows()

    # Find the most recurring number in the OCR readings
    if ocr_readings:
        most_common_number = max(set(ocr_readings))#, key=ocr_readings.count)
        print(f"Most common OCR reading: {most_common_number}")
    
        # Send the most common OCR reading through email
        subject = "OCR Reading of Upper Camera"
        body = f"\n\nThe most common OCR reading is: {most_common_number}cm in BMAX\n"
        send_email(subject, body, sender_email, receiver_email, email_password, smtp_server, smtp_port)
    else:
        print("No OCR readings recorded.")
        subject = "OCR Reading of Upper Camera"
        body = "\n\nNo OCR readings recorded in BMAX\n"
        send_email(subject, body, sender_email, receiver_email, email_password, smtp_server, smtp_port)

if __name__ == "__main__":
    main()


