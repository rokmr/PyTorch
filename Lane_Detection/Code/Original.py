import cv2
import numpy as np
from matplotlib import pyplot as plt

filename = 'original_lane_detection_5.jpg'
frame = cv2.imread(filename)
original=frame.copy()
_height,_width=frame.shape[:2]

hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
cv2.imshow("hls",hls)

#Perspective Transform Parameter
roi_points = np.float32([
            (274, 184),  # Top-left corner
            (0, 337),  # Bottom-left corner
            (575, 337),  # Bottom-right corner
            (371, 184)  # Top-right corner
        ])

padding = int(0.25 * _width)  # padding from side of the image in pixels

desired_roi_points = np.float32([
            [padding, 0],  # Top-left corner
            [padding, _height],  # Bottom-left corner
            [_width - padding, _height],  # Bottom-right corner
            [_width - padding, 0]  # Top-right corner
        ])


# Sliding window parameters
no_of_windows = 10
margin = int((1 / 12) * _width)  # Window width is +/- margin
minpix = int((1 / 24) * _width)  # Min no. of pixels to recenter window

# Light channel Is Threshold
# Relatively light pixels get made white. Dark pixels get made black.
_, sxbinary = cv2.threshold(hls[:, :, 1], 120, 255, cv2.THRESH_BINARY)  # Substitute
# cv2.imshow("im1",sxbinary)

#Gaussian Blur is bused to smooth with kernal 3*3. This step can be avoided.
sxbinary = cv2.GaussianBlur(sxbinary, (3, 3), 0)
# cv2.imshow("im2", sxbinary)

# Canny Edge detection
canny=cv2.Canny(sxbinary, threshold1=120, threshold2=255)
sxbinary=cv2.bitwise_not(canny)
# cv2.imshow("Canny",sxbinary)
# cv2.imshow("binary",canny)

# Binary thresholding on the S (saturation) channel
# A high saturation value means the hue color is pure.
# We expect lane lines to be nice, pure colors (i.e. solid white, yellow)
# and have high saturation channel values.
_, s_binary = cv2.threshold(hls[:, :, 2], 80, 255, cv2.THRESH_BINARY)
# cv2.imshow("s_binary", s_binary)

#Thesh for red channel as yellow and white have rich red in color RGB(255, 255, 0)=yellow, RGB(255, 255, 255)=white
_, r_thresh = cv2.threshold(frame[:, :, 1], 80, 255, cv2.THRESH_BINARY)
# cv2.imshow("r_thresh", r_thresh)

# merging red and saturated image to get
rs_binary = cv2.bitwise_and(s_binary, r_thresh)
# cv2.imshow("rs_binary",rs_binary)

# lane_with_roi
roi_list=[(274, 184),  # Top-left corner
            (0, 337),  # Bottom-left corner
            (575, 337),  # Bottom-right corner
            (371, 184)  # Top-right corner
          ]
cv2.circle(frame,roi_list[0],5,[0,0,255],-1)
cv2.circle(frame,roi_list[1],5,[0,0,255],-1)
cv2.circle(frame,roi_list[2],5,[0,0,255],-1)
cv2.circle(frame,roi_list[3],5,[0,0,255],-1)
# cv2.imshow("Frame",frame)

# To Bird Eye View
_to_bird_eye_matrix=cv2.getPerspectiveTransform(roi_points,desired_roi_points)
_bird_eye_frame=cv2.warpPerspective(rs_binary,_to_bird_eye_matrix,(_width,_height))
(thresh, binary_warped) = cv2.threshold(_bird_eye_frame, 127, 255, cv2.THRESH_BINARY)
warped_frame = binary_warped

warped_copy = warped_frame.copy()
warped_plot = cv2.polylines(warped_copy, np.int32([desired_roi_points]), True, (147, 20, 255), 3)

# cv2.imshow("BirdEye_frame",_bird_eye_frame)
cv2.imshow("Warped_plot",warped_plot)

#Histogram
histogram = np.sum(warped_frame[int(
            warped_frame.shape[0] / 2):, :], axis=0)

 # Draw both the image and the histogram
figure, (ax1, ax2) = plt.subplots(2, 1)  # 2 row, 1 columns
figure.set_size_inches(10, 5)
ax1.imshow(warped_frame, cmap='gray')
ax1.set_title("Warped Binary Frame")
ax2.plot(histogram)
ax2.set_title("Histogram Peaks")
# plt.show()

#Curve Fitting
mid_way=int(histogram.shape[0]/2)
left_way=np.argmax(histogram[:mid_way])
right_way=np.argmax(histogram[mid_way:])+mid_way


#  Got left and right Staring Point
# cv2.circle(warped_frame,(left_way,_height),20,[255],-1)
# cv2.circle(warped_frame,(right_way,_height),20,[255],-1)
# cv2.imshow("New Image",warped_frame)


# Set the height of the sliding windows
window_height = int(warped_frame.shape[0] / no_of_windows)

# Find the x and y coordinates of all the nonzero
# (i.e. white) pixels in the frame.
nonzero = warped_frame.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])

# Store the pixel indices for the left and right lane lines
left_lane_inds = []
right_lane_inds = []

leftx_current = left_way
rightx_current = right_way
frame_sliding_window = warped_frame.copy()

for window in range(no_of_windows):

    # Identify window boundaries in x and y (and right and left)
    win_y_low = warped_frame.shape[0] - (window + 1) * window_height
    win_y_high = warped_frame.shape[0] - window * window_height

    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin


    cv2.rectangle(frame_sliding_window, (win_xleft_low, win_y_low), (
        win_xleft_high, win_y_high), (255, 255, 255), 2)
    cv2.rectangle(frame_sliding_window, (win_xright_low, win_y_low), (
        win_xright_high, win_y_high), (255, 255, 255), 2)

    # : cv2.rectangle(image, start_point, end_point, color, thickness)
    # start_point=represents the top left corner of rectangle
    # end_point=represents the bottom right corner of rectangle
    # cv2.imshow("frame_sliding_window",frame_sliding_window)

    # Identify the nonzero pixels in x and y within the window
    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                      (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                       (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

    # Append these indices to the lists
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)

    # If you found > minpix pixels, recenter next window on mean position

    if len(good_left_inds) > minpix:
        leftx_current = int(np.mean(nonzerox[good_left_inds]))
    if len(good_right_inds) > minpix:
        rightx_current = int(np.mean(nonzerox[good_right_inds]))

cv2.imshow("New Image",frame_sliding_window)
# Concatenate the arrays of indices
left_lane_inds = np.concatenate(left_lane_inds)
right_lane_inds = np.concatenate(right_lane_inds)

# Extract the pixel coordinates for the left and right lane lines
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds]
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds]

# Fit a second order polynomial curve to the pixel coordinates for the left and right lane lines
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

# Create the x and y values to plot on the image
ploty = np.linspace(0, frame_sliding_window.shape[0] - 1, frame_sliding_window.shape[0])
left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

# Generate an image to visualize the result [BGR}
out_img = np.dstack((frame_sliding_window, frame_sliding_window, frame_sliding_window)) * 255

# Add color to the left line pixels and right line pixels
out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

# Plot the figure with the sliding windows
figure, (ax1, ax2, ax3) = plt.subplots(3, 1)  # 3 rows, 1 column
figure.set_size_inches(10, 10)
figure.tight_layout(pad=3.0)
ax1.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
ax2.imshow(frame_sliding_window, cmap='gray')
ax3.imshow(out_img)
ax3.plot(left_fitx, ploty, color='yellow')
ax3.plot(right_fitx, ploty, color='yellow')
ax1.set_title("Original Frame")
ax2.set_title("Warped Frame with Sliding Windows")
ax3.set_title("Detected Lane Lines with Sliding Windows")
# plt.show()

# Generate an image to draw the lane lines on
warp_zero = np.zeros_like(warped_frame).astype(np.uint8)
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
pts_left = np.array([np.transpose(np.vstack([
            left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([
            right_fitx, ploty])))])
pts = np.hstack((pts_left, pts_right))

# Draw lane on the warped blank image
cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 255))

_to_car_eye_matrix=cv2.getPerspectiveTransform(desired_roi_points,roi_points)
_car_eye_frame=cv2.warpPerspective(color_warp,_to_car_eye_matrix,(_width,_height))
blended=cv2.addWeighted(frame, 1, _car_eye_frame, 0.3, 0)

#Plotting
figure, (ax1,ax2,ax3) = plt.subplots(3, 1)
ax1.imshow(original)
ax1.set_title("Original")
ax2.imshow(_car_eye_frame)
ax2.set_title("car_Eye")
ax3.imshow(blended)
ax3.set_title("Blended with Path")
plt.show()

cv2.waitKey(0)