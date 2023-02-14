# AIRS_Special-You
This is an interactive artwork. In the passageway full of people, only "special" person can be shown on the screen while others will disappear magically. This project is divided into two parts, which implement two different functions. Function one is the blurring of pedestrians and function two is the 'disappearance' of pedestrians.
## Pedestrian Blurring
In the section on pedestrian blurring, we mainly use tracking algorithms (SORT, CSRT, etc.) to track all the pedestrians in the video and segment the instances of selected pedestrians based on yolov7 to get the mask of the pedestrians and blur the whole contours of all the pedestrians except for a particular selected clear pedestrian, combining them with a clear background to achieve the effect that visually only one pedestrian in the picture is clear and all others are blurred. In addition, to make the blurring effect less rigid and to blend better with the background, we inflate the masks of the other pedestrians, blurring the inflated parts lightly and blurring the original masks to a higher degree.
<center class="half">
    <img src="https://github.com/Judy-Liang/AIRS_Special-You/blob/main/img_readme/blur_org.jpg" width="200"/><img src="https://github.com/Judy-Liang/AIRS_Special-You/blob/main/img_readme/blur_result.jpg" width="200"/>
</center>
## Pedestrian Disappearing
In the pedestrian disappearance section, similar to the pedestrian blurring section, we still use tracking algorithms (SORT, CSRT, etc.) to track all the pedestrians in the video, and segment the selected pedestrians based on yolov7 to get the mask of the pedestrians, and extract all the outlines of a particular selected line of people, and combine them with a blank background (the first frame of the video where no pedestrians have entered) to achieve the effect that there is only one pedestrian in the picture visually.
## Visual Results
 
